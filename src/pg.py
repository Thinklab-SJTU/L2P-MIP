import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from pyscipopt import Model

from src.networks import *
from src.networks_gcn import *
from src.train_child import *


class PGController:
    def __init__(self, configs):
        self.configs = configs
        self.network = GCN(configs)
        self.opt = optim.Adam(self.network.parameters(), lr=configs["lr"])
        self.value = 0
        self.decay = configs["decay"]
        self.use_ppo = configs["use_ppo"]
        self.eps = configs["eps"]
        self.iter = configs["iter"]
        self.use_adv = configs["use_adv"]
        if configs["cuda"]:
            self.network.to_cuda()
        self.last_action = None

    def get_log_p_from_dist(
        self,
        dis_type,
        dis_prob,
        op_type_sample,
        op_prob_sample,
    ):
        log_p = torch.cat(
            [
                dis_type.log_prob(op_type_sample).unsqueeze(-1),
                dis_prob.log_prob(op_prob_sample).unsqueeze(-1),
            ],
            dim=3,
        ).squeeze(1)
        assert log_p.shape == (
            self.configs["sub_policies"],
            self.configs["sub_policy_ops"],
            2,
        )
        return log_p

    def get_log_p_from_scratch(self, action, Problem):

        if self.configs["cuda"]:
            action = action.cuda()

        #zero = torch.zeros((self.configs["sub_policies"], 1, 1), dtype=torch.float32)
        #if self.configs["cuda"]:
        #    zero = zero.cuda()

        x = self.getInput_GCN(Problem)

        (
            op_type_outputs,
            op_prob_outputs,
        ) = self.network(x)

        dis_type = Categorical(logits=op_type_outputs)
        dis_prob = Categorical(logits=op_prob_outputs)

        action = action.unsqueeze(1)
        op_type_sample = action[:, :, :, 0]
        op_prob_sample = action[:, :, :, 1]

        log_p = self.get_log_p_from_dist(
            dis_type,
            dis_prob,
            op_type_sample,
            op_prob_sample,
        )
        return log_p.cpu()

    def getAB(self, Problem):
        self.model = Model("test")
        self.model.hideOutput()

        self.model.readProblem(Problem)

        self.cons_list = self.model.getConss()
        self.vars_list = self.model.getVars()

        c_ = self.model.getObjective()
        c = np.array([c_[self.vars_list[i]] for i in range(len(self.vars_list))])
        b = []
        b_l = np.zeros((len(self.cons_list),))
        b_r = np.zeros((len(self.cons_list),))
        A = []
        A_ = self.getA()
        for i in range(len(self.cons_list)):
            cons = self.cons_list[i]
            b_l[i] = self.model.getLhs(cons)
            b_r[i] = self.model.getRhs(cons)
            if b_l[i] == -1e20:  # Ax <= b_r
                b.append(b_r[i])
                A.append(A_[i, :])
            elif b_r[i] == 1e20:  # b_l <= Ax --> -Ax <= -b_l
                b.append(-b_l[i])
                A.append(-A_[i, :])
            else:  # both
                b.append(b_r[i])
                A.append(A_[i, :])
                b.append(-b_l[i])
                A.append(-A_[i, :])
        b = np.array(b)
        A = np.array(A)
        
        return A, b

    def getA(self):
        A = np.zeros((len(self.cons_list), len(self.vars_list)))
        for i in range(len(self.cons_list)):
            cons = self.model.getValsLinear(self.cons_list[i])
            # cons_ = self.model.getTransformedCons(self.cons_list[i])
            for j in range(len(self.vars_list)):
                if self.vars_list[j].name in cons:
                    A[i, j] = cons[self.vars_list[j].name]
        return A

    def getInput(self, Problem):
        A, b = self.getAB(Problem)

        A = torch.tensor(A, dtype = torch.float).to("cuda")
        b = torch.tensor(b, dtype = torch.float).to("cuda")

        A = torch.cat([b.unsqueeze(1) , A], 1)


        if A.shape[1] <= 100:
            A = nn.ZeroPad2d(padding=(0, 100 - A.shape[1]))(A)
        else:
            A = torch.split(A, 100, 1)[0]

        return A

    def getInput_GCN(self, Problem):
        self.model = Model("test")
        self.model.hideOutput()
        self.model.readProblem(Problem)

        self.cons_list = self.model.getConss()
        self.vars_list = self.model.getVars()

        c_ = self.model.getObjective()
        c = np.array([c_[self.vars_list[i]] for i in range(len(self.vars_list))])
        b = []
        b_l = np.zeros((len(self.cons_list),))
        b_r = np.zeros((len(self.cons_list),))
        A = []
        A_ = self.getA()
        for i in range(len(self.cons_list)):
            cons = self.cons_list[i]
            b_l[i] = self.model.getLhs(cons)
            b_r[i] = self.model.getRhs(cons)
            if b_l[i] == -1e20:  # Ax <= b_r
                b.append(b_r[i])
                A.append(A_[i, :])
            elif b_r[i] == 1e20:  # b_l <= Ax --> -Ax <= -b_l
                b.append(-b_l[i])
                A.append(-A_[i, :])
            else:  # both
                b.append(b_r[i])
                A.append(A_[i, :])
                b.append(-b_l[i])
                A.append(-A_[i, :])
        b = np.array(b)
        A = np.array(A)

        edge_indices = [[], []]
        n_edges = 0
        for i in range(len(b)):
            for j in range(len(c)):
                if abs(A[i, j]) > 1e-5:
                    edge_indices[0].append(i)
                    edge_indices[1].append(j)
                    n_edges += 1

        edge_features = np.zeros((n_edges, 1))
        for i in range(n_edges):
            edge_features[i, 0] = A[edge_indices[0][i], edge_indices[1][i]]

        constraint_features = np.zeros((len(b), 1))
        for i in range(len(b)):
            constraint_features[i, 0] = b[i]

        variable_features = np.zeros((len(c), 1))
        for i in range(len(c)):
            variable_features[i, 0] = c[i]

        ob = [constraint_features, edge_indices, edge_features, variable_features, len(b), len(c)]
        ob_t = [torch.tensor(ob[0], dtype=torch.float32).to(device),
                torch.tensor(ob[1], dtype=torch.long).to(device),
                torch.tensor(ob[2], dtype=torch.float32).to(device),
                torch.tensor(ob[3], dtype=torch.float32).to(device),
                torch.tensor(ob[4], dtype=torch.long).to(device),
                torch.tensor(ob[5], dtype=torch.long).to(device),
                ]
        return ob_t

    def sample_action(self, Problem):
        
        x = self.getInput_GCN(Problem)

        #zero = torch.zeros((self.configs["sub_policies"], 1, 1), dtype=torch.float32)
        #if self.configs["cuda"]:
        #    zero = zero.cuda()
        (
            op_type_outputs,
            op_prob_outputs,
        ) = self.network(x)

        dis_type = Categorical(logits=op_type_outputs)
        op_type_sample = dis_type.sample()
        dis_prob = Categorical(logits=op_prob_outputs)
        op_prob_sample = dis_prob.sample()

        action = torch.cat(
            [
                op_type_sample.unsqueeze(-1),
                op_prob_sample.unsqueeze(-1),
            ],
            dim=3,
        ).squeeze(1)

        assert action.shape == (
            self.configs["sub_policies"],
            self.configs["sub_policy_ops"],
            2,
        )

        log_p = self.get_log_p_from_dist(
            dis_type,
            dis_prob,
            op_type_sample,
            op_prob_sample,
        )

        return action.cpu(), log_p.cpu()

    def train_one_epoch(self, test_set):
        if(self.configs["cuda"]):
            device = "cuda"
        else:
           device = "cpu"


        #if self.last_action is None:
        #    self.last_action, _ = self.sample_action()

        totol_loss = 0
        totol_reward = 0
        for Problem in (test_set):
            action, old_log_p = self.sample_action(Problem)

            #print("Diff between actions (2-norm): ", torch.sum((action - self.last_action)**2))
            #self.last_action = action

            reward = step(action.float(), self.configs, Problem)
            totol_reward += reward

            if self.value == 0:
                self.value = reward
            adv = reward - self.value if self.use_adv else reward
            old_log_p = torch.sum(old_log_p).detach()

            for it in range(self.iter):
                log_p = self.get_log_p_from_scratch(action, Problem)
                log_p = torch.sum(log_p)
                prob_ratio = torch.exp(log_p - old_log_p.detach())
                loss = -torch.min(
                    prob_ratio * adv,
                    torch.clamp(prob_ratio, 1 - self.eps, 1 + self.eps) * adv,
                )
                totol_loss += loss
                self.opt.zero_grad()
                loss.backward(loss.clone().detach())
                self.opt.step()

        
        #     # Naive policy gradient, for comparison
        #     action, log_p = self.sample_action()
        #     reward, loss_child = step(action, self.configs, test_x, test_y, baseline)
        #     adv = reward - self.value if self.use_adv else reward
        #     log_p = torch.sum(log_p)
        #     loss = -log_p * adv
        #     self.opt.zero_grad()
        #     loss.backward()
        #     self.opt.step()

        self.value = self.value * self.decay + reward * (1 - self.decay)

        return totol_reward, totol_loss