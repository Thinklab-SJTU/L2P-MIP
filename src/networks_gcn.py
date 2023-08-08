import torch
import torch.nn as nn
import torch.nn.functional as F
from src.gcn_model import GCNPolicy


class GCN(nn.Module):

    def __init__(self, configs):
        super(GCN, self).__init__()
        self.configs = configs
        self.gnn_model = GCNPolicy()
        self.op_type, self.op_prob = [], []

        for i in range(configs["sub_policies"]):
            self.op_type.append(
                nn.Linear(64, configs["sub_policy_op_priority"])
            )
            self.op_prob.append(
                nn.Linear(64, configs["sub_policy_op_round"])
            )

    def to_cuda(self):
        self.cuda()
        for i in range(self.configs["sub_policies"]):
            self.op_type[i].cuda()
            self.op_prob[i].cuda()

    def forward(self, A):

        # do LSTM layer
        o = self.gnn_model(A)
        xx = torch.sum(o[0], dim=0, keepdim=True)

        op_type_outputs = torch.zeros((self.configs["sub_policies"], 1, self.configs["sub_policy_ops"], self.configs["sub_policy_op_priority"]), dtype=torch.float32).cuda()
        op_prob_outputs = torch.zeros((self.configs["sub_policies"], 1, self.configs["sub_policy_ops"], self.configs["sub_policy_op_round"]), dtype=torch.float32).cuda()

        for i in range(self.configs["sub_policies"]):
            op_type_outputs[i][0] = self.op_type[i](xx)
            op_prob_outputs[i][0] = self.op_prob[i](xx)


        #op_type_outputs = torch.tensor(op_type_outputs).unsqueeze(2)
        #op_prob_outputs = torch.stack(op_prob_outputs, dim=2)


        # each of the above is of shape (sub_policies, batches, sub_policy_ops, XXX)
        assert op_type_outputs.shape == (
            self.configs["sub_policies"],
            1,
            self.configs["sub_policy_ops"],
            self.configs["sub_policy_op_priority"],
        )

        return op_type_outputs, op_prob_outputs

