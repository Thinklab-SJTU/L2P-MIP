from pyscipopt import Model, quicksum, SCIP_PARAMSETTING
import numpy as np
import ecole

np.set_printoptions(formatter={'float': '{: 0.2f}'.format})


def arg_min(rx, env):
    model = Model(sourceModel=env.model_copy)
    vars_list = model.getVars()
    z = {}
    for i in range(len(vars_list)):
        z[i] = model.addVar()
        # ||rx[vars_list[i]] - vars_list[i]||  abs
        model.addCons(z[i] >= (rx[vars_list[i]] - vars_list[i]))
        model.addCons(z[i] >= -(rx[vars_list[i]] - vars_list[i]))
    if env.arg_min_type == 'sum':
        # sum
        model.setObjective(quicksum(z[i] for i in range(len(vars_list))), clear=True)
    elif env.arg_min_type == 'max':
        # max
        z_m = model.addVar()
        for i in range(len(vars_list)):
            model.addCons(z_m >= z[i])
        model.setObjective(z_m, clear=True)
    else:
        raise NotImplementedError

    vars_list = model.getVars()
    model.optimize()
    res = model.getBestSol()
    sol = [res[vars_list[i]] for i in range(len(vars_list) // 2)]
    return res, sol


def fp_round(x, env):
    x_ = np.array(x)
    tmp = []
    for i in range(len(x)):
        if env.rand:
            x_[i] += np.random.rand() * env.rate - env.rate / 2
        if env.binary_list[env.vars_list[i].name] == 1:
            if x_[i] >= 1:
                x_[i] = 1
            if x_[i] <= 0:
                x_[i] = 0
        if env.continuous_list[env.vars_list[i].name] == 1:
            tmp.append(x[i])
        else:
            tmp.append(int(round(x_[i])))
    return np.array(tmp)


def base_sol(env):
    res_ = env.model.createSol()
    env.model.optimize()
    res = env.model.getBestSol()
    if env.model.getNBestSolsFound() > 0:
        sol = [res[env.vars_list[i]] for i in range(len(env.vars_list))]
        if env.print:
            print(sol)
            print(env.model.getSolObjVal(res))
        sol = fp_round(sol, env)
        if env.print:
            print(sol)
        for i in range(len(sol)):
            # env.model.setSolVal(res, env.vars_list[i], sol[i])
            res_[env.vars_list[i]] = sol[i]
        out = res_, sol, True
    else:
        out = None, None, False
    return out


class Environment:
    def __init__(self):
        self.instance_path = ""
        self.max_iter = 0
        self.model = None
        self.model_copy = None
        # min cx ; s.t. Ax <= b, x in Z
        self.c = None
        self.A_ = None
        self.A = None
        self.b = None
        self.b_l = None
        self.b_r = None
        self.cons_list = None
        self.vars_list = None
        self.continuous_list = {}
        self.binary_list = {}

    def set_instance(self, instance_path, sol_path=None):
        self.model = Model()
        self.model.hideOutput()
        self.model.setPresolve(SCIP_PARAMSETTING.OFF)
        self.model.setParam("presolving/maxrounds", 0)
        self.instance_path = instance_path

        self.model.readProblem(self.instance_path)
        # if sol_path:
        #     self.model.readSol(sol_path)
        # self.cons_list = self.model.getConss()
        # self.vars_list = self.model.getVars()
        # continuous = [1 if self.vars_list[i].vtype() == "CONTINUOUS" else 0 for i in
        #               range(len(self.vars_list))]
        # for i in range(len(self.vars_list)):
        #     self.continuous_list[self.vars_list[i].name] = continuous[i]
        # binary = [1 if self.vars_list[i].vtype() == "BINARY" else 0 for i in
        #           range(len(self.vars_list))]
        # for i in range(len(self.vars_list)):
        #     self.binary_list[self.vars_list[i].name] = binary[i]
        # for i in range(len(self.vars_list)):
        #     self.model.chgVarType(self.vars_list[i], 'C')
        #
        # self.vars_list = self.model.getVars()
        #
        # c_ = self.model.getObjective()
        # self.c = np.array([c_[self.vars_list[i]] for i in range(len(self.vars_list))])
        # self.b = []
        # self.b_l = np.zeros((len(self.cons_list),))
        # self.b_r = np.zeros((len(self.cons_list),))
        # self.A = []
        # self.A_ = self.getA()
        # for i in range(len(self.cons_list)):
        #     cons = self.cons_list[i]
        #     self.b_l[i] = self.model.getLhs(cons)
        #     self.b_r[i] = self.model.getRhs(cons)
        #     if self.b_l[i] == -1e20:  # Ax <= b_r
        #         self.b.append(self.b_r[i])
        #         self.A.append(self.A_[i, :])
        #     elif self.b_r[i] == 1e20:  # b_l <= Ax --> -Ax <= -b_l
        #         self.b.append(-self.b_l[i])
        #         self.A.append(-self.A_[i, :])
        #     else:  # both
        #         self.b.append(self.b_r[i])
        #         self.A.append(self.A_[i, :])
        #         self.b.append(-self.b_l[i])
        #         self.A.append(-self.A_[i, :])
        # self.b = np.array(self.b)
        # self.A = np.array(self.A)
        # # self.b_0 = np.zeros((100,))
        # # self.b_0[0:len(self.cons_list)] = self.b
        # # self.A_0 = np.zeros((100, len(self.vars_list)))
        # # self.A_0[0:len(self.cons_list)] = self.A
        # # self.b = self.b_0
        # # self.A = self.A_0
        # for i in range(len(self.vars_list)):
        #     if self.binary_list[self.vars_list[i].name] == 1:
        #         self.model.addCons(0 <= self.vars_list[i])
        #         self.model.addCons(self.vars_list[i] <= 1)
        # self.model_copy = Model(sourceModel=self.model)
        #
        # # self.model.presolve()
        # self.max_iter = len(self.c) * 5
        # pass

    def getA(self):
        A = np.zeros((len(self.cons_list), len(self.vars_list)))
        for i in range(len(self.cons_list)):
            cons = self.model.getValsLinear(self.cons_list[i])
            # cons_ = self.model.getTransformedCons(self.cons_list[i])
            for j in range(len(self.vars_list)):
                if self.vars_list[j].name in cons:
                    A[i, j] = cons[self.vars_list[j].name]
        return A


class MIPEnvironment(Environment):
    def __init__(self):
        super(MIPEnvironment, self).__init__()
        self.max_iter = 0
        self.iter = 0
        self.x_i = None  # integer solution
        self.x_c = None  # feasible solution
        self.sol_c = None
        self.sol_i = None
        self.best_sol = []
        self.arg_min_type = 'sum'  # sum or max
        self.print = False
        self.rand = 0
        self.rate = 0
        self.ob_dims = -1
        self.action_dims = -1
        self.binary_action = False
        self.ecole_env = ecole.environment.Branching(
            reward_function=-1.5 * ecole.reward.LpIterations() ** 2,
            observation_function=ecole.observation.NodeBipartite(),
        )

    def set_instance(self, instance_path, sol_path=None):
        super(MIPEnvironment, self).set_instance(instance_path, sol_path)
        # # x_i: integer solution by SCIP; sol_i: integer solution by Numpy;
        # # x_c: feasible solution by SCIP; sol_c: feasible solution by Numpy;
        # self.x_i, self.sol_i, stat = base_sol(self)
        # self.x_c, self.sol_c = arg_min(self.x_i, self)
        # self.ob_dims = len(self.b) * len(self.vars_list) + len(self.b) + len(self.vars_list) + len(self.sol_i) * 2
        # self.action_dims = len(self.sol_i)

    def reset(self):
        self.iter = 0
        self.x_c = None  # feasible solution
        self.x_i = None  # integer solution
        self.sol_c = None
        self.sol_i = None
        self.best_sol = []
        self.set_instance(self.instance_path)
        return self.get_observation()

    def calc_violation(self, sol):
        v = np.matmul(self.A, sol) - self.b
        v = [max(v[i], 0) for i in range(len(v))]
        # print(-np.sum(v))
        return -np.sum(v)

    def get_observation(self):
        obs, action_set, reward_offset, done, info = self.ecole_env.reset(self.instance_path)
        return obs

    def get_best_sol(self):
        opt = 0 if len(self.best_sol) == 0 else np.dot(self.best_sol, self.c)
        return self.best_sol, opt

    def get_gap(self):
        gap = np.zeros(len(self.vars_list))
        for i in range(len(self.vars_list)):
            if self.continuous_list[self.vars_list[i].name] == 1:
                continue
            else:
                gap[i] = abs(self.sol_i[i] - self.sol_c[i]) + 1e-5
        gap = gap / np.sum(gap)
        return gap

    def check(self, action):
        if self.continuous_list[self.vars_list[action].name] == 1:
            return False
        else:
            return True
