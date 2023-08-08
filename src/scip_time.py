import random
import copy
from pyscipopt import Model
from pyscipopt import Conshdlr
import pyscipopt
from multiprocessing import Pool, Queue, Process


#x = model.addVar("x")
#y = model.addVar("y", vtype="INTEGER")
#model.setObjective(x + y)
#model.addCons(2*x - y*y >= 0)
#model.optimize()

param_list = ["presolving/stuffing/","presolving/dualsparsify/","presolving/sparsify/",
              "presolving/tworowbnd/","presolving/redvub/","presolving/implics/",
              "presolving/dualinfer/", "presolving/dualagg/", "presolving/domcol/", #neg
              "presolving/gateextraction/", "presolving/boundshift/", "presolving/convertinttobin/",
              "presolving/inttobinary/", "presolving/trivial/", 
              ]
def get_var_origin(Problem, model1):
    # model = Model("test")
    # model.hideOutput()
    # model.setLogfile("log_original.txt")
    # model.readProblem(Problem)

    # model.presolve()
    model = Model(sourceModel=model1)

    vars_list = model.getVars()
    continuous_list = {}
    binary_list = {}

    continuous = [1 if vars_list[i].vtype() == "CONTINUOUS" else 0 for i in
                  range(len(vars_list))]
    for i in range(len(vars_list)):
        continuous_list[vars_list[i].name] = continuous[i]

    binary = [1 if vars_list[i].vtype() == "BINARY" else 0 for i in
              range(len(vars_list))]
    for i in range(len(vars_list)):
        binary_list[vars_list[i].name] = binary[i]

    for i in range(len(vars_list)):
        model.chgVarType(vars_list[i], 'C')
    for i in range(len(vars_list)):
        if binary_list[vars_list[i].name] == 1:
            model.addCons(0 <= vars_list[i])
            model.addCons(vars_list[i] <= 1)

    model.optimize()
    res = model.getBestSol()
    sol = [res[vars_list[i]] for i in range(len(vars_list))]
    total_int = 0
    solved_int = 0
    for i in range(len(sol)):
        if continuous_list[vars_list[i].name] == 0:
            total_int += 1
            if abs(int(sol[i]) - sol[i]) < 1e-5:
                solved_int += 1
    return solved_int, total_int

def get_var(Problem, configs, action, model2):
    # model = Model("test")
    # model.hideOutput()
    # model.setLogfile("log_original.txt")
    # model.readProblem(Problem)
    #
    # for i in range(configs["sub_policies"]):
    #
    #     priority = action[i][0][0]
    #     round = int(action[i][0][1])
    #     time = int(action[i][0][2])
    #
    #     model.setParam(param_list[i] + "priority", priority)
    #     model.setParam(param_list[i] + "maxrounds", round)
    #     model.setParam(param_list[i] + "timing", time)

    # model.presolve()
    model = Model(sourceModel=model2)

    vars_list = model.getVars()
    continuous_list = {}
    binary_list = {}

    continuous = [1 if vars_list[i].vtype() == "CONTINUOUS" else 0 for i in
                  range(len(vars_list))]
    for i in range(len(vars_list)):
        continuous_list[vars_list[i].name] = continuous[i]

    binary = [1 if vars_list[i].vtype() == "BINARY" else 0 for i in
              range(len(vars_list))]
    for i in range(len(vars_list)):
        binary_list[vars_list[i].name] = binary[i]

    for i in range(len(vars_list)):
        model.chgVarType(vars_list[i], 'C')
    for i in range(len(vars_list)):
        if binary_list[vars_list[i].name] == 1:
            model.addCons(0 <= vars_list[i])
            model.addCons(vars_list[i] <= 1)

    model.optimize()

    res = model.getBestSol()
    sol = [res[vars_list[i]] for i in range(len(vars_list))]
    total_int = 0
    solved_int = 0
    for i in range(len(sol)):
        if continuous_list[vars_list[i].name] == 0:
            total_int += 1
            if abs(int(sol[i]) - sol[i]) < 1e-5:
                solved_int += 1
    return solved_int, total_int

def get_time_origin(Problem, q=Queue()):
    model = Model("test")
    model.hideOutput()
    model.setLogfile("log_original.txt")
    model.readProblem(Problem)

    T = model.getTotalTime()
    model.optimize()
    T = model.getTotalTime() - T
    nv = model.getNVars()
    nc = model.getNConss()
    solved_int, total_int = get_var_origin(Problem, model)
    q.put(('origin', [T, nv, nc, solved_int, total_int]))
    return T, nv, nc, solved_int, total_int
def get_time(Problem, configs, action, q=Queue()):
    model = Model("test")
    model.hideOutput()
    model.setLogfile("log_optimized.txt")

    model.readProblem(Problem)
    model.setParam("limits/time", 600)
    for i in range(configs["sub_policies"]):

        priority = action[i][0][0]
        round = int(action[i][0][1])
        time = int(action[i][0][2])

        model.setParam(param_list[i] + "priority", priority)
        model.setParam(param_list[i] + "maxrounds", round)
        model.setParam(param_list[i] + "timing", time)

    T = model.getTotalTime()
    model.optimize()
    T = model.getTotalTime() - T
    nv = model.getNVars()
    nc = model.getNConss()
    solved_int, total_int = get_var(Problem, configs, action, model)
    q.put(('optimized', [T, nv, nc, solved_int, total_int]))
    return T, nv, nc, solved_int, total_int

if __name__ == "__main__":
    Random_delCons("./pk1.mps", 1)