from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import math
from src.train_child import step
import yaml
import json
from multiprocessing import Pool, Queue, Process
import os
import pandas as pd

top_k = 5

# define aim function
# define aim function
def aimFunction(x, Problem, configs, instance_id, process_id):
    # y = x[0] ** 2 + x[1] ** 3 - np.sum(x)
    priority = x[0: 14]
    round = x[14:28]
    time = x[28:]
    action = np.array([[priority[i], round[i], time[i]] for i in range(14)]).reshape(14, 1, 3)
    reward_, t1, t2, nv1, nc1, nv2, nc2, solved_int1, total_int1, solved_int2, total_int2, obj1, obj2, real_obj1, real_obj2, nv1_origin, nc1_origin, nv2_origin, nc2_origin, e_reward1, e_reward2, primal_dual_integral_1, primal_dual_integral_2 = step(
        action,
        configs,
        Problem,
        instance_id,
        process_id)
    int_rate1 = solved_int1 / (total_int1 + 1e-5)
    int_rate2 = solved_int2 / (total_int2 + 1e-5)
    reward1 = (int_rate2 - int_rate1) * 100 * -1
    reward2 = (e_reward2 - e_reward1) * 100
    reward = reward_
    # reward = reward1 + reward2 * 0.1
    # print("reward:", reward, reward1, reward2)
    return reward, reward1, reward2, t1, t2, nv1, nc1, nv2, nc2, solved_int1, total_int1, solved_int2, total_int2, obj1, obj2, real_obj1, real_obj2, nv1_origin, nc1_origin, nv2_origin, nc2_origin, e_reward1, e_reward2, primal_dual_integral_1, primal_dual_integral_2


def get_best_presolves(Problem_set, configs, process_id):
    for i in range(len(Problem_set)):
        # get_best_presolve(Problem_set[i], configs, i, process_id)
        try:
            with open("results/new_reward_results_{}.txt".format(DATASET), 'a') as f:
                print("Try : Process {}, Instance {}, {}".format(process_id, i, Problem_set[i]), file=f)
            get_best_presolve(Problem_set[i], configs, i, process_id)
        except:
            with open("results/new_reward_results_{}.txt".format(DATASET), 'a') as f:
                print("Fail : Process {}, Instance {}, {}".format(process_id, i, Problem_set[i]), file=f)
            pass
    pass


def get_best_presolve(Problem, configs, instance_id, process_id):
    T = 1e5  # initiate temperature
    Tmin = 1e-2  # minimum value of temperature
    alpha = 0.9
    n_x = 42
    x_min = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    x_max = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3,
             3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    # x_d = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    x = [np.random.uniform(low=x_min[j], high=x_max[j]) for j in range(n_x)]  # initiate x
    # x = [6.7863936531085995, 0, 8.860619454983928, 2.496013273530493, 0.28003816228698863, 8.338769797973013, 8.50211187128312, 5.935901331019144, 2.3353672651980597, 4.478722078226046, 0.31684511182357605, 6.984904250635263, 1.455634877123595, 1.1715185018274576]
    k = 50  # times of internal circulation
    t = 0  # time
    x_list = []
    y_list = []
    for j in range(n_x):
        x[j] = min(x[j], x_max[j])
        x[j] = max(x[j], x_min[j])
        if j >= 14:
            x[j] = round(x[j])

    y = aimFunction(x, Problem, configs, instance_id, process_id)[0]  # initiate result

    while T >= Tmin:
        for i in range(k):
            # calculate y
            # generate a new x in the neighborhood of x by transform function
            xNew = np.zeros((n_x,))
            for j in range(n_x):
                xNew[j] = x[j] + np.random.uniform(low=-1, high=1)
            for j in range(n_x):
                xNew[j] = min(xNew[j], x_max[j])
                xNew[j] = max(xNew[j], x_min[j])
                if j >= 14:
                    xNew[j] = round(xNew[j])

            yNew = aimFunction(xNew, Problem, configs, instance_id, process_id)[0]

            y_list.append(yNew)
            x_list.append(xNew)
            if len(x_list) > top_k:
                i = np.argmax(y_list)
                del y_list[i]
                del x_list[i]

            print('y: {}%, yNew: {}%'.format(y, yNew))
            if yNew - y < 0:
                x = xNew
                y = yNew
                break
            else:
                # metropolis principle
                p = math.exp(-(yNew - y) / T)
                r = np.random.uniform(low=0, high=1)
                if r < p:
                    x = xNew
                    y = yNew
        t += 1
        print(t, T)
        T = T * alpha
    with open("results/annealing_results_{}_full.txt".format(DATASET), 'a') as f:
        print("Success : Process {}, Instance {}".format(process_id, instance_id), file=f)
        print(Problem, file=f)
        for i in range(len(x_list)):
            print(x_list[i], file=f)
            print(aimFunction(x_list[i], Problem, configs, instance_id, process_id), file=f)
            if y_list[i] < y:
                y = y_list[i]
                x = x_list[i]
    with open("results/annealing_results_{}.txt".format(DATASET), 'a') as f:
        print("Success : Process {}, Instance {}".format(process_id, instance_id), file=f)
        print(Problem, file=f)
        print(x, file=f)
        print(aimFunction(x, Problem, configs, instance_id, process_id), file=f)


if __name__ == '__main__':
    DATASET = "medium_mik_train"
    DATASET_dir = "data/mik/train/"
    process_num = 20
    Problem_sets = [[] for _ in range(process_num)]

    # load from previous results
    last_run_results = "results/annealing_results_{}.csv".format(DATASET)
    if os.path.exists(last_run_results):
        df = pd.read_csv(last_run_results, header=None)
        solved_data = df[0].tolist()
    else:
        solved_data = []

    # traverse the training data
    list_dirs = os.walk(DATASET_dir)
    count = 0
    for root, ds, fs in list_dirs:
        for f in fs:
            path = os.path.join(DATASET_dir, f)
            check = False
            if path not in solved_data:
                Problem_sets[count % process_num].append(path)
                count = (count + 1)

    print("Hello World! {} Instances to be solved".format(count))
    # with open('experiments/{}.json'.format(DATASET), 'r') as f:
    #     Problem_set = json.load(f)
    #
    config_file = open("src/config.yaml")
    configs = yaml.load(config_file, Loader=yaml.FullLoader)

    # Problem = "data/datasetv1_220522/medium/corlat/train/cor-lat-2f+r-u-10-10-10-5-100-3.300.b342.000000.prune2.lp"
    # get_best_presolves([Problem], configs, 0)

    ps = []
    for i in range(len(Problem_sets)):
        p = Process(target=get_best_presolves, args=(Problem_sets[i], configs, i))
        p.start()
        ps.append(p)
    for i in range(len(Problem_sets)):
        ps[i].join()
