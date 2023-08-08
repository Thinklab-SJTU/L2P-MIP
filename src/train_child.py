from absl import app
from easydict import EasyDict
import numpy as np
import random
import torch
import torchvision
import os
import yaml
from multiprocessing import Pool, Queue, Process
from src.networks import LSTMValue, LSTMPolicy
from src.utils import discrete2interval, discrete2bool, set_seed
from src.scip_time import get_time, get_time_origin, get_var, get_var_origin
import copy
from tqdm import tqdm

device = "cuda"

origin_time = {}

def step(original_action, configs, Problem):


    # action = original_action.clone()
    action = copy.deepcopy(original_action)

    for i in range(configs["sub_policies"]):
        for j in range(configs["sub_policy_ops"]):

            priority = action[i][j][0]
            round = action[i][j][1]
            time = action[i][j][2]

            # typ = 0  # XXX
            #print(priority, round)
            if(i < configs["sub_policies_neg"]):
                priority = discrete2interval(priority, configs["sub_policy_op_priority"], -1000000, -1)
            else:
                priority = discrete2interval(priority, configs["sub_policy_op_priority"], 1, 1000000)
            round = round - 1
            time = 2 ** (time + 2)

            # action[i][j][0] = typ
            # print(priority, round)

            action[i][j][0] = priority
            action[i][j][1] = round
            action[i][j][2] = time

    #print(action)
    # global origin_time
    # if(Problem not in origin_time):
    #    origin_time[Problem] = get_time_origin(Problem)
    #    print("origin_time[", Problem , "]: ", origin_time[Problem])
    q = Queue()
    p1 = Process(target=get_time_origin, args=(Problem, q))
    p2 = Process(target=get_time, args=(Problem, configs, action, q))
    p1.start()
    p2.start()
    p1.join()
    p2.join()

    for _ in range(2):
        t = q.get()
        if t[0] == 'origin':
            origin_time[Problem] = float(t[1][0])
            nv1 = int(t[1][1])
            nc1 = int(t[1][2])
            solved_int1 = int(t[1][3])
            total_int1 = int(t[1][4])
        else:
            time = float(t[1][0])
            nv2 = int(t[1][1])
            nc2 = int(t[1][2])
            solved_int2 = int(t[1][3])
            total_int2 = int(t[1][4])


    # origin_time[Problem], nv1, nc1 = get_time_origin(Problem)
    # time, nv2, nc2 = get_time(Problem, configs, action)
    # solved_int1, total_int1 = get_var_origin(Problem, model1)
    # solved_int2, total_int2 = get_var(Problem, configs, action, model2)
    print("origin[", Problem, "]: ", origin_time[Problem], nv1, nc1, solved_int1, total_int1)
    print("optimized[", Problem, "]: ", time, nv2, nc2, solved_int2, total_int2)
    return 100 * (origin_time[Problem] - time) / origin_time[Problem], origin_time[Problem], time, nv1, nc1, nv2, nc2, solved_int1, total_int1, solved_int2, total_int2


if __name__ == "__main__":
    get_init_acc()
