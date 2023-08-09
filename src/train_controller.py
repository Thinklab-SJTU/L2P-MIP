from src.pg import *

import json


def train_controller_main(configs):

    average = 0
    r_list = []
    l_list = []

    test_set = None
    with open('mip_list.json','r') as f :
        test_set = json.load(f)

    controller = PGController(configs)

    print("--- Start Training ---")
    for i in range(configs["epochs"]):
        reward, loss = controller.train_one_epoch(test_set)
        average_reward = 0.95 * average_reward + 0.05 * reward if i > 0 else reward
        average_loss = 0.95 * average_loss + 0.05 * loss if i > 0 else loss
        r_list.append(average_reward)
        l_list.append(average_loss)
        print(
            "Epoch: {}\tReward: {:.2f}\tAverage: {:.2f}\tLoss: {:.2f}\tAverage: {:.2f}".format(
                i, reward, average_reward, loss, average_loss
            )
        )

    return r_list, l_list
