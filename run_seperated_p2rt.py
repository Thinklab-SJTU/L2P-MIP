import os
from listMLE import listMLE
import argparse
import json
from gcn_model_seperated_p2rt import GCNPolicy
import torch
import numpy as np
from memory import Memory
import logging
import copy
from src.train_child import step
import yaml
from datetime import datetime
import pickle

# TEST_CASE = "priority_mse_round_time_seperated_p2rt_single_layer+_no_grad_with_dynamic"
# TEST_CASE: combination of {priority_mse, priority_rank, priority_list, round, time, label}
# priority_mse/priority_rank/priority_list for evaluating priority with three different loss;
# round for evaluating round, time for evaluating time, label for evaluating the ground truth;
# can use combination: priority_mse_round_time for evaluating all three parts;

# load_dir = "priority_mse_round_time_seperated_p2rt_single_layer+_no_grad_with_dynamic"

priority_weight = 1
round_weight = 1
time_weight = 1
# weight for summing up the three losses

epoch_num = 100000
test_freq = 1000

validation_split = True
# if true, use the validation set for evaluation; if false, choose several instance in the train set for evaluation.

test_number = 5
# if validation_split is false, use $test_number of instances per batch for evaluation

batch_size = 8
learning_rate = 1e-4

emb_size = 64


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


def test_presolve(names, preds):
    assert len(names) == len(preds)
    n = len(names) if validation_split else test_number
    results_0 = []
    results_1 = []
    results_2 = []
    config_file = open("src/config.yaml")
    configs = yaml.load(config_file, Loader=yaml.FullLoader)
    if args.large:
        configs['time_limit'] = 60
    same_count = 0
    better_count = 0
    for i in range(n):
        # preds[i][0: 14] = preds[i][0: 14] * 10
        # round_class = 4
        # preds[i][14: 28] = [np.argmax(preds[i][14 + j * round_class: 14 + (j + 1) * round_class]) for j in range(14)]
        # start = 14 + 14 * round_class
        # assert start == 70
        # time_class = 4
        # preds[i][28: 42] = [np.argmax(preds[i][start + j * time_class: start + (j + 1) * time_class]) for j in
        #                     range(14)]
        result = aimFunction(preds[i][0:42], names[i], configs, 0, 0)
        if result[5] == result[7] and result[6] == result[8] and result[9] == result[11] and result[10] == result[12]:
            delta_time = 0.0
            delta_rate = 0.0
            same_count += 1
        else:
            if not args.large:
                delta_time = result[3] - result[4]
                delta_rate = (result[3] - result[4]) / result[3]
                if delta_time > 0:
                    better_count += 1
            else:
                delta_time = result[23] - result[24]
                delta_rate = (result[23] - result[24]) / result[23]
                if delta_time > 0:
                    better_count += 1
        if not args.large:
            results_0.append(result[3])  # origin_time
        else:
            results_0.append(result[23])
        results_1.append(delta_time)
        results_2.append(delta_rate)
        with open(log_path + ".txt", 'a') as f:
            print(names[i], file=f)
            print(preds[i][0:42], file=f)
            print(result, file=f)

    return np.sum(results_0), np.sum(results_1), np.sum(results_2), same_count, better_count


def rank(preds, labels):
    length = len(preds[0]) * len(preds[0])
    assert length == 14 * 14
    preds_new_1 = torch.zeros([len(preds), length], device=device)
    preds_new_2 = torch.zeros([len(preds), length], device=device)
    labels_new = torch.zeros([len(preds), length], device=device)
    for i in range(len(preds)):
        for j in range(length):
            x1 = j // len(preds[0])
            x2 = j % len(preds[0])
            preds_new_1[i][j] = preds[i][x1]
            preds_new_2[i][j] = preds[i][x2]
            if labels[i][x1] > labels[i][x2]:
                labels_new[i][j] = 1
            elif labels[i][x1] < labels[i][x2]:
                labels_new[i][j] = -1
            else:
                preds_new_1[i][j] = preds_new_2[i][j] + 1
                labels_new[i][j] = 1
    rank_loss = torch.nn.MarginRankingLoss()
    return rank_loss(preds_new_1, preds_new_2, labels_new)


def calc_inputs(mini_batch):
    inputs = []
    labels_priority = []
    labels_round = []
    labels_time = []
    names = []
    sum_constraint = 0
    sum_variable = 0
    # sizes = []
    for i in range(len(mini_batch)):
        # sizes.append(mini_batch[i][0][5])
        name = mini_batch[i][2]
        names.append(name)
        # if name not in instances:
        #     instances[name] = 1
        # else:
        #     instances[name] += 1
        if len(inputs) == 0:
            inputs = copy.deepcopy(mini_batch[i][0])
            labels_priority = [mini_batch[i][1][0: 14]]
            labels_round = [mini_batch[i][1][14: 28]]
            labels_time = [mini_batch[i][1][28: 42]]
            inputs[1] = np.array(inputs[1])
            sum_constraint = inputs[4]
            sum_variable = inputs[5]
            inputs[4] = np.array([inputs[4]])
            inputs[5] = np.array([inputs[5]])
        else:
            inputs[0] = np.concatenate((inputs[0], mini_batch[i][0][0]), axis=0)
            edge_indices = copy.deepcopy(mini_batch[i][0][1])
            for x in range(len(edge_indices[0])):
                edge_indices[0][x] += sum_constraint
            for x in range(len(edge_indices[1])):
                edge_indices[1][x] += sum_variable
            inputs[1] = np.concatenate((inputs[1], edge_indices), axis=1)
            inputs[2] = np.concatenate((inputs[2], mini_batch[i][0][2]), axis=0)
            inputs[3] = np.concatenate((inputs[3], mini_batch[i][0][3]), axis=0)
            sum_constraint += mini_batch[i][0][4]
            sum_variable += mini_batch[i][0][5]
            inputs[4] = np.concatenate((inputs[4], [mini_batch[i][0][4]]), axis=0)
            inputs[5] = np.concatenate((inputs[5], [mini_batch[i][0][5]]), axis=0)
            labels_priority = np.concatenate((labels_priority, [mini_batch[i][1][0: 14]]), axis=0)
            labels_round = np.concatenate((labels_round, [mini_batch[i][1][14: 28]]), axis=0)
            labels_time = np.concatenate((labels_time, [mini_batch[i][1][28: 42]]), axis=0)
    inputs_tensor = [torch.tensor(inputs[0], dtype=torch.float32).to(device),
                     torch.tensor(inputs[1], dtype=torch.long).to(device),
                     torch.tensor(inputs[2], dtype=torch.float32).to(device),
                     torch.tensor(inputs[3], dtype=torch.float32).to(device),
                     torch.tensor(inputs[4], dtype=torch.long).to(device),
                     torch.tensor(inputs[5], dtype=torch.long).to(device),
                     ]
    labels_priority_tensor = torch.tensor(labels_priority, dtype=torch.float32).to(device)
    labels_round_tensor = torch.tensor(labels_round, dtype=torch.long).to(device)
    labels_time_tensor = torch.tensor(labels_time, dtype=torch.long).to(device)
    return names, inputs_tensor, labels_priority_tensor, labels_round_tensor, labels_time_tensor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=0, type=int, help="gpu id")
    parser.add_argument("--mode", default=1, type=int, help="mode 0: 14 + 64; mode 1: 64 + 64")
    parser.add_argument("--no_grad", default=0, type=int,
                        help="no grad to connection of the priority part")  # default setting is normal gradient
    parser.add_argument("--single_layer", default=0, type=int,
                        help="change the embedding to single layer")  # default setting is double layer embedding
    parser.add_argument("--dynamic", default=1, type=int, help="use dynamic weight of the three loss")
    parser.add_argument("--dataset", default="corlat")
    parser.add_argument("--train_dir", default="generated_data/hard_nips_anonymous_train.pkl")
    parser.add_argument("--test_dir", default="generated_data/hard_nips_anonymous_validation.pkl")
    parser.add_argument("--test_case", default="priority_mse_round_time", help="the test case")
    # TEST_CASE: combination of {priority_mse, priority_rank, priority_list, round, time, label}
    # priority_mse/priority_rank/priority_list for evaluating priority with three different loss;
    # round for evaluating round, time for evaluating time, label for evaluating the ground truth;
    # can use combination: priority_mse_round_time for evaluating all three parts;
    # free to add more details of this run
    parser.add_argument("--large", default=0, type=int,
                        help="if the dataset is large, set time limit to 30 and use primal dual integral")
    parser.add_argument("--load", default=0, type=int)
    parser.add_argument("--load_dir", default="")
    args = parser.parse_args()


    TEST_CASE = "{}-dataset_{}-mode_{}-no_grad_{}-single_layer_{}-dynamic_{}".format(args.test_case, args.dataset,
                                                                                     args.mode, args.no_grad,
                                                                                     args.single_layer, args.dynamic)
    gpu_id = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log_dir = "log/" + TEST_CASE + "/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    log_path = os.path.join(log_dir, datetime.now().strftime('%Y%m%d-%H%M%S'))
    fh = logging.FileHandler(log_path + ".log")
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    # print args
    logger.info(TEST_CASE)
    logger.info(gpu_id)
    logger.info(epoch_num)
    logger.info(test_freq)
    logger.info(validation_split)
    logger.info(test_number)
    logger.info(batch_size)
    logger.info(learning_rate)
    logger.info(priority_weight)
    logger.info(round_weight)
    logger.info(time_weight)
    logger.info(emb_size)

    train_data_path = args.train_dir
    validation_data_path = args.test_dir

    torch.nn.MarginRankingLoss()
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    device = torch.device("cuda")

    net = GCNPolicy(args.mode, args.no_grad, args.single_layer, emb_size).cuda()

    if args.load > 0:
        if args.load_dir == "":
            load_dir = 'model/{}/'.format(TEST_CASE)
        else:
            load_dir = args.load_dir
        net.load_state_dict(torch.load(load_dir + 'param_{}_{}.dat'.format(args.load, TEST_CASE)), strict=False)
    # net.load_state_dict(torch.load('model/param_{}_{}.dat'.format(14000, TEST_CASE)))

    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logger.info('Total: {}; Trainable: {}'.format(total_num, trainable_num))
    # torch.save(net.state_dict(), 'model/param_{}.dat'.format(0))
    # net.load_state_dict(torch.load('model/param_{}.dat'.format(0)))
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    with open(train_data_path, "rb") as f:
        # train_data = json.load(f)
        train_data = json.load(f)
        # train_data = list(ijson.items(f, 'data.item'))
        train_memory = Memory(len(train_data))
        for data in train_data:
            label = []
            for i in range(len(data["label"])):
                if i < 14:
                    label.append(data["label"][i] / 10)
                else:
                    label.extend([1 if j == data["label"][i] else 0 for j in range(4)])
            label = data["label"]
            train_memory.add((data["feature"], label, data["name"]))

    with open(validation_data_path, "rb") as f:
        # validation_data = json.load(f)
        validation_data = json.load(f)
        validation_memory = Memory(len(validation_data))
        for data in validation_data:
            label = []
            for i in range(len(data["label"])):
                if i < 14:
                    label.append(data["label"][i] / 10)
                else:
                    label.extend([1 if j == data["label"][i] else 0 for j in range(4)])
            label = data["label"]
            validation_memory.add((data["feature"], label, data["name"]))
    old_loss_1 = 0
    old_loss_2 = 0
    old_loss_3 = 0
    for e in range(epoch_num):
        # instances = {}
        test_size = 0
        total_loss = 0.0
        total_loss_1 = 0.0
        total_loss_2 = 0.0
        total_loss_3 = 0.0
        total_origin_time = 0.0
        total_delta_time = 0.0
        total_delta_rate = 0.0
        total_same_count = 0
        total_better_count = 0
        for _ in range(len(train_data) // batch_size + 1):
            mini_batch = train_memory.sample(batch_size)
            names, inputs_tensor, labels_priority_tensor, labels_round_tensor, labels_time_tensor = calc_inputs(
                mini_batch)
            labels_priority = labels_priority_tensor.cpu().detach().numpy()
            labels_round = labels_round_tensor.cpu().detach().numpy()
            labels_time = labels_time_tensor.cpu().detach().numpy()
            preds_tensor = net(inputs_tensor)
            # vise = make_dot(preds_tensor, params=dict(net.named_parameters()))
            # vise.view('model_structure.pdf')
            # exit()
            # preds_tensor = labels_tensor
            optimizer.zero_grad()

            # priority
            # preds_priority = torch.nn.Sigmoid()(preds_tensor[:, 0: 14]) * 10
            preds_priority = preds_tensor[:, 0: 14]

            if 'priority_mse' in TEST_CASE:
                loss1 = torch.nn.MSELoss()(preds_priority, labels_priority_tensor)
            elif 'priority_rank' in TEST_CASE:
                loss1 = rank(preds_priority, labels_priority_tensor)
            elif 'priority_list' in TEST_CASE:
                loss1 = listMLE(preds_priority, labels_priority_tensor)
            else:
                loss1 = torch.nn.MSELoss()(preds_priority, labels_priority_tensor)
                loss1 = loss1 - loss1

            # loss1.backward()
            loss_1 = loss1.item()

            gamma = 1
            loss_func = torch.nn.CrossEntropyLoss()
            # round
            round_class = 4
            if 'round' in TEST_CASE:
                for i in range(14):
                    if i == 0:
                        loss2 = loss_func(preds_tensor[:, 14 + i * round_class: 14 + (i + 1) * round_class],
                                          labels_round_tensor[:, i])
                    else:
                        loss2 += loss_func(preds_tensor[:, 14 + i * round_class: 14 + (i + 1) * round_class],
                                           labels_round_tensor[:, i])
            else:
                loss2 = loss1

            # loss2.backward()
            loss_2 = loss2.item()

            # time
            start = 14 + 14 * round_class
            assert start == 70
            time_class = 4
            if 'time' in TEST_CASE:
                for i in range(14):
                    if i == 0:
                        loss3 = loss_func(preds_tensor[:, start + i * time_class: start + (i + 1) * time_class],
                                          labels_time_tensor[:, i])
                    else:
                        loss3 += loss_func(preds_tensor[:, start + i * time_class: start + (i + 1) * time_class],
                                           labels_time_tensor[:, i])
            else:
                loss3 = loss1
            # loss3.backward()
            loss_3 = loss3.item()

            total_loss_1 += loss_1 * len(mini_batch)
            total_loss_2 += loss_2 * len(mini_batch)
            total_loss_3 += loss_3 * len(mini_batch)

            loss = loss1 * priority_weight + loss2 * round_weight + loss3 * time_weight
            loss.backward()
            optimizer.step()
            # evaluation on the samples of the train set
            if (e % test_freq == test_freq - 1 or args.test_case == 'label') and not validation_split:
                preds = preds_tensor.detach().cpu().numpy()
                for i in range(len(preds)):
                    if 'priority' in TEST_CASE:
                        preds[i][0: 14] = preds_priority[i].detach().cpu().numpy()
                    else:
                        preds[i][0: 14] = labels_priority[i]

                    round_class = 4
                    if 'round' in TEST_CASE:
                        preds[i][14: 28] = [np.argmax(preds[i][14 + j * round_class: 14 + (j + 1) * round_class]) for j
                                            in range(14)]
                    else:
                        preds[i][14: 28] = labels_round[i]

                    start = 14 + 14 * round_class
                    assert start == 70
                    time_class = 4
                    if 'time' in TEST_CASE:
                        preds[i][28: 42] = [np.argmax(preds[i][start + j * time_class: start + (j + 1) * time_class])
                                            for j
                                            in range(14)]
                    else:
                        preds[i][28: 42] = labels_time[i]

                origin_time, delta_time, delta_rate, same_count, better_count = test_presolve(names, preds)
                total_origin_time += origin_time
                total_delta_time += delta_time
                total_delta_rate += delta_rate
                total_same_count += same_count
                total_better_count += better_count
                test_size += test_number
        total_loss += total_loss_1 * priority_weight + total_loss_2 * round_weight + total_loss_3 * time_weight
        logger.info(
            "epoch: {}/{}, loss: {:2f}/{:2f},{:2f},{:2f}, weight: {:2f},{:2f},{:2f}".format(e, epoch_num, total_loss,
                                                                                            total_loss_1 * priority_weight,
                                                                                            total_loss_2 * round_weight,
                                                                                            total_loss_3 * time_weight,
                                                                                            priority_weight,
                                                                                            round_weight,
                                                                                            time_weight))
        if args.dynamic and old_loss_1 != 0:
            T = 1
            r1 = total_loss_1 / old_loss_1 / T
            r2 = total_loss_2 / old_loss_2 / T
            r3 = total_loss_3 / old_loss_3 / T
            total = np.exp(r1) + np.exp(r2) + np.exp(r3)
            priority_weight = np.exp(r1) / total * 3
            round_weight = np.exp(r2) / total * 3
            time_weight = np.exp(r3) / total * 3
        old_loss_1 = total_loss_1
        old_loss_2 = total_loss_2
        old_loss_3 = total_loss_3

        if e % test_freq == test_freq - 1 or args.test_case == 'label':
            save_dir = 'model/{}/'.format(TEST_CASE)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(net.state_dict(), save_dir + 'param_{}_{}.dat'.format(e, TEST_CASE))
            # net.load_state_dict(torch.load('model/param_{}.dat'.format(e)))
            # evaluation on the validation set
            if validation_split:
                total_origin_time = 0.0
                total_delta_time = 0.0
                total_delta_rate = 0.0
                total_same_count = 0
                total_better_count = 0
                mini_batch = validation_memory.sample(len(validation_data))
                names, inputs_tensor, labels_priority_tensor, labels_round_tensor, labels_time_tensor = calc_inputs(
                    mini_batch)
                labels_priority = labels_priority_tensor.cpu().detach().numpy()
                labels_round = labels_round_tensor.cpu().detach().numpy()
                labels_time = labels_time_tensor.cpu().detach().numpy()
                preds_tensor = net(inputs_tensor)
                preds_priority = preds_tensor[:, 0: 14]
                preds = preds_tensor.detach().cpu().numpy()
                for i in range(len(preds)):
                    if 'priority' in TEST_CASE:
                        preds[i][0: 14] = preds_priority[i].detach().cpu().numpy()
                    else:
                        preds[i][0: 14] = labels_priority[i]

                    round_class = 4
                    if 'round' in TEST_CASE:
                        preds[i][14: 28] = [np.argmax(preds[i][14 + j * round_class: 14 + (j + 1) * round_class]) for j
                                            in range(14)]
                    else:
                        preds[i][14: 28] = labels_round[i]

                    start = 14 + 14 * round_class
                    assert start == 70
                    time_class = 4
                    if 'time' in TEST_CASE:
                        preds[i][28: 42] = [np.argmax(preds[i][start + j * time_class: start + (j + 1) * time_class])
                                            for j
                                            in range(14)]
                    else:
                        preds[i][28: 42] = labels_time[i]

                origin_time, delta_time, delta_rate, same_count, better_count = test_presolve(names, preds)
                total_origin_time = origin_time
                total_delta_time = delta_time
                total_delta_rate = delta_rate
                total_same_count = same_count
                total_better_count = better_count
                test_size = len(preds)
            logger.info(
                "testing presolve, delta time: {:2f}%/{:2f}/{:2f}, delta rate: {:2f}%, same/better/total: {}/{}/{}".format(
                    total_delta_time / total_origin_time * 100, total_delta_time,
                    total_origin_time,
                    total_delta_rate * 100 / test_size,
                    total_same_count,
                    total_better_count,
                    test_size))
            if args.test_case == 'label':
                break
