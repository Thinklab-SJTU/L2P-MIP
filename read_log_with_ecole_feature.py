import numpy as np
import os
import gzip
import pickle
import csv
import json
# from dqn_env import MIPEnvironment

if __name__ == '__main__':
    feature_mapping = {}
    DATASET = 'hard_nips_anonymous'
    ecole_feature_dir = "ecole_features/anonymous/train/"
    name = 'results/reward3_annealing_results_{}.txt'.format(DATASET)
    list_dirs = os.walk(ecole_feature_dir)
    count = 0
    for root, ds, fs in list_dirs:
        for f in fs:
            path = os.path.join(ecole_feature_dir, f)
            if not os.path.exists(path):
                continue
            with gzip.open(path, 'rb') as f:
                data = pickle.load(f)
                instance_id = data['instance'].name
                if instance_id not in feature_mapping:
                    feature_mapping[instance_id] = path

    f = open(name)
    line = f.readline()
    reward = []
    results = []
    name = ""
    is_data = False
    result = []
    statistics_vars = []
    statistics_conss = []
    while line:
        print(line, end='')
        if 'data' in line:
            name = line[: -1]
        if line and line[0] == '[':
            is_data = True
        if is_data:
            if line[-2] == ']':
                is_data = False
                line = line[:-1]
            line = line[1:-1]
            line_data = line.split(' ')
            for data in line_data:
                if data != "":
                    result.append(float(data))
            if not is_data:
                if name[name.rfind('/') + 1:] in feature_mapping:
                    ecole_path = feature_mapping[name[name.rfind('/') + 1:]]
                    with gzip.open(ecole_path, 'rb') as fp:
                        ecole_data = pickle.load(fp)
                        assert name[name.rfind('/') + 1:] == ecole_data['instance'].name
                        ecole_feature = ecole_data['data'][0]
                    # name = title + name + '.lp'
                    # env.set_instance(name)
                    # features = env.get_observation()
                    Nconss = len(ecole_feature[0])
                    Nvarss = len(ecole_feature[2])
                    statistics_vars.append(Nvarss)
                    statistics_conss.append(Nconss)
                    constraint_features = ecole_feature[0].tolist()
                    edge_indices = ecole_feature[1][0].tolist()
                    edge_features = np.reshape(ecole_feature[1][1], (len(ecole_feature[1][1]), 1)).tolist()
                    # remove variable feature 13<incumbent_value> and 14<average_incumbent_value> since they are Nan.
                    variable_features_raw = ecole_feature[2]
                    variable_features = np.delete(variable_features_raw, [13, 14], axis=1)
                    variable_features = variable_features.tolist()
                    features_ = [constraint_features, edge_indices, edge_features, variable_features, Nconss, Nvarss]
                    results.append({"name": name, "feature": features_, "label": result})
                    result = []
                else:
                    print("not found: ", name[name.rfind('/') + 1:])
        line = f.readline()
    print(len(results))

    train_size = int(len(results) * 0.9)
    results_train = results[0: train_size]

    with open("generated_data/{}_train.pkl".format(DATASET), "wb") as f:
        pickle.dump(results_train, f)

    results_validation = results[train_size:]
    with open("generated_data/{}_validation.pkl".format(DATASET), "wb") as f:
        pickle.dump(results_validation, f)

    # results_json = json.dumps(results_train)
    # with open("generated_data/annealing_data_{}_train_small.json".format(DATASET), "w") as f:
    #     f.write(results_json)
    #
    # results_validation = results[train_size:]
    # results_json = json.dumps(results_validation)
    # with open("generated_data/annealing_data_{}_validation.json".format(DATASET), "w") as f:
    #     f.write(results_json)
