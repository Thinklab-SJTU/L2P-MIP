import numpy as np
import csv

if __name__ == '__main__':
    DATASET = "hard_nips_anonymous"  # name for the output file
    i = 3  # denotes the type of reward (default 3)
    name = 'results/reward{}_annealing_results_{}.txt'.format(i, DATASET)
    f = open(name)
    line = f.readline()
    reward = []
    results = []
    name = ""
    while line:
        print(line, end='')
        if 'data' in line:
            name = line[0:-1]
        if line and line[0] == '(':
            line = line[1:-2]
            data = line.split(',')
            y = float(data[0])
            y1 = float(data[1])
            y2 = float(data[2])
            t1 = float(data[3])
            t2 = float(data[4])
            nv1 = int(data[5])
            nc1 = int(data[6])
            nv2 = int(data[7])
            nc2 = int(data[8])
            solved_int1 = int(data[9])
            total_int1 = int(data[10])
            solved_int2 = int(data[11])
            total_int2 = int(data[12])
            obj1 = float(data[13])
            obj2 = float(data[14])
            real_obj1 = float(data[15])
            real_obj2 = float(data[16])
            nv1_origin = int(data[17])
            nc1_origin = int(data[18])
            nv2_origin = int(data[19])
            nc2_origin = int(data[20])
            bound_gap1 = float(data[21])
            bound_gap2 = float(data[22])
            result = [name, y, y1, y2, nv1, nc1, solved_int1, total_int1, obj1, t1, bound_gap1, nv2, nc2, solved_int2,
                      total_int2, obj2, t2, bound_gap2]
            results.append(result)
            print(result)
        line = f.readline()
    print(len(results))
    with open("results/reward{}_annealing_results_{}.csv".format(i, DATASET), "w") as f:
        writer = csv.writer(f)
        writer.writerows(results)
