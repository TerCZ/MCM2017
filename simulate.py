from model import LaneManager
from math import exp
from random import random
import csv
import numpy as np
import pickle


TIME_STEP = 0.3
LANE_NUM = 4
BOOTH_NUM = 8
LANE_LEN = 100              # 区域总长
VEHICLE_PER_HOUR = 3600
SIM_TIME = 360
alphas = [1, 1, -1, -1, -1]


def simulate(lane_num=LANE_NUM, booth_num=BOOTH_NUM, shape="symmetric", interval=3, auto_ratio=0, non_stop_num=0,
             total_time=SIM_TIME):
    manager = LaneManager(lane_num=lane_num, booth_num=booth_num, shape=shape, interval=interval, auto_ratio=auto_ratio,
                          non_stop_num=non_stop_num, time_step=TIME_STEP)

    vehicle_per_time_step = VEHICLE_PER_HOUR * TIME_STEP / 3600
    timer, remainder, vehicle_count = 0, 0, 0
    while timer <= total_time:
        # 更新时间
        timer += TIME_STEP

        # 计算此循环内新加入车辆数，泊松分布
        l = exp(-vehicle_per_time_step)
        new_vehicle_num = 0
        p = random()
        while p > l:
            p *= random()
            new_vehicle_num += 1

        # 添加车辆
        if new_vehicle_num:
            manager.add_vehicles(int(new_vehicle_num))
            vehicle_count += new_vehicle_num

        # 全局更新
        # manager.info()
        manager.update()

    return manager.get_stat() + (vehicle_count,)


def loop():
    with open("result.csv", "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["L", "B", "形状", "m", "通行量", "平均车速", "换道次数", "急刹车次数", "造价"])
        data = {}
        for L in range(2, 5):
            data[L] = {}
            for B in range(L+1, 17):
                all_design = []
                data[L][B] = all_design
                for shape in ["symmetric", "side"]:
                    for m in range(1, 5):
                        result = simulate(lane_num=L, booth_num=B, interval=m, shape=shape)
                        out_count, total_time_spent, crash_count, lane_change_count, vehicle_count = result
                        if shape == "symmetric":
                            cost = 804.21*1.75*m*(5*B-1.3*L+14)+105*(B-L)+294
                        elif shape == "side":
                            cost = 804.21*3.5*m*(5*B-1.3*L+14)+105*(B-L)+294
                        writer.writerow([L, B, shape, m, out_count / 3600, total_time_spent / vehicle_count, lane_change_count, crash_count, cost])
                        all_design.append([out_count, total_time_spent / vehicle_count, lane_change_count, crash_count, cost])
                        print(L, B, shape, m, "down")
        pickle.dump(data, open("data.pkl", "wb"))


def evaluate():
    # 计算目标层到准则层的判断矩阵
    matrix = np.array([1, 1 / 2, 5, 4, 1 / 3,
                       2, 1, 6, 5, 1 / 2,
                       1 / 5, 1 / 6, 1, 1 / 2, 1 / 7,
                       1 / 4, 1 / 5, 2, 1, 1 / 5,
                       3, 2, 7, 5, 1])
    matrix.shape = (5, 5)
    e_val, e_vec = np.linalg.eig(matrix)
    e_list = list(zip(e_val[e_val == e_val.real], e_vec[e_val == e_val.real]))
    e_list.sort(key=lambda tup: tup[0], reverse=True)
    index_ratings = e_list[0][1]

    data = pickle.load(open("data.pkl", "rb"))
    choices = {}
    for L in range(2, 5):
        choices[L] = {}
        for B in range(L + 1, 17):
            #  每个LB分别评估方案
            sim_result = data[L][B]
            plan_ratings = np.zeros((1, 8))
            for i in range(5):
                matrix = np.ones((8, 8))
                for j in range(8):  # 遍历八种方案结果，相除得到相对权重
                    for k in range(8):
                        if alphas[i] > 0:
                            if sim_result[k][i] == 0 and sim_result[j][i] == 0:
                                ajk = 1
                            elif sim_result[k][i] == 0:
                                ajk = 9
                            else:
                                ajk = (sim_result[j][i] / sim_result[k][i]) ** alphas[i]
                        else:
                            if sim_result[k][i] == 0 and sim_result[j][i] == 0:
                                ajk = 1
                            elif sim_result[j][i] == 0:
                                ajk = 9
                            else:
                                ajk = (sim_result[k][i] / sim_result[j][i]) ** -alphas[i]

                        matrix[j][k] = ajk
                e_val, e_vec = np.linalg.eig(matrix)
                e_list = list(zip(e_val[e_val == e_val.real], e_vec[e_val == e_val.real]))
                e_list.sort(key=lambda tup: tup[0], reverse=True)
                plan_ratings += e_list[0][1].real * index_ratings[i].real

            temp = list(zip(range(8), plan_ratings))
            temp.sort(key=lambda tup: tup[1], reverse=True)
            if temp[0][0] < 4:
                shape = "symmetric"
            else:
                shape = "side"
            choices[L][B] = shape + " {}".format(temp[0][0] % 4)
            print(L, B, shape + " {}".format(temp[0][0] % 4))


def main():
    loop()
    # evaluate()


if __name__ == '__main__':
    main()
