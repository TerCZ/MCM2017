from model import LaneManager
from math import exp
from random import random
import csv


TIME_STEP = 0.3
LANE_NUM = 4
BOOTH_NUM = 8
LANE_LEN = 100              # 区域总长
VEHICLE_PER_HOUR = 3600
SIM_TIME = 360


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


def main():
    with open("result.csv", "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["L", "B", "形状", "m", "通行量", "平均车速", "换道次数", "急刹车次数", "造价"])
        for L in range(2, 5):
            for B in range(L+1, 17):
                for shape in ["symmetric", "side"]:
                    for m in range(1, 5):
                        result = simulate(lane_num=L, booth_num=B, interval=m, shape=shape)
                        out_count, total_time_spent, crash_count, lane_change_count, vehicle_count = result
                        writer.writerow([L, B, shape, m, out_count / 3600, total_time_spent / vehicle_count, lane_change_count, crash_count])
                        print(L, B, shape, m, "down")



if __name__ == '__main__':
    main()
