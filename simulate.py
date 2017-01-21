from model import LaneManager
from time import sleep


TIME_STEP = 0.5
LANE_NUM = 2
BOOTH_NUM = 2
LANE_LEN = 100           # 每段路长度，由收费站分成两段
VEHICLE_PER_HOUR = 7000
SPEED_LIMIT = 80        # km/h
SHAPE = "right"


def main():
    manager = LaneManager(LANE_NUM, BOOTH_NUM, LANE_LEN, SPEED_LIMIT, SHAPE, TIME_STEP)

    vehicle_per_sec = 3600 / VEHICLE_PER_HOUR
    timer, remainder = 0, 0
    out_num = 0
    while timer <= 1000:
        # 更新时间
        timer += TIME_STEP
        # 计算此循环内新加入车辆数
        new_vehicle_num = 0
        if (remainder + TIME_STEP) // vehicle_per_sec > 0:
            new_vehicle_num = (remainder + TIME_STEP) // vehicle_per_sec
            remainder = (remainder + TIME_STEP) % vehicle_per_sec
        else:
            remainder += TIME_STEP

        # 添加车辆
        if new_vehicle_num:
            manager.add_vehicles(int(new_vehicle_num))

        # 全局更新
        manager.info()
        manager.update()
        out_num += manager.get_recent_out()

    print("throughput:", out_num / timer)

if __name__ == '__main__':
    main()
