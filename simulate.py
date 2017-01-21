from model import LaneManager
from time import sleep


TIME_STEP = 0.3
LANE_NUM = 4
BOOTH_NUM = 10
LANE_LEN = 100              # 区域总长
VEHICLE_PER_HOUR = 6000


def main():
    manager = LaneManager(lane_num=LANE_NUM, booth_num=BOOTH_NUM, lane_length=LANE_LEN, shape="side",
                          pattern="regular", interval=4, booth_type="human", time_step=TIME_STEP)

    vehicle_per_sec = 3600 / VEHICLE_PER_HOUR
    timer, remainder, vehicle_count = 0, 0, 0
    while timer <= 3600:
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
            vehicle_count += new_vehicle_num

        # 全局更新
        manager.info()
        sleep(0.04)
        manager.update()
    out_count, total_time_spent, crash_count, lane_change_count = manager.get_stat()
    print("throughput\t", out_count / timer)
    print("latency\t", total_time_spent / vehicle_count)
    print("lane change\t", lane_change_count)
    print("crash\t", crash_count)

if __name__ == '__main__':
    main()
