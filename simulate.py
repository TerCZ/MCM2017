from model import Vehicle, LaneManager, TollPlaza


TIME_STEP = 0.5
LANE_NUM = 2
BOOTH_NUM = 4
LANE_LEN = 50           # 每段路长度，由收费站分成两段
VEHICLE_PER_HOUR = 3600


def main():
    lanes_in = LaneManager(LANE_NUM)
    lanes_out = LaneManager(LANE_NUM)
    plaza = TollPlaza(BOOTH_NUM)

    vehicle_per_sec = 3600 / VEHICLE_PER_HOUR
    timer, remainder = 0, 0
    while True:
        # 更新时间
        timer += TIME_STEP
        # 计算此循环内新加入车辆数
        if (remainder + TIME_STEP) // vehicle_per_sec > 0:
            new_vehicle_num = (remainder + TIME_STEP) // vehicle_per_sec
            remainder = (remainder + TIME_STEP) % vehicle_per_sec
        else:
            remainder += TIME_STEP

        lanes_in.add(new_vehicle_num)
