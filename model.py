class Vehicle:
    def __init__(self, type, length, lane, speed, front=None):
        self.type = type        # 车辆类型
        self.length = length    # 车辆长度
        self.speed = speed      # 初始速度
        self.lane = lane        # 车道
        self.position = 0       # 车头位置
        self.front = front      # 前一车辆
        self.back = None        # 后一车辆，在后一辆车生成后修改


    def update(self):
        pass


class Lane():
    def __init__(self):
        pass


class LaneManager():
    def __init__(self, lane_num):
        self.lane_num = lane_num


class TollPlaza:
    def __init__(self, booth_num):
        self.booth_num = booth_num

