from random import choice


CAR_LEN = 2.5
CAR_EXPECTED_SPEED = 60


class Vehicle:
    def __init__(self, type, length, lane, speed):
        self.type = type        # 车辆类型
        self.length = length    # 车辆长度
        self.speed = speed      # 初始速度
        self.lane = lane        # 车道
        self.position = 0       # 车头位置

    def update(self):
        pass


class LaneManager:
    """
    管理区域道路，包括道路分流合流、收费站减速停车等情况
    """
    def __init__(self, lane_num, booth_num, lane_length, speed_limit, shape, time_step):
        self.lane_num = lane_num
        self.booth_num = booth_num
        self.lane_length = lane_length
        self.speed_limit = speed_limit
        self.shape = shape
        self.time_step = time_step

        # 记录车辆情况
        self.vehicles = []                          # 按车头位置降序排序
        self.lanes = [[] for i in range(booth_num)] # 记录每车道车辆，按车头位置降序排序
        self.valid_lane_indices = []                # 记录可进出的车道编号

        # 根据形状建立车道
        if shape == "isosceles":
            pass
        elif shape == "right":
            seg_num = booth_num - lane_num + 1
            seg_length = lane_length / 2 / seg_num
            for i, lane_index in enumerate(range(lane_num, booth_num)):
                self.add_barrier(lane_index, seg_length * i)
        self.vehicles.sort(key=lambda x:x.position, reverse=True)   # 确保所有车辆按车头位置降序排列

    def add_barrier(self, lane_index, barrier_length):
        barrier1 = Vehicle("barrier", barrier_length, lane_index, 0)
        barrier2 = Vehicle("barrier", barrier_length, lane_index, 0)
        barrier1.position = barrier_length
        barrier2.position = self.lane_length
        self.lanes[lane_index] += [barrier2, barrier1]  # 降序排列
        self.vehicles += [barrier1, barrier2]           # 此列表排序在添加所有障碍物后进行

    def add_vehicle(self, lane_index):
        # 新建一辆车
        vehicle = Vehicle("car", CAR_LEN, lane_index, CAR_EXPECTED_SPEED)
        # 检查是否能进入指定车道
        last_vehicle = self.lanes[-1]
        if



    def add_vehicles(self, num):
        for i in range(num):
            lane_index = choice(self.valid_lane_indices)
            self.add_vehicle(lane_index)

    def update(self):
        pass

    def get_recent_out(self):
        pass