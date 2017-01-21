from random import choice
from math import sqrt


CAR_LEN = 2.5
DESIRED_VELOCITY = 80
MAX_ACCELERATION = 5
COMFORTABLE_DECELERATION = 5
ACCELERATION_EXPONENT = 0.5
MIN_SPACE = 2.5
DESIRED_HEADWAY = 5


class Vehicle:
    def __init__(self, type, length, lane, speed):
        self.type = type        # 车辆类型
        self.length = length    # 车辆长度
        self.speed = speed      # 初始速度
        self.lane = lane        # 车道
        self.position = 0       # 车头位置
        self.v0 = DESIRED_VELOCITY
        self.a = MAX_ACCELERATION
        self.b = COMFORTABLE_DECELERATION
        self.T = DESIRED_HEADWAY
        self.politeness = 0.3
        self.dist_thr = 1


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
        self.vehicles = []                              # 按车头位置升序排序
        self.lanes = [[] for i in range(booth_num + 2)] # 记录每车道车辆，按车头位置升序排序，多出两车道放置障碍物，便于处理变道
        self.valid_lane_indices = []                    # 记录可进出的车道编号

        # 根据形状建立车道
        if shape == "isosceles":
            pass
        elif shape == "right":
            seg_num = booth_num - lane_num + 1
            seg_length = lane_length / 2 / seg_num
            for i in range(1, booth_num - lane_num + 1):
                self.add_barrier(lane_num + i, seg_length * i)
            self.valid_lane_indices = range(1, lane_num + 1)
        self.vehicles.sort(key=lambda x: x.position)                # 确保所有车辆按车头位置升序排列

        barrier1 = Vehicle("barrier", lane_length, 0, 0)            # 第一、最后道放置障碍物
        barrier2 = Vehicle("barrier", lane_length, booth_num, 0)
        barrier1.position = lane_length
        barrier2.position = lane_length
        self.lanes[0].append(barrier1)
        self.lanes[-1].append(barrier2)

    def add_barrier(self, lane_index, barrier_length):
        barrier1 = Vehicle("barrier", barrier_length, lane_index, 0)    # 前方障碍
        barrier2 = Vehicle("barrier", barrier_length, lane_index, 0)    # 后方障碍
        barrier1.position = barrier_length                              # 升序排列
        barrier2.position = self.lane_length
        self.lanes[lane_index].append(barrier1)
        self.lanes[lane_index].append(barrier2)
        self.vehicles += [barrier1, barrier2]                           # 此列表排序在添加所有障碍物后进行

    def add_vehicle(self, lane_index):
        # 新建一辆车
        vehicle = Vehicle("car", CAR_LEN, lane_index, DESIRED_VELOCITY)
        # 检查是否能进入指定车道
        if self.lanes[lane_index]:  # 若车道非空
            last_vehicle = self.lanes[lane_index][0]
            if last_vehicle.position - last_vehicle.length >= MIN_SPACE:
                self.lanes[lane_index].insert(0, vehicle)
                self.vehicles.insert(0, vehicle)
            else:
                del vehicle
        else:                       # 车道为空
            self.lanes[lane_index].append(vehicle)
            self.vehicles.insert(0, vehicle)

    def add_vehicles(self, num):
        for i in range(num):
            lane_index = choice(self.valid_lane_indices)
            self.add_vehicle(lane_index)

    def update(self):
        # 所有车辆前进
        self.forward()

        # 车辆变道
        self.info()
        # self.change_lane_dist()
        if self.change_lane_dist():
            print("changed!")
            self.info()

    def forward(self):
        for lane in self.lanes[1:-1]:
            # 若车道为空，跳过
            if not lane:
                continue
            # 考虑前面n-1量车
            for i in range(len(lane) - 1):
                this_one = lane[i]
                if this_one.type == "barrier":  # 跳过障碍物
                    continue
                front_one = lane[i + 1]

                # 计算加速度
                s_star = MIN_SPACE + max(0, this_one.speed * this_one.T +
                                         (this_one.speed * (this_one.speed - front_one.speed)) / 2 /
                                         sqrt(this_one.a * this_one.b))
                if front_one.position - this_one.position - front_one.length == 0:
                    acc = 0
                else:
                    acc = this_one.a * (1 - (this_one.speed / this_one.v0) ** ACCELERATION_EXPONENT -
                                        (s_star / (front_one.position - this_one.position - front_one.length)))

                # 前进并判断是否停车
                if this_one.speed + acc * self.time_step > 0:   # 正常情况
                    this_one.position += this_one.speed * self.time_step + 0.5 * acc * self.time_step ** 2
                    this_one.speed += acc * self.time_step
                else:   # 速度降为零
                    this_one.position += 0.5 * this_one.speed * self.time_step
                    this_one.speed = 0

            # 考虑最后一辆车，将最后一辆车的gap视为很长
            this_one = lane[-1]
            if this_one.type == "barrier":
                continue
            s_star = MIN_SPACE + max(0, this_one.speed * this_one.T + this_one.speed ** 2 / 2 / sqrt(
                     this_one.a * this_one.b))
            acc = this_one.a * (1 - (this_one.speed / this_one.v0) ** ACCELERATION_EXPONENT - (s_star / 1000))
            if this_one.speed + acc * self.time_step > 0:   # 正常情况
                this_one.position += this_one.speed * self.time_step + 0.5 * acc * self.time_step ** 2
                this_one.speed += acc * self.time_step
            else:   # 速度降为零
                this_one.position += 0.5 * this_one.speed * self.time_step
                this_one.speed = 0

    def change_lane_dist(self):
        any_change = False
        for lane_index, lane in enumerate(self.lanes[1:-1]):    # 考虑实际有车车道
            lane_index += 1
            for vehicle_index, vehicle in enumerate(lane):      # 车道每辆车单独处理
                # 忽略障碍物
                if vehicle.type == "barrier":
                    continue

                # 前方无车辆则不变道
                if vehicle_index == len(lane) - 1:
                    continue

                # 计算当前道路距离
                my_dist_before = lane[vehicle_index + 1].position - lane[vehicle_index + 1].length - vehicle.position

                # 查找左边插队的车（后方）
                nearest_left_behind, nearest_left_front = None, None
                left_overlap_found = False
                for left_index, left in enumerate(self.lanes[lane_index - 1]):
                    if vehicle.position - vehicle.length - left.position > 0:
                        nearest_left_behind = left
                    elif left.position - left.length - vehicle.position > 0:
                        nearest_left_front = left
                        break
                    else:
                        left_overlap_found = True
                        break

                # 若左车道满足安全原则
                if not left_overlap_found:
                    # 计算变道后本车前方可行驶距离
                    if nearest_left_front is None:       # 前方无车时，可行驶距离即道路剩余距离
                        my_dist_left_after = self.lane_length - vehicle.position
                    else:
                        my_dist_left_after = nearest_left_front.position - nearest_left_front.length - vehicle.position
                    # 变化量
                    my_dist_left_delta = my_dist_left_after - my_dist_before

                    # 计算变道前后，后方车辆可行驶距离
                    if nearest_left_behind is None:     # 若后方无车，距离变化量为零
                        left_behind_dist_delta = 0
                    else:                               # 若有车，考虑前方是否有车
                        # 变道前
                        if nearest_left_front is None:   # 前方无车时，可行驶距离即道路剩余距离
                            left_behind_dist_before = self.lane_length - nearest_left_behind.position
                        else:
                            left_behind_dist_before = nearest_left_front.position - nearest_left_front.length - nearest_left_behind.position
                        # 变道后
                        left_behind_dist_after = vehicle.position - vehicle.length - nearest_left_behind.position
                        # 变化量
                        left_behind_dist_delta = left_behind_dist_after - left_behind_dist_before

                # 查找右边插队的车（后方）
                nearest_right_behind, nearest_right_front = None, None
                right_overlap_found = False
                for right_index, right in enumerate(self.lanes[lane_index + 1]):
                    if vehicle.position - vehicle.length - right.position > 0:
                        nearest_right_behind = right
                    elif right.position - right.length - vehicle.position > 0:
                        nearest_right_front = right
                        break
                    else:
                        right_overlap_found = True
                        break

                # 若右车道满足安全原则
                if not right_overlap_found:
                    # 计算变道后本车前方可行驶距离
                    if nearest_right_front is None:  # 前方无车时，可行驶距离即道路剩余距离
                        my_dist_right_after = self.lane_length - vehicle.position
                    else:
                        my_dist_right_after = nearest_right_front.position - \
                                              nearest_right_front.length - vehicle.position
                    # 变化量
                    my_dist_right_delta = my_dist_right_after - my_dist_before

                    # 计算变道前后，后方车辆可行驶距离
                    if nearest_right_behind is None:    # 若后方无车，距离变化量为零
                        right_behind_dist_delta = 0
                    else:                               # 若有车，考虑前方是否有车
                        # 变道前
                        if nearest_right_front is None: # 前方无车时，可行驶距离即道路剩余距离
                            right_behind_dist_before = self.lane_length - nearest_right_behind.position
                        else:
                            right_behind_dist_before = nearest_right_front.position - nearest_right_front.length - \
                                                       nearest_right_behind.position
                        # 变道后
                        right_behind_dist_after = vehicle.position - vehicle.length - nearest_right_behind.position
                        # 变化量
                        right_behind_dist_delta = right_behind_dist_after - right_behind_dist_before

                # 跳过左右均不可换道情况
                if left_overlap_found and right_overlap_found:
                    continue

                # 看是否满足变道激励原则
                go_left, go_right = False, False
                if not left_overlap_found:  # 左侧可安全变道
                    if my_dist_left_delta + vehicle.politeness * left_behind_dist_delta >= vehicle.dist_thr:
                        go_left = True
                if not right_overlap_found: # 右侧可安全变道
                    if my_dist_right_delta + vehicle.politeness * right_behind_dist_delta >= vehicle.dist_thr:
                        go_right = True
                if go_left and go_right:                                    # 如果同时可以左右换道
                    if left_behind_dist_delta >= right_behind_dist_delta:   # 选择对后方车辆影响更小的车道，此处左转
                        if nearest_left_front is None:
                            self.lanes[lane_index - 1].append(vehicle)
                        else:
                            self.lanes[lane_index - 1].insert(left_index, vehicle)
                    else:                                                   # 右转
                        if nearest_right_front is None:
                            self.lanes[lane_index + 1].append(vehicle)
                        else:
                            self.lanes[lane_index + 1].insert(right_index, vehicle)
                    lane[vehicle_index] = None
                elif go_left:
                    if nearest_left_front is None:
                        self.lanes[lane_index - 1].append(vehicle)
                    else:
                        self.lanes[lane_index - 1].insert(left_index, vehicle)
                    lane[vehicle_index] = None
                elif go_right:
                    if nearest_right_front is None:
                        self.lanes[lane_index + 1].append(vehicle)
                    else:
                        self.lanes[lane_index + 1].insert(right_index, vehicle)
                    lane[vehicle_index] = None

            # 更新当前车道，去掉换道的车辆
            self.lanes[lane_index] = [x for x in lane if x is not None]
            if len(self.lanes[lane_index]) != len(lane):
                any_change = True

        return any_change

    def get_recent_out(self):
        counter = 0
        for lane_index, lane in enumerate(self.lanes):
            for vehicle_index, vehicle in enumerate(lane):
                if vehicle.position >= self.lane_length and vehicle.type != "barrier":
                    lane[vehicle_index] = None
                    counter += 1
            self.lanes[lane_index] = [x for x in lane if x is not None]

        return counter

    def info(self):
        for lane_index, lane in enumerate(self.lanes):
            print("lane %i, cars %.2s " % (lane_index, len(lane)), end="")
            lane_list = list(" " * (400))
            for vehicle in lane:
                if vehicle.position - vehicle.length >= 0:
                    lane_list[int(vehicle.position - vehicle.length)] = "["
                    for index in range(int(vehicle.position - vehicle.length) + 1, int(vehicle.position)):
                        lane_list[index] = "-"
                    lane_list[int(vehicle.position)] = "]"
                else:
                    for index in range(int(vehicle.position)):
                        lane_list[index] = "-"
                    lane_list[int(vehicle.position)] = "]"
            for mark in lane_list:
                print(mark, end="")
            print()
        print()