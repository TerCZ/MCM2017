from random import choice
from math import sqrt


CAR_LEN = 5
DESIRED_VELOCITY = 20
MAX_ACCELERATION = 1
COMFORTABLE_DECELERATION = 3
ACCELERATION_EXPONENT = 5
MIN_SPACE = 2
DESIRED_HEADWAY = 1.5


class Vehicle:
    def __init__(self, type, length=None, speed=None):
        self.type = type                        # 车辆类型
        self.position = 0                       # 车头位置
        self.timer = 0                          # 记录存在时间
        if type == "car":
            self.length = CAR_LEN               # 车辆长度
            self.speed = DESIRED_VELOCITY       # 初始速度
            self.v0 = DESIRED_VELOCITY          # 理想速度（限速区域即修改此值）
            self.a = MAX_ACCELERATION           # 最大加速度
            self.b = COMFORTABLE_DECELERATION   # 舒适减速度
            self.T = DESIRED_HEADWAY            # 等待时间
            self.s0 = MIN_SPACE                 # 与前车最小间隔
            self.politeness = 0.5               # 礼貌程度，影响变道决策
            self.dist_thr = 5                   # 变道的带来收益的阈值
        elif type == "auto":
            self.length = CAR_LEN  # 车辆长度
            self.speed = DESIRED_VELOCITY  # 初始速度
            self.v0 = DESIRED_VELOCITY  # 理想速度（限速区域即修改此值）
            self.a = MAX_ACCELERATION  # 最大加速度
            self.b = COMFORTABLE_DECELERATION  # 舒适减速度
            self.T = DESIRED_HEADWAY  # 等待时间
            self.s0 = MIN_SPACE  # 与前车最小间隔
            self.politeness = 0.5  # 礼貌程度，影响变道决策
            self.dist_thr = 5  # 变道的带来收益的阈值
        elif type == "barrier":
            self.length = length  # 车辆长度
            self.speed = speed  # 初始速度


class LaneManager:
    """
    管理区域道路，包括道路分流合流、收费站减速停车等情况
    """
    def __init__(self, lane_num, booth_num, lane_length, shape, pattern, booth_type, time_step):
        # 道路基本信息
        self.lane_num = lane_num                        # 通行车道数
        self.booth_num = booth_num                      # 收费亭个数，默认大于等于车道数
        self.lane_length = lane_length                  # 区域总长度，收费站放置于中央位置
        self.limit_length = lane_length / 2             # 限速区域长度
        self.speed_limit = 10                           # 收费区域限速
        self.booth_type = booth_type                    # 收费亭类型
        self.shape = shape                              # 收费区域形状
        self.pattern = pattern                          # 合流模式
        self.time_step = time_step                      # 仿真时间间隔

        # 动态变化信息
        self.lanes = [[] for i in range(booth_num + 2)] # 记录每车道车辆，按车头位置升序排序，多出两车道放置障碍物，便于处理变道
        self.valid_lane_indices = []                    # 记录可进出的车道编号
        self.lane_change_count = 0                      # 记录变道次数
        self.crash_count = 0                            # 记录撞车次数
        self.out_count = 0                              # 记录通过数
        self.total_time_spent = 0                       # 记录通过车辆总花费时间

        # 根据形状建立车道
        if shape == "isosceles":
            pass
        elif shape == "right":
            seg_num = booth_num - lane_num + 1
            seg_length = lane_length / 2 / seg_num
            for i in range(1, booth_num - lane_num + 1):
                self.add_barrier(lane_num + i, seg_length * i)
            self.valid_lane_indices = range(1, lane_num + 1)

        # 第一、最后道放置障碍物
        barrier1 = Vehicle("barrier", lane_length, 0)
        barrier2 = Vehicle("barrier", lane_length, 0)
        barrier1.position = lane_length
        barrier2.position = lane_length
        self.lanes[0].append(barrier1)
        self.lanes[-1].append(barrier2)

    def add_barrier(self, lane_index, barrier_length):
        barrier1 = Vehicle("barrier", barrier_length, 0)    # 前方障碍
        barrier2 = Vehicle("barrier", barrier_length, 0)    # 后方障碍
        barrier1.position = barrier_length                  # 升序排列
        barrier2.position = self.lane_length
        self.lanes[lane_index].append(barrier1)
        self.lanes[lane_index].append(barrier2)

    def add_vehicle(self, lane_index):
        # 新建一辆车
        vehicle = Vehicle("car")

        # 检查是否能进入指定车道
        if self.lanes[lane_index]:  # 若车道非空
            last_vehicle = self.lanes[lane_index][0]
            if last_vehicle.position - last_vehicle.length >= vehicle.s0:
                self.lanes[lane_index].insert(0, vehicle)
            else:
                del vehicle
        else:                       # 车道为空
            self.lanes[lane_index].append(vehicle)

    def add_vehicles(self, num):
        for i in range(num):
            lane_index = choice(self.valid_lane_indices)
            self.add_vehicle(lane_index)

    def update(self):
        # 所有车辆前进
        self.forward_all()

        # 车辆变道
        self.change_lane_dist()

        # 全体计数器计时
        for lane in self.lanes:
            for vehicle in lane:
                vehicle.timer += self.time_step

        # 记录统计信息
        for lane_index, lane in enumerate(self.lanes):
            for vehicle_index, vehicle in enumerate(lane):
                if vehicle.position >= self.lane_length and vehicle.type != "barrier":
                    self.out_count += 1
                    self.total_time_spent += vehicle.timer
                    lane[vehicle_index] = None
            self.lanes[lane_index] = [x for x in lane if x is not None]


    def forward(self, back, front=None):
        if front is None:   # 前方无车时，视前车车速正无穷，车距正无穷
            # 计算加速度
            if self.lane_length / 2 - self.limit_length <= back.position < self.lane_length / 2:
                acc = back.a * (1 - (back.speed / self.speed_limit) ** ACCELERATION_EXPONENT)
            else:
                acc = back.a * (1 - (back.speed / back.v0) ** ACCELERATION_EXPONENT)

            # 判断是否停车
            if back.speed + acc * self.time_step > 0:  # 正常情况
                new_position = back.position + back.speed * self.time_step + 0.5 * acc * self.time_step ** 2
                # 经过收费站停车
                if back.position < self.lane_length / 2 and new_position >= self.lane_length / 2 and self.booth_type != "nonstop":
                    back.position = self.lane_length / 2
                    back.speed = 0
                else:
                    back.position = new_position
                    back.speed += acc * self.time_step
            else:  # 速度降为零
                back.position += 0.5 * back.speed * self.time_step
                back.speed = 0
        else:
            # 若两车间无间隔，无需移动后车，直接返回
            if front.position - front.length - back.position == 0:
                back.speed = 0
                return

            # 计算加速度，区分是否在减速去内
            s_star = back.s0 + max(0, back.speed * back.T + (back.speed * (back.speed - front.speed)) / 2 / sqrt(
                     back.a * back.b))
            if self.lane_length / 2 - self.limit_length <= back.position < self.lane_length / 2:
                acc = back.a * (1 - (back.speed / self.speed_limit) ** ACCELERATION_EXPONENT - (
                      s_star / (front.position - back.position - front.length)))
            else:
                acc = back.a * (1 - (back.speed / back.v0) ** ACCELERATION_EXPONENT - (
                      s_star / (front.position - back.position - front.length)))

            # 判断是否停车
            if back.speed + acc * self.time_step > 0:   # 正常情况
                new_position = back.position + back.speed * self.time_step + 0.5 * acc * self.time_step ** 2
                # 经过收费站停车
                if back.position < self.lane_length / 2 and new_position >= self.lane_length / 2 and self.booth_type != "nonstop":
                    back.position = self.lane_length / 2
                    back.speed = 0
                else:
                    back.position = new_position
                    back.speed += acc * self.time_step
            else:                                       # 速度降为零
                back.position += 0.5 * back.speed * self.time_step
                back.speed = 0

            # 判断是否撞车
            if front.position - front.length - back.position < 0:
                back.position = front.position - front.length
                back.speed = 0
                self.crash_count += 1

    def forward_all(self):
        for lane in self.lanes[1:-1]:
            if not lane:    # 若车道为空，跳过
                continue

            # 考虑前面n-1量车
            for i in range(len(lane) - 1):
                if lane[i].type == "barrier":   # 跳过障碍物
                    continue
                self.forward(lane[i], lane[i + 1])

            # 最后一辆车
            if lane[-1].type == "barrier":
                continue
            self.forward(lane[-1])

    def change_lane_dist(self):
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
                            left_behind_dist_before = nearest_left_front.position - nearest_left_front.length - \
                                                      nearest_left_behind.position
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

            # 记录换道次数
            self.lane_change_count += len(lane) - len(self.lanes[lane_index])

    def info(self):
        print(chr(27) + "[2J")
        for lane_index, lane in enumerate(self.lanes):
            print("lane %i, cars %.2s " % (lane_index, len(lane)), end="")
            lane_list = list(" " * int(self.lane_length + 1))
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

    def get_stat(self):
        return self.out_count, self.total_time_spent, self.crash_count, self.lane_change_count