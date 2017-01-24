from random import choice, randint, random
from math import sqrt, ceil, floor
from time import sleep


CAR_LEN = 5                         # 车辆长度
OUT_BUFFER = 21                     # 出站固定缓冲区长度
HUMAN_WAIT_TIME = 90                # 人工收费等待时间
ELE_WAIT_TIME = 30                  # 电子收费站等待时间

DESIRED_VELOCITY = 60               # 人工驾车理想速度
MAX_ACCELERATION = 2                # 人工驾车最大加速度
COMFORTABLE_DECELERATION = 3        # 人工驾车舒适减速度
ACCELERATION_EXPONENT = 5           # 人工驾车加速度指数delta
MIN_SPACE = 2                       # 人工驾车车间最小距离
DESIRED_HEADWAY = 1.5               # 人工驾车headway时间
POLITENESS = 0.2                    # 人工驾车礼貌程度
DIST_THR = 3                        # 人工驾车换道阈值

AUTO_DESIRED_VELOCITY = 80          # 自动汽车理想速度
AUTO_MAX_ACCELERATION = 5           # 自动汽车最大加速度
AUTO_COMFORTABLE_DECELERATION = 5   # 自动汽车舒适减速度
AUTO_ACCELERATION_EXPONENT = 5      # 自动汽车加速度指数delta
AUTO_MIN_SPACE = 1                  # 自动汽车车间最小距离
AUTO_DESIRED_HEADWAY = 1            # 自动汽车headway时间
AUTO_POLITENESS = 1                 # 自动汽车礼貌程度
AUTO_DIST_THR = 0                   # 自动汽车换道阈值


class Vehicle:
    """
    单纯记录车辆数据，对车辆数据的更新有LaneManager完成
    其中"barrier"类型为速度为零的障碍物
    """
    def __init__(self, type, speed=None, length=None):
        self.type = type
        self.position = 0
        self.timer = 0

        if type == "barrier":
            self.length = length
            self.speed = 0
        elif type == "car":
            self.length = CAR_LEN
            self.speed = DESIRED_VELOCITY
            self.v0 = DESIRED_VELOCITY
            self.a = MAX_ACCELERATION
            self.b = COMFORTABLE_DECELERATION
            self.T = DESIRED_HEADWAY
            self.s0 = MIN_SPACE
            self.politeness = POLITENESS
            self.dist_thr = DIST_THR
        elif type == "auto":
            self.length = CAR_LEN
            self.speed = AUTO_DESIRED_VELOCITY
            self.v0 = AUTO_DESIRED_VELOCITY
            self.a = AUTO_MAX_ACCELERATION
            self.b = AUTO_COMFORTABLE_DECELERATION
            self.T = AUTO_DESIRED_HEADWAY
            self.s0 = AUTO_MIN_SPACE
            self.politeness = AUTO_POLITENESS
            self.dist_thr = AUTO_DIST_THR


class LaneManager:
    """
    管理区域道路，包括道路分流合流、收费站减速停车等情况
    """
    def __init__(self, lane_num, booth_num, upper_lane_num, interval, auto_ratio, ele_num, non_stop_num, time_step):
        # 道路基本信息
        self.lane_num = lane_num                        # 通行车道数
        self.booth_num = booth_num                      # 收费亭个数，默认大于等于车道数
        self.non_stop_num = non_stop_num                # 不停车车道数量
        self.ele_num = ele_num                          # 电子收费站（不找零）车道数量
        self.upper_lane_num = upper_lane_num            # 收费站后上方需变道车道数量
        self.auto_ratio = auto_ratio                    # 自动驾驶汽车比例
        self.time_step = time_step                      # 仿真时间间隔，用于记录车道出车延迟

        # 动态变化信息
        self.timer = 0                                  # 记录仿真时间，用于计算车道出车延迟
        self.lanes = [[] for i in range(booth_num + 2)] # 记录每车道车辆，按车头位置升序排序，多出两车道放置障碍物，便于处理变道
        self.valid_lane_indices = []                    # 记录可进出的车道编号
        self.lane_change_count = 0                      # 记录变道次数
        self.lane_clock = [0] * booth_num               # 记录上一辆车发出的时刻
        self.crash_count = 0                            # 记录撞车次数
        self.out_count = 0                              # 记录通过数
        self.all_time_spent_reci = 0                    # 记录通过车辆总花费时间倒数
        self.all_time_spent = 0                         # 记录通过车辆总花费时间

        # 根据形状建立车道
        # 若车道类型相同，上方车道数只需取[0, (B-L)//2]。当车道类型分布不同时，上方车道数在[0, (B-L)]分别对应不同结果
        upper_extra_lane_num = upper_lane_num
        lower_extra_lane_num = booth_num - lane_num - upper_lane_num
        if upper_extra_lane_num >= lower_extra_lane_num:
            self.lane_length = OUT_BUFFER + (upper_extra_lane_num + 1) * interval * (CAR_LEN + MIN_SPACE)
            upper_seg_length = interval * (CAR_LEN + MIN_SPACE)
            lower_seg_length = (self.lane_length - OUT_BUFFER) / (lower_extra_lane_num + 1)
            for i, j in zip(range(1, upper_extra_lane_num + 1), range(upper_extra_lane_num, 0, -1)):
                self.add_barrier(i, upper_seg_length * j)
            for i in range(1, lower_extra_lane_num + 1):
                self.add_barrier(upper_extra_lane_num + lane_num + i, lower_seg_length * i)
        else:
            self.lane_length = OUT_BUFFER + (lower_extra_lane_num + 1) * interval * (CAR_LEN + MIN_SPACE)
            upper_seg_length = (self.lane_length - OUT_BUFFER) / (upper_extra_lane_num + 1)
            lower_seg_length = interval * (CAR_LEN + MIN_SPACE)
            for i, j in zip(range(1, upper_extra_lane_num + 1), range(upper_extra_lane_num, 0, -1)):
                self.add_barrier(i, upper_seg_length * j)
            for i in range(1, lower_extra_lane_num + 1):
                self.add_barrier(upper_extra_lane_num + lane_num + i, lower_seg_length * i)

        # 第一、最后道放置障碍物
        barrier1 = Vehicle("barrier", 0, self.lane_length)
        barrier2 = Vehicle("barrier", 0, self.lane_length)
        barrier1.position = self.lane_length
        barrier2.position = self.lane_length
        self.lanes[0].append(barrier1)
        self.lanes[-1].append(barrier2)

    def add_barrier(self, lane_index, barrier_length):
        """
        在指定车道道路后方添加障碍物
        :param lane_index: 车道编号，编号0为边界障碍物车道
        :param barrier_length: 障碍物长度
        """

        barrier = Vehicle("barrier", 0, barrier_length)
        barrier.position = self.lane_length
        self.lanes[lane_index].append(barrier)

    def add_vehicle(self, lane_index):
        """
        向指定车道添加车辆，车辆类型由自动驾驶汽车比例决定。
        若车辆不满足入道条件则直接丢弃此次添加，而非放入缓冲区
        :param lane_index: 车道编号，编号0为边界障碍物车道
        """

        # 判断是否能进
        if self.non_stop_num < lane_index <= self.ele_num + self.non_stop_num :
            if self.timer < self.lane_clock[lane_index - 1] + ELE_WAIT_TIME:    # 此处lane_index与数列self.lane_clock中
                return                                                          # 对应序号相差1
        elif self.ele_num + self.non_stop_num  < lane_index:
            if self.timer < self.lane_clock[lane_index - 1] + HUMAN_WAIT_TIME:
                return

        # 根据自动驾驶汽车比例新建一辆车
        if random() < self.auto_ratio:
            vehicle = Vehicle("auto")
        else:
            vehicle = Vehicle("car")

        # 判断是否从零加速，假设不停车车道有最小车道编号
        if lane_index > self.non_stop_num:  # 若在停车通道，则速度为零
            vehicle.speed = 0

        # 检查是否能进入指定车道
        if self.lanes[lane_index]:  # 若车道非空
            last_vehicle = self.lanes[lane_index][0]
            if last_vehicle.position - last_vehicle.length >= vehicle.s0:
                self.lanes[lane_index].insert(0, vehicle)
                self.lane_clock[lane_index - 1] = self.timer
            else:
                del vehicle
        else:                       # 车道为空
            self.lanes[lane_index].append(vehicle)
            self.lane_clock[lane_index - 1] = self.timer

    def add_vehicles(self, num):
        """
        向仿真区域随机添加num量车辆
        :param num: 新增车辆数
        """

        for i in range(num):
            lane_index = randint(1, self.booth_num)
            self.add_vehicle(lane_index)

    def update(self):
        """
        更新动态信息，认为用户每经过一个时间间隔时必调用一次此方法
        """

        # 更新全局计时器
        self.timer += self.time_step

        # 全体计数器计时
        for lane in self.lanes:
            for vehicle in lane:
                vehicle.timer += self.time_step

        # 所有车辆前进
        self.forward_all()

        # 车辆变道
        self.change_lane_dist()

        # 记录统计信息
        for lane_index, lane in enumerate(self.lanes):
            for vehicle_index, vehicle in enumerate(lane):
                if vehicle.position >= self.lane_length and vehicle.type != "barrier":
                    self.out_count += 1
                    self.all_time_spent += vehicle.timer
                    self.all_time_spent_reci += 1 / vehicle.timer
                    lane[vehicle_index] = None
            self.lanes[lane_index] = [x for x in lane if x is not None]

    def forward(self, back, front=None):
        """
        将车道内后车back根据IDM模型前进
        :param back: 后车，即前进车辆
        :param front: 前车，为后车前进计算提供参数
        """

        if front is None:   # 前方无车时，视前车车速正无穷，车距正无穷
            # 计算加速度
            acc = back.a * (1 - (back.speed / back.v0) ** ACCELERATION_EXPONENT)

            # 判断是否停车
            if back.speed + acc * self.time_step > 0:   # 正常情况
                new_position = back.position + back.speed * self.time_step + 0.5 * acc * self.time_step ** 2
                back.position = new_position
                back.speed += acc * self.time_step
            else:                                       # 降速为零
                back.position += 0.5 * back.speed * self.time_step
                back.speed = 0
        else:   # 若前方有车
            # 若两车间无间隔，无需移动后车，直接返回
            if front.position - front.length - back.position == 0:
                back.speed = 0
                return

            # 计算加速度
            s_star = back.s0 + max(0, back.speed * back.T + (back.speed * (back.speed - front.speed)) / 2 / sqrt(
                     back.a * back.b))
            acc = back.a * (1 - (back.speed / back.v0) ** ACCELERATION_EXPONENT - (
                  s_star / (front.position - back.position - front.length)))

            # 判断是否停车
            if back.speed + acc * self.time_step > 0:   # 正常情况
                new_position = back.position + back.speed * self.time_step + 0.5 * acc * self.time_step ** 2
                back.position = new_position
                back.speed += acc * self.time_step
            else:                                       # 降速为零
                back.position += 0.5 * back.speed * self.time_step
                back.speed = 0

            # 判断是否撞车
            if front.position - front.length - back.position < 0:
                # 撞车则将后车急刹至前车后方，速度减为零，增加计数器
                back.position = front.position - front.length
                back.speed = 0
                self.crash_count += 1

    def forward_all(self):
        """
        将行车道内所有车辆根据IDM模型前进
        """

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
        """
        按照距离关系对当前车道内所有车辆进行变道决策进行判断
        """

        for lane_index, lane in enumerate(self.lanes[1:-1]):
            lane_index += 1     # 考虑实际有车车道
            for vehicle_index, vehicle in enumerate(lane):  # 车道每辆车单独处理
                # 忽略障碍物
                if vehicle.type == "barrier":
                    continue

                # 前方无车辆则不变道
                if vehicle_index == len(lane) - 1:
                    continue

                # 计算当前可行驶道路距离
                my_dist_before = lane[vehicle_index + 1].position - lane[vehicle_index + 1].length - vehicle.position

                # 查找向左插队位置最近的前后两车
                nearest_left_behind, nearest_left_front = None, None
                left_overlap_found = False
                for left_index, left in enumerate(self.lanes[lane_index - 1]):
                    if vehicle.position - vehicle.length - left.position >= 0:
                        nearest_left_behind = left
                    elif left.position - left.length - vehicle.position >= 0:
                        nearest_left_front = left
                        break
                    else:
                        left_overlap_found = True
                        break

                # 若左车道满足安全原则，即存在变道空间
                if not left_overlap_found:
                    # 计算变道后本车前方可行驶距离
                    if nearest_left_front is None:  # 前方无车时，可行驶距离即道路剩余距离
                        my_dist_left_after = self.lane_length - vehicle.position
                    else:
                        my_dist_left_after = nearest_left_front.position - nearest_left_front.length - vehicle.position
                    # 变化量
                    my_dist_left_delta = my_dist_left_after - my_dist_before

                    # 计算变道前后，后方车辆可行驶距离
                    if nearest_left_behind is None: # 若后方无车，距离变化量为零
                        left_behind_dist_delta = 0
                    else:   # 若有车，考虑前方是否有车
                        # 变道前
                        if nearest_left_front is None:  # 前方无车时，可行驶距离即道路剩余距离
                            left_behind_dist_before = self.lane_length - nearest_left_behind.position
                        else:
                            left_behind_dist_before = nearest_left_front.position - nearest_left_front.length - \
                                                      nearest_left_behind.position
                        # 变道后
                        left_behind_dist_after = vehicle.position - vehicle.length - nearest_left_behind.position
                        # 变化量
                        left_behind_dist_delta = left_behind_dist_after - left_behind_dist_before

                # 查找向右插队位置最近的前后两车
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

                # 若右车道满足安全原则，即存在变道空间
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
                    else:   # 若有车，考虑前方是否有车
                        # 变道前
                        if nearest_right_front is None:     # 前方无车时，可行驶距离即道路剩余距离
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

                # 进行变道决策
                if go_left and go_right:                                    # 如果同时可以左右换道
                    if left_behind_dist_delta >= right_behind_dist_delta:   # 选择对后方车辆影响更小的车道，此处左转
                        if nearest_left_front is None:
                            self.lanes[lane_index - 1].append(vehicle)
                        else:
                            self.lanes[lane_index - 1].insert(left_index, vehicle)
                    else:                                                   # 否则右转
                        if nearest_right_front is None:
                            self.lanes[lane_index + 1].append(vehicle)
                        else:
                            self.lanes[lane_index + 1].insert(right_index, vehicle)
                    lane[vehicle_index] = None
                elif go_left:                                               # 仅可左转
                    if nearest_left_front is None:
                        self.lanes[lane_index - 1].append(vehicle)
                    else:
                        self.lanes[lane_index - 1].insert(left_index, vehicle)
                    lane[vehicle_index] = None
                elif go_right:                                              # 仅可右转
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
        """
        炫酷的ASCII演示
        """

        print(chr(27) + "[2J")  # 清空命令行
        for lane_index, lane in enumerate(self.lanes):  # 依次打印每个车道
            print("lane %i " % (lane_index), end="")    # 车道信息
            lane_list = list(" " * int(self.lane_length + 1))   # 用一个字符列表记录
            for vehicle in lane:    # 依次修改每辆车对应的字符表示
                if vehicle.position - vehicle.length >= 0:  # 若车尾在显示范围内，打印整辆车（可能仅车头进入仿真区域）
                    lane_list[int(vehicle.position - vehicle.length)] = "["
                    for index in range(int(vehicle.position - vehicle.length) + 1, int(vehicle.position)):
                        lane_list[index] = "-"
                    lane_list[int(vehicle.position)] = "]"
                else:   # 否则只打印车身和车位
                    for index in range(int(vehicle.position)):
                        lane_list[index] = "-"
                    lane_list[int(vehicle.position)] = "]"
            for mark in lane_list:  # 打印这一车道
                print(mark, end="")
            print()
        sleep(0.04)     # 需要设置合适的更新间隙否则看不清楚

    def get_stat(self):
        """
        返回统计信息
        :return: 总通过车辆，平均速度，平均用时，急刹车（撞车）次数，换道次数
        """

        return self.out_count, self.lane_length * self.all_time_spent_reci / self.out_count, \
               self.all_time_spent / self.out_count, self.crash_count, self.lane_change_count