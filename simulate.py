from model import LaneManager
from math import exp, floor, ceil
from random import random
from scipy import interpolate

import matplotlib.pyplot as plt
import numpy as np
import csv
import pickle


TIME_STEP = 0.3                             # 仿真时间间隔
VEHICLE_PER_HOUR = 7000                     # 涌入收费站默认流量
SIM_TIME = 3600                             # 仿真总时间

alphas = [1, -1, -1, -1, -1]                # 计算层次分析法方案层到标准层的评价矩阵
LANE_LEAST = 2                              # 常规车道L的最小值
LANE_MOST = 8                               # 常规车道L的最大值
BOOTH_THR = 8                               # 收费站个数B的阈值
M_RANGE = np.arange(0.2, 4.1, 0.2)          # m的模拟范围

color_sequence = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#c5b0d5']     # 画图线条颜色


def simulate(lane_num=4, booth_num=8, upper_extra_lane_num=0, interval=3.0, auto_ratio=0, ele_num=0,
             non_stop_num=0, total_time=SIM_TIME, traffic_flow=VEHICLE_PER_HOUR):
    """
    以指定参数进行一次仿真，并返回统计结果
    :param lane_num: 常规车道数
    :param booth_num: 收费站数
    :param upper_extra_lane_num: 上方需变道车道数
    :param interval: m值，m * (车长 + 车辆安全距离)为变道车道间长度之差
    :param auto_ratio: 自动驾驶汽车比例
    :param ele_num: 电子收费站（不找零）个数
    :param non_stop_num: 不停车车道个数
    :param total_time: 仿真总时间
    :param traffic_flow: 驶入收费站车流量
    :return: 总通过车辆，平均速度，平均用时，急刹车（撞车）次数，换道次数，驶入收费站车辆数
    """

    # 新建LaneManager进行仿真操作
    manager = LaneManager(lane_num=lane_num, booth_num=booth_num, upper_lane_num=upper_extra_lane_num, interval=interval, auto_ratio=auto_ratio,
                          ele_num=ele_num, non_stop_num=non_stop_num, time_step=TIME_STEP, UI=False)

    # 计算单位时间间隔平均涌入车辆数量，用于柏松分布计算
    vehicle_per_time_step = traffic_flow * TIME_STEP / 3600

    # 初始化计数器
    timer, vehicle_count = 0, 0
    while timer <= total_time:
        # 更新时间
        timer += TIME_STEP

        # 根据泊松分布计算此时间间隔内新驶入收费站的车辆数
        # 公式来自维基百科
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
        if UI:  # 显示ASCII动画
            manager.info()
        manager.update()

    return manager.get_stat() + (vehicle_count,)


def loop(write=False, file_num=None):
    """
    以L, B, m, x参数为变量进行多次仿真
    :param write: 是否将仿真结果写入文件
    :param file_num: 输出文件编号
    """

    # 存为CSV表格
    if write:
        outfile = open("result_{}.csv".format(file_num), "w", newline="")
        writer = csv.writer(outfile)
        writer.writerow(["L", "B", "x", "m", "throughput", "average time spent per unit length",
                         "lane change per unit area", "abrupt brake per unit area", "cost",
                         "average time spent", "average speed"])    # 共7项记录下来的指标

    # 记录仿真结果
    data = {}

    for L in range(LANE_LEAST, LANE_MOST + 1):              # L的范围为[LANE_LEAST, LANE_MOST]
        data[L] = {}
        for B in range(L + 1, max(BOOTH_THR, 2 * L) + 1):   # B的范围为[L+1, max(BOOTH_THR, 2L)]
            data[L][B] = []
            for x in range(floor((B - L) / 2) + 1):         # x的范围为[0, (B-L)//2]
                for m in M_RANGE:                           # m的范围由上方全局变量决定
                    # 以上述L, B, x, m参数进行仿真
                    result = simulate(lane_num=L, booth_num=B, interval=m, upper_extra_lane_num=x)
                    # 记录结果
                    out_count, average_speed, average_time_spent, crash_count, lane_change_count, vehicle_try_in = result
                    # 计算仿真区域长度、面积
                    length = 21 + (B - L - x + 1) * m * 7
                    area = (294+105*(B-L)+3.5*m*(5*B-1.3*L+14)*(B-L-x))
                    # 根据面积计算造价
                    cost = 804.21*area
                    # 记录仿真结果
                    data[L][B].append([out_count, average_time_spent / length, lane_change_count / area,
                                       crash_count / area, cost, average_time_spent, average_speed, {"x":x, "m":m}])
                    # 打印提示
                    print(L, B, x, m, "down")

                    # 写入CSV表格
                    if write:
                        writer.writerow([out_count, average_time_spent / length, lane_change_count / area,
                                       crash_count / area, cost, average_time_spent, average_speed])

    # 将数据对象存为二进制序列
    if write:
        pickle.dump(data, open("data_{}.pkl".format(file_num), "wb"))


def evaluate(file_num=None):
    """
    以层次分析法进行分析
    :param file_num: 分析的数据来源
    """

    # 计算目标层到准则层的几个判断矩阵
    # 五个准则分别是：  通行量 用时/长度 换道/面积 刹车数/面积 造价
    # 安全 > 性能 > 造价
    matrix = np.array([[1,      2,      0.2,    0.25,   3],
                       [0,      1,      0.167,  0.2,    2],
                       [0,      0,      1,      2,      7],
                       [0,      0,      0,      1,      5],
                       [0,      0,      0,      0,      1]])

    # # 性能 = 安全 > 造价
    # matrix = np.array([[1,      1,      1,      1,      3],
    #                    [0,      1,      1,      1,      3],
    #                    [0,      0,      1,      1,      3],
    #                    [0,      0,      0,      1,      3],
    #                    [0,      0,      0,      0,      1]])

    # # 性能 > 安全 > 造价
    # matrix = np.array([[1,      1,      3,      3,      5],
    #                    [0,      1,      3,      3,      5],
    #                    [0,      0,      1,      1,      3],
    #                    [0,      0,      0,      1,      3],
    #                    [0,      0,      0,      0,      1]])

    # # 性能 < 安全 < 造价
    # matrix = np.array([[1,      1,      0.33,   0.33,   0.2],
    #                    [0,      1,      0.33,   0.33,   0.2],
    #                    [0,      0,      1,      1,      0.33],
    #                    [0,      0,      0,      1,      0.33],
    #                    [0,      0,      0,      0,      1]])

    # 计算完整矩阵
    for i in range(1, 5):
        for j in range(i):
            matrix[i][j] = 1 / matrix[j][i]
    # 计算特征值、特征向量
    e_val, e_vec = np.linalg.eig(matrix)
    e_vec = e_vec.real.transpose()
    # 找到最大特征值对应特征向量
    e_list = list(zip(e_val.real[e_val == e_val.real], e_vec[e_val == e_val.real]))
    e_list.sort(key=lambda tup: tup[0], reverse=True)
    # 归一化处理
    index_ratings = e_list[0][1] / e_list[0][1].sum()
    index_ratings /= index_ratings.sum()

    # 加载本地数据
    data = pickle.load(open("data_{}.pkl".format(file_num), "rb"))

    # 记录每个L, B下的方案选择
    choices = {}

    # 每个L,B分别评估方案
    for L in range(LANE_LEAST, LANE_MOST + 1):
        choices[L] = {}
        for B in range(L + 1, max(BOOTH_THR, 2 * L) + 1):
            sim_result = data[L][B]
            plan_ratings = np.array([0] * len(sim_result), dtype=np.float64)

            # 分别计算方案层到准则层对应的五个判断矩阵
            for i in range(5):
                # 矩阵值由每个方案实际仿真结果之间相互的比值决定
                matrix = np.ones((len(sim_result), len(sim_result)))
                max_ajk = 0
                for j in range(len(sim_result)):  # 遍历方案结果，相除得到相对权重
                    for k in range(len(sim_result)):
                        if sim_result[k][i] == 0:
                            continue
                        else:
                            # 将比值从[0, 1, +oo]映射到[1/9, 1, 9]
                            temp = 9 - 80 / (sim_result[j][i] / sim_result[k][i] + 9)
                            ajk = temp ** alphas[i]
                            if ajk > max_ajk:
                                max_ajk = ajk
                            matrix[j][k] = ajk

                # 处理分母为零情况
                for j in range(len(sim_result)):
                    for k in range(len(sim_result)):
                        if sim_result[k][i] == 0 and sim_result[j][i] == 0:
                            matrix[j][k] = 1
                        elif sim_result[k][i] == 0:
                            matrix[j][k] = 1.5 * max_ajk

                # 计算特征值、特征向量
                e_val, e_vec = np.linalg.eig(matrix)
                e_vec = e_vec.real.transpose()

                # 找出最大特征值对应特征向量
                e_list = list(zip(e_val.real[e_val == e_val.real], e_vec[e_val == e_val.real]))
                e_list.sort(key=lambda tup: tup[0], reverse=True)

                # 更新各个方案的评分
                plan_ratings += e_list[0][1] / e_list[0][1].sum() * index_ratings[i]

            # 选出评分最高的方案，并记录结果
            temp = list(zip(range(len(sim_result)), plan_ratings))
            temp.sort(key=lambda tup: tup[1], reverse=True)
            choices[L][B] = sim_result[temp[0][0]][-1]

    # 打印结果
    max_b = 0
    print("--", end="\t")
    for L in range(LANE_LEAST, LANE_MOST + 1):
        if max(BOOTH_THR, 2 * L) > max_b:
            max_b = max(BOOTH_THR, 2 * L)
        print(L, end="\t")
    print()

    for B in range(LANE_LEAST + 1, max_b + 1):
        print("B =", B, end=" &")
        for L in range(LANE_LEAST, LANE_MOST + 1):
            if B in choices[L].keys():
                print("m =", choices[L][B]["m"], "x =", choices[L][B]["x"], end=" & ")
            else:
                print("----", end=" & ")
        print()


def draw1():
    """
    画论文第三部分的图
    """

    # 变道、平均用时/长度、平均用时、造价在特定LB组合下随m的变化
    titles = ["Average Time Spent per Unit Length at L = {}, B = {}", "Average Time Spent at L = {}, B = {}",
              "Average Speed at L = {}, B = {}", "Lane Switches per Unit Area at L = {}, B = {}",
              "Lane Switches at L = {}, B = {}", "Cost at L = {}, B = {}"]
    y_labels = ["Average time spent per unit length ($s/m$)", "Average time spent ($s$)", "Average speed ($m/s^2$)",
                "Lane switches per unit area ($1/m^2$)", "Lane switches", "Cost ($US dollar$)"]

    data_draw1 = pickle.load(open("data_draw1.pkl", "rb"))
    for L, B in [(2, 6), (4, 12), (6, 20)]:
        # data_draw1[L] = {}
        if floor((B - L) / 2) <= 2:
            choices_of_x = [0, floor((B - L) / 4), floor((B - L) / 2)]
        else:
            choices_of_x = [0, floor((B - L) / 6), floor((B - L) / 3), floor((B - L) / 2)]

        x_val = np.linspace(0.5, 10, 20)
        x_fit = np.linspace(0.5, 10, 200)
        # y_vals = []
        # for index in range(len(titles)):
        #     y_vals.append([])
        # for line_num, x in enumerate(choices_of_x):  # line_num指不同x对应的线的编号
        #     for index in range(len(titles)):
        #         y_vals[index].append([])
        #
        #     for m in x_val:
        #         sim_result = simulate(lane_num=L, booth_num=B, interval=m, upper_extra_lane_num=x)
        #         out_count, average_speed, average_time_spent, crash_count, lane_change_count, vehicle_try_in = sim_result
        #         length = 21 + (B - L - x + 1) * m * 7
        #         area = (294 + 105 * (B - L) + 3.5 * m * (5 * B - 1.3 * L + 14) * (B - L - x))
        #         cost = 804.21 * area
        #         for index, target in enumerate([average_time_spent / length, average_time_spent, average_speed,
        #                                         lane_change_count / area, lane_change_count, cost]):
        #             y_vals[index][line_num].append(target)
        #         print(L, B, x, m, "down")

        y_vals = data_draw1[L][B]

        for index in range(len(titles)):
            fig, ax = plt.subplots()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)

            for line_num, x in enumerate(choices_of_x):  # line_num指不同x对应的线的编号
                y_val = y_vals[index][line_num]
                y_bspline = interpolate.splev(x_fit, interpolate.splrep(x_val, y_val))  # B-spline
                y_poly_fit = np.polyval(np.polyfit(x_val, y_val, 2), x_fit)             # 多项式拟合
                plt.plot(x_val, y_val, "o", color=color_sequence[line_num])
                plt.plot(x_fit, y_bspline, color=color_sequence[line_num], label="x = {}".format(x))
            ax.legend().get_frame().set_alpha(0.3)
            title = titles[index].format(L, B)
            plt.xlabel("Difference in length between consecutive lane")
            plt.ylabel(y_labels[index])
            plt.savefig(filename="./plot/{}".format(title))

    # 通行量、急刹车在特定m下（x取对称的情况）随L的变化
    titles = ["Throughput at m = {}", "Abrupt Brakes per Unit Area at m = {}", "Abrupt Brakes at m = {}"]
    for m in [1, 3, 5]:
        choices_of_B_over_L = [1, 1.3, 1.7, 2, 3]
        x_val = np.arange(2, 15)
        x_fit = np.linspace(2, 14, 200)
        # y_vals = []
        # for index in range(len(titles)):
        #     y_vals.append([])
        # for line_num, B_over_L in enumerate(choices_of_B_over_L):  # line_num指不同B对应的线的编号
        #     for index in range(len(titles)):
        #         y_vals[index].append([])
        #
        #     for L in x_val:
        #         B = ceil(L * B_over_L)
        #         x = floor((B - L) / 2)
        #
        #         sim_result = simulate(lane_num=L, booth_num=B, interval=m, upper_extra_lane_num=x)
        #         out_count, average_speed, average_time_spent, crash_count, lane_change_count, vehicle_try_in = sim_result
        #         area = (294 + 105 * (B - L) + 3.5 * m * (5 * B - 1.3 * L + 14) * (B - L - x))
        #         cost = 804.21 * area
        #         for index, target in enumerate([out_count, crash_count / area, crash_count]):
        #             y_vals[index][line_num].append(target)
        #         print(L, B, x, m, "down")

        y_vals = data_draw1[m]

        for index in range(len(titles)):
            fig, ax = plt.subplots()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)

            for line_num, B_over_L in enumerate(choices_of_B_over_L):  # line_num指不同x对应的线的编号
                B = ceil(L * B_over_L)

                y_val = y_vals[index][line_num]
                y_bspline = interpolate.splev(x_fit, interpolate.splrep(x_val, y_val))  # B-spline
                y_poly_fit = np.polyval(np.polyfit(x_val, y_val, 2), x_fit)  # 多项式拟合
                ax.plot(x_val, y_val, "o", color=color_sequence[line_num])
                ax.plot(x_fit, y_bspline, color=color_sequence[line_num], label="B = {}".format(B))
            ax.legend().get_frame().set_alpha(0.3)
            title = titles[index].format(L, B)
            plt.xlabel("The number of lanes")
            plt.ylabel(y_labels[index])
            plt.savefig(filename="./plot/{}".format(title))


def draw2_traffic():
    # 交通流量的影响
    titles = ["Throughput Under Different Traffic Condition", "Average Time Spent Under Different Traffic Condition",
              "Average Time Spent Per Unit Length Under Different Traffic Condition",
              "Average Speed Under Different Traffic Condition",
              "Lane Switches Per Unit Area Under Different Traffic Condition",
              "Abrupt Brakes Per Unit Area Under Different Traffic Condition"]
    y_labels = ["Throughput ($vehicles/h$)", "Average time spent ($s$)", "Average time spent per unit length ($s/m$)",
                "Average speed ($m/s^2$)", "Lane switches per unit area ($1/m^2$)",
                "Abrupt brakes per unit area ($1/m^2$)", ]

    y_vals = pickle.load(open("data_draw2_traffic.pkl", "rb"))

    x_val = np.linspace(100, 10000, 20)
    x_fit = np.linspace(100, 10000, 200)
    # y_vals = []
    # for index in range(len(titles)):
    #     y_vals.append([])
    #
    # for line_num, (L, B, non_stop_num) in enumerate([(2, 4, 0), (2, 4, 1), (4, 12, 4), (6, 20, 6)]):
    #     x = floor((B - L) / 2)
    #     m = 3
    #
    #     for index in range(len(titles)):
    #         y_vals[index].append([])
    #
    #     for traffic_flow in x_val:
    #         sim_result = simulate(lane_num=L, booth_num=B, interval=m, upper_extra_lane_num=x, non_stop_num=non_stop_num, traffic_flow=traffic_flow)
    #         out_count, average_speed, average_time_spent, crash_count, lane_change_count, vehicle_try_in = sim_result
    #         length = 21 + (B - L - x + 1) * m * 7
    #         area = (294 + 105 * (B - L) + 3.5 * m * (5 * B - 1.3 * L + 14) * (B - L - x))
    #         cost = 804.21 * area
    #         for index, target in enumerate([out_count, average_time_spent, average_time_spent / length, average_speed,
    #                                         lane_change_count / area, crash_count / area]):
    #             y_vals[index][line_num].append(target)
    #         print(L, B, x, m, "down")

    for index in range(len(titles)):
        fig, ax = plt.subplots()

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

        plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)

        for line_num, (L, B, non_stop_num) in enumerate([(2, 4, 0), (2, 4, 1), (4, 12, 4), (6, 20, 6)]):
            y_val = y_vals[index][line_num]
            y_bspline = interpolate.splev(x_fit, interpolate.splrep(x_val, y_val))  # B-spline
            y_poly_fit = np.polyval(np.polyfit(x_val, y_val, 2), x_fit)             # 多项式拟合
            plt.plot(x_val, y_val, "o", color=color_sequence[line_num])
            plt.plot(x_fit, y_bspline, color=color_sequence[line_num], label="L = {}, B = {}, electronic toll booths = {}".format(L, B, non_stop_num))

        ax.legend().get_frame().set_alpha(0.3)
        
        title = titles[index].format(L, B)
        plt.xlabel("Traffic flow ($vehicles/h$)")
        plt.ylabel(y_labels[index])
        plt.savefig(filename="./plot/{}".format(title))


def draw2_auto():
    # 自动汽车的影响
    titles = ["Throughput Under Different Autonomous Vehicles Ratio", "Average Time Spent Under Different Autonomous Vehicles Ratio",
              "Average Time Spent Per Unit Length Under Different Autonomous Vehicles Ratio",
              "Average Speed Under Different Autonomous Vehicles Ratio",
              "Lane Switches Per Unit Area Under Different Autonomous Vehicles Ratio",
              "Abrupt Brake Per Unit Area Under Different Autonomous Vehicles Ratio"]
    y_labels = ["Throughput ($vehicles/h$)", "Average time spent ($s$)", "Average time spent per unit length ($s/m$)",
                "Average speed ($m/s^2$)", "Lane switches per unit area ($1/m^2$)",
                "Abrupt brakes per unit area ($1/m^2$)", ]
    y_vals = pickle.load(open("data_draw2_auto.pkl", "rb"))

    x_val = np.linspace(0, 1, 4)
    x_fit = np.linspace(0, 1, 200)
    # y_vals = []
    # for index in range(len(titles)):
    #     y_vals.append([])
    #
    # for line_num, (L, B, non_stop_num) in enumerate([(2, 4, 0), (2, 4, 1), (4, 12, 4), (6, 20, 6)]):
    #     x = floor((B - L) / 2)
    #     m = 3
    #
    #     for index in range(len(titles)):
    #         y_vals[index].append([])
    #
    #     for auto_ratio in x_val:
    #         sim_result = simulate(lane_num=L, booth_num=B, interval=m, upper_extra_lane_num=x, non_stop_num=non_stop_num, auto_ratio=auto_ratio)
    #         out_count, average_speed, average_time_spent, crash_count, lane_change_count, vehicle_try_in = sim_result
    #         length = 21 + (B - L - x + 1) * m * 7
    #         area = (294 + 105 * (B - L) + 3.5 * m * (5 * B - 1.3 * L + 14) * (B - L - x))
    #         cost = 804.21 * area
    #         for index, target in enumerate([out_count, average_time_spent, average_time_spent / length, average_speed,
    #                                         lane_change_count / area, crash_count / area]):
    #             y_vals[index][line_num].append(target)
    #         print(L, B, x, m, "down")

    for index in range(len(titles)):
        fig, ax = plt.subplots()

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

        plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)

        for line_num, (L, B, non_stop_num) in enumerate([(2, 4, 0), (2, 4, 1), (4, 12, 4), (6, 20, 6)]):
            y_val = y_vals[index][line_num]
            # y_bspline = interpolate.splev(x_fit, interpolate.splrep(x_val, y_val))  # B-spline
            y_poly_fit = np.polyval(np.polyfit(x_val, y_val, 2), x_fit)             # 多项式拟合
            plt.plot(x_val, y_val, "o", color=color_sequence[line_num])
            plt.plot(x_fit, y_poly_fit, color=color_sequence[line_num], label="L = {}, B = {}, electronic toll booths = {}".format(L, B, non_stop_num))

        ax.legend().get_frame().set_alpha(0.3)
        
        title = titles[index].format(L, B)
        plt.xlabel("Proportion of autonomous vehicles")
        plt.ylabel(y_labels[index])
        plt.savefig(filename="./plot/{}".format(title))


def draw2_booth1():
    titles = ["Throughput With Different Proportions of Toll Booth",
              "Average Time Spent With Different Proportions of Toll Booth",
              "Average Time Spent Per Unit Length With Different Proportions of Toll Booth",
              "Average Speed With Different Proportions of Toll Booth",
              "Lane Switches Per Unit Area With Different Proportions of Toll Booth",
              "Abrupt Brake Per Unit Area With Different Proportions of Toll Booth"]
    y_labels = ["Throughput ($vehicles/h$)", "Average time spent ($s$)", "Average time spent per unit length ($s/m$)",
                "Average speed ($m/s^2$)", "Lane switches per unit area ($1/m^2$)",
                "Abrupt brakes per unit area ($1/m^2$)", ]

    y_vals = pickle.load(open("data_draw2_booth1.pkl", "rb"))

    # y_vals = []
    # for index in range(len(titles)):
    #     y_vals.append([])
    #
    # for line_num, (L, B) in enumerate([(2, 4), (4, 12), (6, 20)]):
    #     x = floor((B - L) / 2)
    #     m = 3
    #
    #     for index in range(len(titles)):
    #         y_vals[index].append([])
    #
    #     for non_stop_num in range(B+1):
    #         sim_result = simulate(lane_num=L, booth_num=B, interval=m, upper_extra_lane_num=x,
    #                               non_stop_num=non_stop_num)
    #         out_count, average_speed, average_time_spent, crash_count, lane_change_count, vehicle_try_in = sim_result
    #         length = 21 + (B - L - x + 1) * m * 7
    #         area = (294 + 105 * (B - L) + 3.5 * m * (5 * B - 1.3 * L + 14) * (B - L - x))
    #         cost = 804.21 * area
    #         for index, target in enumerate([out_count, average_time_spent, average_time_spent / length, average_speed,
    #                                         lane_change_count / area, crash_count / area]):
    #             y_vals[index][line_num].append(target)
    #         print(L, B, x, m, "down")

    for index in range(len(titles)):
        fig, ax = plt.subplots()

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

        plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)

        for line_num, (L, B) in enumerate([(2, 4), (4, 12), (6, 20)]):
            x_val = [non_stop_num / B for non_stop_num in range(B+1)]
            x_fit = np.linspace(x_val[0], x_val[-1], 200)
            y_val = y_vals[index][line_num]
            y_bspline = interpolate.splev(x_fit, interpolate.splrep(x_val, y_val))  # B-spline
            y_poly_fit = np.polyval(np.polyfit(x_val, y_val, 2), x_fit)  # 多项式拟合
            plt.plot(x_val, y_val, "o", color=color_sequence[line_num])
            plt.plot(x_fit, y_bspline, color=color_sequence[line_num],
                     label="L = {}, B = {}".format(L, B))
        plt.subplots_adjust(bottom = 0.15)
        ax.legend().get_frame().set_alpha(0.3)
        title = titles[index].format(L, B)
        plt.xlabel("Proportion of electronic toll collection booths\nshifting from conventional tollbooths")
        plt.ylabel(y_labels[index])
        plt.savefig(filename="./plot/{}".format(title))


def draw2_booth2():
    titles = ["Throughput With Different Proportions of Toll Booth (from ele)",
              "Average Time Spent With Different Proportions of Toll Booth (from ele)",
              "Average Time Spent Per Unit Length With Different Proportions of Toll Booth (from ele)",
              "Average Speed With Different Proportions of Toll Booth (from ele)",
              "Lane Switches Per Unit Area With Different Proportions of Toll Booth (from ele)",
              "Abrupt Brake Per Unit Area With Different Proportions of Toll Booth (from ele)"]
    y_labels = ["Throughput ($vehicles/h$)", "Average time spent ($s$)", "Average time spent per unit length ($s/m$)",
                "Average speed ($m/s^2$)", "Lane switches per unit area ($1/m^2$)",
                "Abrupt brakes per unit area ($1/m^2$)", ]

    y_vals = pickle.load(open("data_draw2_booth2.pkl", "rb"))

    # y_vals = []
    # for index in range(len(titles)):
    #     y_vals.append([])
    #
    # for line_num, (L, B) in enumerate([(2, 4), (4, 12), (6, 20)]):
    #     x = floor((B - L) / 2)
    #     m = 3
    #
    #     for index in range(len(titles)):
    #         y_vals[index].append([])
    #
    #     for non_stop_num in range(B+1):
    #         sim_result = simulate(lane_num=L, booth_num=B, interval=m, upper_extra_lane_num=x,
    #                               ele_num=B-non_stop_num, non_stop_num=non_stop_num)
    #         out_count, average_speed, average_time_spent, crash_count, lane_change_count, vehicle_try_in = sim_result
    #         length = 21 + (B - L - x + 1) * m * 7
    #         area = (294 + 105 * (B - L) + 3.5 * m * (5 * B - 1.3 * L + 14) * (B - L - x))
    #         cost = 804.21 * area
    #         for index, target in enumerate([out_count, average_time_spent, average_time_spent / length, average_speed,
    #                                         lane_change_count / area, crash_count / area]):
    #             y_vals[index][line_num].append(target)
    #         print(L, B, x, m, "down")

    for index in range(len(titles)):
        fig, ax = plt.subplots()

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

        plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)

        for line_num, (L, B) in enumerate([(2, 4), (4, 12), (6, 20)]):
            x_val = [non_stop_num / B for non_stop_num in range(B+1)]
            x_fit = np.linspace(x_val[0], x_val[-1], 200)
            y_val = y_vals[index][line_num]
            # y_bspline = interpolate.splev(x_fit, interpolate.splrep(x_val, y_val))  # B-spline
            y_poly_fit = np.polyval(np.polyfit(x_val, y_val, 2), x_fit)  # 多项式拟合
            plt.plot(x_val, y_val, "o", color=color_sequence[line_num])
            plt.plot(x_fit, y_poly_fit, color=color_sequence[line_num],
                     label="L = {}, B = {}".format(L, B))
        plt.subplots_adjust(bottom=0.15)
        ax.legend().get_frame().set_alpha(0.3)
        title = titles[index].format(L, B)
        plt.xlabel("Proportion of electronic toll collection booths\nshifting from exact-change tollbooths")
        plt.ylabel(y_labels[index])
        plt.savefig(filename="./plot/{}".format(title))


def main():
    simulate(lane_num=4, booth_num=6, non_stop_num=2)
    # loop(True, 3)
    # evaluate(3)
    # draw1()
    # draw2_traffic()
    # draw2_auto()
    # draw2_booth1()
    # draw2_booth2()

if __name__ == '__main__':
    main()
