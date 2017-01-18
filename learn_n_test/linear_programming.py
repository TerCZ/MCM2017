import numpy as np
import matplotlib.pyplot as plt
import pulp  # 线性规划包


def example1():
    # 创建问题
    prob = pulp.LpProblem("example1", pulp.LpMaximize)

    # 添加变量，指定上下界
    x0 = pulp.LpVariable("x0", cat=pulp.LpInteger)  # 指定为整数
    x1 = pulp.LpVariable("x1", -3)

    # 添加目标，这应该是+=运算符的第一个输入
    prob += -x0 + 4*x1, "这段话解释目标方程的含义"

    # 添加限制
    prob += -3*x0 + x1 <= 6
    prob += x0 + 2.23*x1 <= 4

    # 用PuLP选择的solver求解
    prob.solve()

    # 打印结果
    print("Status:", pulp.LpStatus[prob.status])
    for v in prob.variables():
        print(v.name, v.varValue)
    print(pulp.value(prob.objective))


def example2():
    # 输入参数
    actions = ["sleep", "game", "study"]
    energy = dict(sleep=0.3, game=0.5, study=0.8)
    happiness = dict(sleep=0.4, game=1, study=-0.2)
    fulfilment = dict(sleep=0.2, game=-0.1, study=0.6)

    # 创建问题
    prob = pulp.LpProblem("example2", pulp.LpMinimize)

    # 添加变量
    variables = pulp.LpVariable.dicts("action", actions, 0)

    # 添加目标
    prob += pulp.lpSum([energy[action]*variables[action] for action in actions]), "所有活动消耗的精力"

    # 添加限制
    prob += pulp.lpSum([happiness[action]*variables[action] for action in actions]) >= 10
    prob += pulp.lpSum([fulfilment[action] * variables[action] for action in actions]) >= 5

    # 用PuLP选择的solver求解
    prob.solve()

    # 打印结果
    print("Status:", pulp.LpStatus[prob.status])
    for v in prob.variables():
        print(v.name, v.varValue)
    print(pulp.value(prob.objective))


def example3():
    """
    set partition问题，但可以直接穷举法而跳过线性规划包装
    """

    max_tables = 5
    max_table_size = 4
    guests = "A B C D E F G I J K L M N O P Q R".split()

    def happiness(table):
        """
        Find the happiness of the table
        - by calculating the maximum distance between the letters
        """
        return abs(ord(table[0]) - ord(table[-1]))

    # 创建变量——所有的分组方案
    possible_tables = [tuple(c) for c in pulp.allcombinations(guests, max_table_size)]

    # 使每个分组方案对应变量只能去0或1
    x = pulp.LpVariable.dicts("table", possible_tables, lowBound=0, upBound=1, cat=pulp.LpInteger)

    # 创建问题
    seating_model = pulp.LpProblem("Wedding Seating Model", pulp.LpMinimize)

    # 添加目标
    seating_model += sum([happiness(table) * x[table] for table in possible_tables])

    # 限制桌数
    seating_model += sum([x[table] for table in possible_tables]) <= max_tables, "Maximum_number_of_tables"

    # 限制每个客人在且只在一个桌上出现
    for guest in guests:
        seating_model += sum([x[table] for table in possible_tables if guest in table]) == 1, "Must_seat_%s" % guest

    # 求解
    seating_model.solve()

    print("The choosen tables are out of a total of %s:" % len(possible_tables))
    for table in possible_tables:
        if x[table].value() == 1.0:
            print(table)


if __name__ == '__main__':
    example3()
