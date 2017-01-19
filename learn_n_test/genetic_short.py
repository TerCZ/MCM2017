import random
import numpy
from deap import base, creator, tools, algorithms
from string import ascii_letters


target = list("computer science is brain fuck")
charset = ascii_letters + ' .:/'

# 自定义类
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# 注册函数
toolbox = base.Toolbox()
toolbox.register("attr_string", random.choice, charset)  # 生成参数
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_string, len(target))  # 生成个体
toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # 生成种群


def eval(individual):
    return sum([a == b for a, b in zip(individual, target)]),

# 注册算符
toolbox.register("evaluate", eval)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=2)


def main():
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.05, ngen=100, stats=stats, halloffame=hof, verbose=True)
    print(hof)


if __name__ == '__main__':
    main()