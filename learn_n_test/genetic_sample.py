from string import ascii_lowercase
from random import choice, random


target = list("really really fucking amazing")
charset = ascii_lowercase + ' .:/'
parent = [choice(charset) for _ in range(len(target))]
minmutaterate = .09
C = range(100)
perfectfitness = len(target)


def fitness(trial):
    return sum(t == h for t, h in zip(trial, target))


def mutaterate(parent):
    return 1.0-((perfectfitness - fitness(parent)) / perfectfitness * (1.0 - minmutaterate))


def mutate(parent, rate):
    return [(ch if random() <= rate else choice(charset)) for ch in parent]


def log(iterations, rate, parent):
    print("#%-4i, rate: %4.3f, fitness: %4.1f%%, '%s'" % (iterations, rate, fitness(parent)*100./perfectfitness, ''.join(parent)))


iterations = 0
while parent != target:
    rate = mutaterate(parent)
    iterations += 1
    # if iterations % 10 == 0:
    #     log(iterations, rate, parent)
    copies = [ mutate(parent, rate) for _ in C ]  + [parent]
    parent = max(copies, key=fitness)
print()
log(iterations, rate, parent)
