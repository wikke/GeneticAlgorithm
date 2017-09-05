import numpy as np
import random
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

POPULATION = 10
SURVIVAL_NUM = 4

POINTS = 64
X_LIM = 4
Y_LIM = 2
uniform_x = [(i/POINTS)*X_LIM*2-X_LIM for i in range(POINTS)]
target_y = norm.pdf(uniform_x, loc=0, scale=1) * 4

PLOT_NUM = 0
def draw(y, x=uniform_x, title=''):
    global PLOT_NUM
    PLOT_NUM += 1

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    ax[0].set(xlim=[-X_LIM, X_LIM], ylim=[-Y_LIM, Y_LIM], title='target')
    ax[0].plot(uniform_x, target_y)

    ax[1].set(xlim=[-X_LIM, X_LIM], ylim=[-Y_LIM, Y_LIM], title=title)
    ax[1].plot(x, y)

    plt.show(block=False)

def init(n):
    return np.random.uniform(low=0, high=0.01, size=(POPULATION, POINTS))

def adaptation(p):
    return mean_squared_error(p, target_y)

def produce(parents):
    son = np.zeros((POINTS,))
    for i in range(POINTS):
        son[i] = parents[random.randint(0,1)][i]

        # variation
        if random.random() < 0.05:
            son[i] += (random.random() - 0.5) / 20

    return son

race = init(POPULATION)
for step in range(100000):
    scores = []
    for i in range(POPULATION):
        scores.append((adaptation(race[i]), i))

    scores.sort(reverse=False)

    if (step+1) % 500 == 0:
        draw(race[scores[0][1]], title='mse {}'.format(scores[0][0]))
        if scores[0][0] < 0.00001:
            input('Already good enough, Enter to exit')
            break

    next_race, survivals = [], []
    for i in range(SURVIVAL_NUM):
        idx = scores[i][1]
        survivals.append(race[idx])
    next_race.extend(survivals)

    for _ in range(POPULATION - SURVIVAL_NUM):
        son = produce(random.choices(survivals, k=2))
        next_race.append(son)

    race = next_race
