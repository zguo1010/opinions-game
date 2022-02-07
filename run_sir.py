#!/usr/bin/env python
# coding: utf-8

import scipy.integrate as spi
import numpy as np
import matplotlib.pyplot as plt
import time
import pylab as pl


def diff_eqs(INP, t, beta, gamma):
    '''The main set of equations'''
    Y = np.zeros((3))
    V = INP
    Y[0] = - beta * V[1] * V[0]
    Y[1] = beta * V[1] * V[0] - gamma * V[1]
    Y[2] = gamma * V[1]
    return Y  # For odeint


def dynamic_parameter_diff_eqs(INP, max_t, beta_list, gamma_list):
    # assert len(beta_list) == max_t
    Y = np.zeros((max_t, 3))
    Y[0] = INP
    for t in range(1, max_t):
        Y_t = spi.odeint(func=diff_eqs, y0=Y[t - 1], t=[1, 2], args=(beta_list[t], gamma_list[t]))
        Y[t] = Y_t[1]
    return Y


def SIR_approximate_gradient(Y_true, INP, max_t, beta_list, gamma_list, progress_coefficient=1.01):
    Y_true_crop = Y_true[0:max_t]
    Y_old = dynamic_parameter_diff_eqs(INP, max_t, beta_list, gamma_list)

    beta_new = beta_list * progress_coefficient
    gamma_new = gamma_list * progress_coefficient

    Y_new = dynamic_parameter_diff_eqs(INP, max_t, beta_new, gamma_new)

    obj_diff = np.square(Y_old[:, 1] - Y_true_crop[:, 1]) - np.square(Y_new[:, 1] - Y_true_crop[:, 1])
    obj_diff_sum = np.mean(obj_diff)


    obj_diff_gamma = np.square(Y_old[:, 2] - Y_true_crop[:, 2]) - np.square(Y_new[:, 2] - Y_true_crop[:, 2])
    obj_diff_sum_gamma = np.mean(obj_diff_gamma)

    beta_grad = (obj_diff + obj_diff_gamma) / (beta_list - beta_new) / 2
    gamma_grad = (obj_diff + obj_diff_gamma) / (gamma_list - gamma_new) / 2

    print("sum i:%f \t sum gamma:%f" % (obj_diff_sum, obj_diff_sum_gamma))
    return beta_grad, gamma_grad


def gradient_descent(Y_true, INP, max_t, beta_list, gamma_list, beta_low, beta_high, gamma_low, gamma_high, lr,
                     max_iters=5000, progress_coefficient=1.01):
    for it in range(0, max_iters):
        print("iteration: %d" % it)
        beta_grad, gamma_grad \
            = SIR_approximate_gradient(Y_true, INP, max_t, beta_list, gamma_list, progress_coefficient)

        beta_list = beta_list - lr * beta_grad
        gamma_list = gamma_list - lr * gamma_grad  # 1e-10 gamma only

        beta_list = np.clip(beta_list, beta_low, beta_high)
        gamma_list = np.clip(gamma_list, gamma_low, gamma_high)

    # print("beta:%f \t gamma:%f" %(beta, gamma))
    print("beta:%s \t gamma:%s" % (list(beta_list), list(gamma_list)))

    Y = dynamic_parameter_diff_eqs(INPUT, max_t, beta_list, gamma_list)
    return Y, beta_list, gamma_list


TS = 1.0
S0 = 800
I0 = 100
R0 = 100
N = 1000  # N will reduce
max_t = 100
# beta = 20 #np.random.random()
beta_0_l, beta_0_r = 0.02, 0.03
gamma_0_l, gamma_0_r = 0.01, 0.09
beta_list = np.random.uniform(beta_0_l, beta_0_r, max_t)
gamma_list = np.random.uniform(gamma_0_l, gamma_0_r, max_t)
# gamma = 0.005
beta_low, beta_high = 0.000001, 0.0008
gamma_low, gamma_high = 0.001, 0.1
lr = 5e-9

INPUT = (S0, I0, R0)

dt = '12_29'
df = 'k'  # "c"/"k"
tm = int(time.time())
models = ['e']
for o in models:
    Y_true = np.zeros((200, 3))
    file = open('results/%s/sir_k_%s.txt' % (dt, o), 'r')
    Y_true[0] = np.array([S0, I0, R0])
    num = 1
    while num < 200:
        line = file.readline()
        linespl = line.split(' [')[1]
        raw = [float(x) for x in linespl[:-2].split(', ')]
        Y_true[num] = np.array([raw[0], raw[1] + raw[3], raw[2]+100])
        num += 1
    print(Y_true)
    Y_estimate, beta_estimate, gamma_estimate = \
        gradient_descent(Y_true, INPUT, max_t, beta_list, gamma_list, beta_low, beta_high, gamma_low, gamma_high, lr)
    # print('Y_estimate:', list(Y_estimate))

    # save parameters to a file
    with open('results/parameters_range_%s_%d.txt' % (o, tm), 'w') as file:
        file.write("RECORD RANGE for %s\n" % o)
        file.write("INITIAL b c:  %s %s %s %s\n" % (beta_0_l, beta_0_r, gamma_0_l, gamma_0_r))
        file.write("CLIP b c:  %s %s, %s %s\n" % (beta_low, beta_high, gamma_low, gamma_high))
        file.write("LR %s\n" % lr)
        file.write("FINAL b c:  %s %s\n" % (list(beta_estimate), list(gamma_estimate)))
        for line in Y_true:
            file.write("%s\n" % list(line))
    file.close()


plt.figure(figsize=(5.4, 4.2))
plt.gcf().subplots_adjust(left=0.18)
plt.gcf().subplots_adjust(right=0.99)
plt.gcf().subplots_adjust(bottom=0.16)
plt.gcf().subplots_adjust(top=1)
colors = ['blue', 'red', 'green', 'yellow']
plt.plot(Y_estimate[:max_t, 0], 'o', color=colors[2], markersize=1, label='S')
plt.plot(Y_estimate[:max_t, 1], 'o', color=colors[1], markersize=1, label='I')
plt.plot(Y_estimate[:max_t, 2], 'o', color=colors[0], markersize=1, label='R')

# plt.xlabel('number of interactions per node', fontsize=22, x=0.45)
plt.xlabel('beta: %f, gamma: %f' % (beta_estimate[-1], gamma_estimate[-1]), fontsize=22)
plt.ylabel('SIR frequency', fontsize=22)
plt.tick_params(axis='both', labelsize=18)
plt.legend(markerscale=4, fontsize=16)
plt.savefig('results/sir_est_%s.png' % models[0])
# plt.show()
plt.clf()

plt.figure(figsize=(5.4, 4.2))
plt.gcf().subplots_adjust(left=0.18)
plt.gcf().subplots_adjust(right=0.99)
plt.gcf().subplots_adjust(bottom=0.16)
plt.gcf().subplots_adjust(top=1)
colors = ['blue', 'red', 'green']
plt.plot(Y_true[:max_t, 0], 'o', color=colors[2], markersize=1, label='S')
plt.plot(Y_true[:max_t, 1], 'o', color=colors[1], markersize=1, label='I')
plt.plot(Y_true[:max_t, 2], 'o', color=colors[0], markersize=1, label='R')
plt.plot(Y_estimate[:max_t, 0], '|', color=colors[2], alpha=0.8, markersize=2, label='S-sim')
plt.plot(Y_estimate[:max_t, 1], '|', color=colors[1], alpha=0.8, markersize=2, label='I-sim')
plt.plot(Y_estimate[:max_t, 2], '|', color=colors[0], alpha=0.8, markersize=2, label='R-sim')

plt.xlabel('number of interactions per node', fontsize=22, x=0.45)
plt.ylabel('SIR frequency', fontsize=22)
plt.tick_params(axis='both', labelsize=18)
plt.legend(markerscale=4, fontsize=12)
plt.savefig('results/sir_cmp_%s_%s.png' % (models[0], tm))
# plt.show()
plt.clf()