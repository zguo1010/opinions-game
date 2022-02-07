#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from numpy.random import randint, choice, seed, shuffle
import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.community as nx_m
import warnings
import math, time
from datetime import date
import random, sys, os
from operator import itemgetter
plt.switch_backend('agg') #very important for cluster to see real-time result

wt = np.array([0.998, 0.001, 0.001, 1])  # b, d, u, a
wf = np.array([0.001, 0.998, 0.001, 0])  # 0.01
# strategy choices for attackers, defenders, and users
# choices: A-k, D-l, U-m
strategy = {"A": ['DG', 'C', 'DN', 'S'],
            "DF": ['T', 'M'],
            "U": ['SU', 'U', 'NU'], "H": ['SU', 'U', 'NU'],
            "AS": ['SU', 'U', 'NU'], "D": ['SU', 'U', 'NU'],
            "E": ['SU', 'U', 'NU'], "T": ['SU', 'U', 'NU']}


def initialization(likers, topics, N, pa, pt, phos, mup, sigmap, mur, sigmar):
    # fix seed for N random nodes and assign attackers and true informers
    seed_i = 375
    seed(seed_i) #375
    print('seed: ', seed_i)
    legit_data = likers.loc[likers['label'] == 'Legit', :]
    legit_sample = choice(legit_data.index, N, replace=False)
    likers_s = likers.loc[sorted(legit_sample)]
    topics_s = topics.loc[sorted(legit_sample)]

    # create graph and set initial features
    G = nx.empty_graph()
    # nx.set_node_attributes(G, 0, 'feeding') #P^f
    # nx.set_node_attributes(G, 0, 'posting') #P^p
    for i in likers_s.index.values:
        G.add_node(i)
        # G.nodes[i]['friends'] = max(likers_s.loc[i, 'friends'], 5)
        G.nodes[i]['friends'] = likers_s.loc[i, 'friends']
        G.nodes[i]['feeding'] = likers_s.loc[i, 'feeding']
        G.nodes[i]['posting'] = likers_s.loc[i, 'posting']
        G.nodes[i]['inviting'] = likers_s.loc[i, 'inviting']
        # add other features if possible
    nx.set_node_attributes(G, '', 'user_type')  # A U H T
    nx.set_node_attributes(G, 0, 'omega')  # b, d, u, a, changes with time t
    nx.set_node_attributes(G, 0, 'observation')  # r, s, W #useless
    nx.set_node_attributes(G, '', 'evidence_strategy')  # evidence for strategies
    nx.set_node_attributes(G, '', 'utility_strategy')  # utility from payoff function
    nx.set_node_attributes(G, 0, 'strategy')  # prob for each strategy evidence
    nx.set_node_attributes(G, 0, 'belief')  # P(b)
    nx.set_node_attributes(G, 0, 'disbelief')  # P(d)
    nx.set_node_attributes(G, 0, 'evidence_share')  # nP^f, nP^p
    nx.set_node_attributes(G, 1, 'phi')  # friendship threshold
    nx.set_node_attributes(G, 1, 'rho')  # report threshold
    nx.set_node_attributes(G, 0, 'capital')  # structural capital
    nx.set_node_attributes(G, 0, 'redundancy')  # redundancy
    nx.set_node_attributes(G, -1, 'target')  # the chosen opponent
    nx.set_node_attributes(G, '', 'choice')  # the chosen strategy
    nx.set_node_attributes(G, 0, 'update')  # the expected opinion change
    nx.set_node_attributes(G, 0, 'report')  # user can report the different opinion
    nx.set_node_attributes(G, False, 'full')  # added enough friends
    nx.set_node_attributes(G, False, 'remove_node')  # defender remove node
    nx.set_node_attributes(G, '', 'remove_edge')  # user remove friendship
    nx.set_node_attributes(G, np.empty(0), 'nash_utility')

    # assgin A T to high degree nodes
    degrees = [(x, y) for (x, y) in list(G.nodes(data='friends')) if x in legit_sample]
    sort_degrees = {k: v for (k, v) in sorted(degrees, key=lambda x: x[1], reverse=True)}
    data_keys = list(sort_degrees.keys())
    data_ta = data_keys[: int(N * pa) + int(N * pt)]
    node_attacker = choice(data_ta, int(N * pa), replace=False)
    node_true = [x for x in data_ta if x not in node_attacker]
    node_h = choice([x for x in data_keys if x not in data_ta], int(N * (1 - (pa + pt)) * phos), replace=False)
    if phos == 1:
        node_u, node_e, node_a, node_d = [], [], [], []
    else:
        node_os = [node for node in list(G.nodes()) if node not in node_attacker and
                  node not in node_true and node not in node_h]
        shuffle(node_os)
        node_len = len(node_os)
        node_u = node_os[:int(node_len/4)]
        node_a = node_os[int(node_len/4) + 1: int(node_len/2)]
        node_d = node_os[int(node_len/2) + 1: int(3*node_len/4)]
        node_e = node_os[int(3*node_len/4) + 1:]

    print('attackers', sorted(node_attacker))
    print('true informers', sorted(node_true))
    # print('U users', sorted(node_u))
    print('H users', sorted(node_h))
    for i in G.nodes():
        if i in node_attacker:
            G.nodes[i]['user_type'] = 'A'
            G.nodes[i]['observation'] = [1, 998, 1]
            G.nodes[i]['evidence_share'] = [1, 1, 0, 0]
            G.nodes[i]['strategy'] = {x: 0.25 for x in strategy['A']}
            G.nodes[i]['evidence_strategy'] = {x: 1 for x in strategy['A']}
            G.nodes[i]['utility_strategy'] = {x: 0 for x in strategy['A']}
            # other codes/features
        elif i in node_true:
            G.nodes[i]['user_type'] = 'T'
            G.nodes[i]['observation'] = [998, 1, 1]
            G.nodes[i]['evidence_share'] = [1, 1, 0, 0]
            G.nodes[i]['evidence_strategy'] = {x: 1 for x in strategy['T']}
            G.nodes[i]['utility_strategy'] = {x: 0 for x in strategy['T']}
            G.nodes[i]['strategy'] = {x: 0.33 for x in strategy['T']}
            G.nodes[i]['choice'] = 'NU'
        else:
            G.nodes[i]['observation'] = [1, 1, 998]
            G.nodes[i]['evidence_share'] = [max(N * 0.01, int(N * G.nodes[i]['feeding'])),
                                            max(N * 0.01, int(N * G.nodes[i]['posting'])), 0, 0]
            G.nodes[i]['strategy'] = {x: 0.33 for x in strategy['H']}
            G.nodes[i]['evidence_strategy'] = {x: 1 for x in strategy['H']}
            G.nodes[i]['utility_strategy'] = {x: 0 for x in strategy['H']}
            if i in node_h:
                G.nodes[i]['user_type'] = 'H'
            elif i in node_u:
                G.nodes[i]['user_type'] = 'U'
            elif i in node_a:
                G.nodes[i]['user_type'] = 'AS'
            elif i in node_d:
                G.nodes[i]['user_type'] = 'D'
            else:
                G.nodes[i]['user_type'] = 'E'

    phi = np.random.normal(mup, sigmap, N)
    rho = np.random.normal(mur, sigmar, N)
    # Initial omega values
    opinion(G)
    c = 0
    for i in G.nodes():
        omega = G.nodes[i]['omega'][:]
        G.nodes[i]['belief'] = omega[0] + omega[2] * omega[3]
        G.nodes[i]['disbelief'] = omega[1] + omega[2] * (1 - omega[3])
        G.nodes[i]['phi'] = max(0.01, phi[c])
        G.nodes[i]['rho'] = max(0.01, rho[c])
        c += 1

    return G, topics_s, {'T': node_true, 'A': node_attacker, 'H': node_u, 'U': node_h,
                         'AS': node_a, 'D': node_d, 'E': node_e}, legit_sample


def opinion(G):
    '''calculate opinion based on observations'''
    for i in G.nodes():
        obs = G.nodes[i]['observation']
        total = obs[0] + obs[1] + obs[2]
        if G.nodes[i]['user_type'] == 'A':
            a = 0
        elif G.nodes[i]['user_type'] == 'T':
            a = 1
        else:
            a = 0.5
        G.nodes[i]['omega'] = np.array([obs[0] / total, obs[1] / total, obs[2] / total, a])
    return


def maximum_cap(data, feature='freq_posts', thresh=0.05, binnum=10):
    '''change max values'''
    result1 = np.histogram(data.loc[data['label'] == "Fake", :][feature], bins=binnum)
    cnt1 = sum(data['label'] == "Fake")
    xmax1 = result1[1][-1]

    # Legit class
    result2 = np.histogram(data.loc[data['label'] == "Legit", :][feature], bins=binnum)
    cnt2 = sum(data['label'] == "Legit")
    xmax2 = result2[1][-1]

    # set x-axix cut
    xval = max(result1[1][sum(result1[0] >= cnt1 * thresh)],
               result2[1][sum(result2[0] >= cnt2 * thresh)])
    data.loc[data[feature] > xval, feature] = xval
    return


def projected_belief(w):
    return w[0] + w[2] * w[3], w[1] + w[2] * (1 - w[3])


def vacuity_opinion(w1):
    '''equation 5'''
    if w1[2] >= 0.05:  # low u will maximize vacuity
        return w1[2]
    pb = projected_belief(w1)
    if pb[0] <= 0.001 or pb[1] <= 0.001:
        return w1[2]
    # p(b) and p(d) are non-zero
    if w1[3] <= 0.001:
        return pb[1]
    elif w1[3] >= 0.999:
        return pb[0]
    else:
        return min(pb[0] / w1[3], pb[1] / (1 - w1[3]))


def uncertainty_discount(w1, w2):
    '''equation 6'''
    u1 = vacuity_opinion(w1)
    u2 = vacuity_opinion(w2)
    return max(0, (1 - u1) * (1 - u2))


def homophily_discount(w1, w2):
    '''equation 7'''
    numerator = w1[0] * w2[0] + w1[1] * w2[1]
    if numerator == 0:
        return 1.0
    denom = math.sqrt(w1[0] ** 2 + w1[1] ** 2) * math.sqrt(w2[0] ** 2 + w2[1] ** 2)
    return min(1, numerator / denom)


def opinion_consensus(w1, w2, method):
    '''equation 9 , i, j'''
    c = 1  # safe for true informer
    if method == 'U':
        c = uncertainty_discount(w1, w2)
    elif method == 'H':
        c = homophily_discount(w1, w2)
    else:  # for 'T'
        print('consensus appears T: ', w1, w2)
        return {}
    if w1[2] <= 0.001 and w2[2] <= 0.001:  # very small u don't update
        return w1
    beta = 1 - c * (1 - w1[2]) * (1 - w2[2])  # beta = 1 - c*(1-ui)(1-uj) #ui uj cannot be 1 at the same time
    if beta <= 0.001:
        return w1[:]
    b = (w1[0] * (1 - c * (1 - w2[2])) + c * w2[0] * w1[2]) / beta  # b =
    d = (w1[1] * (1 - c * (1 - w2[2])) + c * w2[1] * w1[2]) / beta
    u2 = w1[2] * (1 - c * (1 - w2[2])) / beta  # u = 1 - b - d
    if u2 <= 0.001:  # very small u is 0
        u2 = 0
    if w1[2] >= 0.999 and w2[2] >= 0.999:  # two users
        a = (w1[3] + w2[3]) / 2
    a = ((w1[3] - w1[3] * w1[2] - w2[3] * w1[2]) * (1 - c * (1 - w2[2])) + w2[3] * w1[2]) / (
            beta - w1[2] * (1 - c * (1 - w2[2])))
    return np.array([b, d, u2, a])  # w1[3]


def opinion_difference(w1, w2):
    pd = abs(w1[0] - w2[0]) + abs(w1[1] - w2[1])
    return pd / 2


def opinion_encounter(w1, w2):
    '''directly consensus w1 and w2, simple Equation 9'''
    beta = w1[2] + w2[2] - w1[2] * w2[2]
    b = (w1[0] * w2[2] + w2[0] * w1[2]) / beta
    d = (w1[1] * w2[2] + w2[1] * w1[2]) / beta
    u = w1[2] * w2[2] / beta
    a = (w1[3] * w2[2] + w2[3] * w1[2] - (w1[3] + w2[3]) * w1[2] * w2[2]) / (w1[2] + w2[2] - 2 * w1[2] * w2[2])
    return np.array([b, d, u, a])


def opinion_assertion(w1, w2):
    '''using equation 22, each item is bounded by (0,1)'''
    b = w1[0] + w2[0] * (1 - w1[0])
    d = w1[1] + w2[1] * (1 - w1[1])
    u = w1[2] + w2[2] * (1 - w1[2])
    total = b + d + u
    a = min(1, w1[3] + w2[0] * (w2[3] - 0.5) * (1 - math.fabs(2 * w1[3] - 1)))
    return np.array([b / total, d / total, u / total, a])


def opinion_herding(G, i, w1):  # user i with w1
    '''using equation 23'''
    b = 0
    d = 0
    a = 0
    if len(list(G.neighbors(i))) == 0:
        return w1
    for j in list(G.neighbors(i)):
        w2 = G.nodes[j]['omega']
        b = b + (1 - w2[2]) * (w2[0] - w1[0])
        d = d + (1 - w2[2]) * (w2[1] - w1[1])
        a = a + (1 - w2[2]) * (w2[3] - w1[3])
    b = min(w1[0] + w1[2] / len(list(G.neighbors(i))) * b, 1)
    d = min(w1[1] + w1[2] / len(list(G.neighbors(i))) * d, 1 - b)
    u = 1 - b - d
    a = min(w1[3] + w1[2] / len(list(G.neighbors(i))) * a, 1)
    return np.array([b, d, u, a])  # a


def opinion_herding_pair(w1, w2):
    b = min(w1[0] + (1 - w2[2]) * (w2[0] - w1[0]), 1)
    d = min(w1[1] + (1 - w2[2]) * (w2[1] - w1[1]), 1 - b)
    u = 1 - b - d
    a = min(w1[3] + (1 - w2[2]) * (w2[3] - w1[3]), 1)
    return np.array([b, d, u, a])  # a


# # Payoffs

# In[31]:

def imperfect_observation(d, a=0.9):
    '''Users observe opponent's strategies with 90% accuracy'''
    new_obs = {}
    for k, v in d.items():
        new_obs[k] = v * (1 + (random.random() - 0.5) / 5)
    return new_obs


# In[32]:

def attacker_opinion(G, i, k):
    '''Attacker i choose k strategy, k = DG, C, DN, or S'''
    if k == 'DG':
        return np.array([0.001, 0.001, 0.998, 0.5])  # initial user's opinion
    elif k == 'S':
        return wf[:]
    else:  # receive an opinion
        wn = G.nodes[i]['omega'][:]  # from last user opponent
        if k == 'C':  # reverse the b and d
            return np.array([wn[1], wn[0], wn[2], wn[3]])
        if k == 'DN':  # not forwarding b, true info to friends
            return np.array([0, wn[1], wn[0] + wn[2], wn[3]])
        return wn


# In[33]:

def all_similarity(G, tf):
    count = 0
    sim = 0
    for i in G.nodes():
        method = G.nodes[i]['user_type']
        if method in ['H', 'U', 'AS', 'D', 'E']:
            w = G.nodes[i]['omega'][:]
            if tf == 'True':
                sim += homophily_discount(w, wt)
            else:
                sim += homophily_discount(w, wf)
            count += 1
    return sim / count


# In[34]:

def all_similarity_attacks(G, w1_new):
    count = 0
    sim = 0
    for i in G.nodes():
        method = G.nodes[i]['user_type']
        if method in ['H', 'U', 'AS', 'D', 'E']:
            w2 = G.nodes[i]['omega'][:]
            if method in ['H', 'U']:
                w2_new = opinion_consensus(w2, w1_new, method)
            elif method == 'D':
                # w2_new = opinion_herding(G, i, w2)
                w2_new = opinion_herding_pair(w2, w1_new)
            elif method == 'AS':
                w2_new = opinion_assertion(w2, w1_new)
            elif method == 'E':
                w2_new = opinion_encounter(w2, w1_new)
            sim += homophily_discount(w2_new, wt)
            count += 1
    return sim / count


# In[35]:

def strategy_attacker_nash(G, i, j, defender_strategy, flag, phos):
    '''Attacker i uses strategy k, against a user j'''
    #update attacker nash matrix
    w1 = G.nodes[i]['omega'][:] #attacker
    w2 = G.nodes[j]['omega'][:] #user
    method = G.nodes[j]['user_type'] #no guess
    if flag == 1 and w2[2] < 0.95: #and w1[1]>0.5
        flag = 1
    else:
        flag = 0
    for k in strategy['A']:
        w1_new = attacker_opinion(G, i, k)
        gl = all_similarity_attacks(G, w1_new)  # all users are attacked by attacker
        for m in ['SU', 'U', 'NU']: #
            ds = 0
            if method == 'T': #true informer guess a type
                w2_new = w2[:]
            if method in ['U', 'H']:
                w2_new = opinion_consensus(w2, w1_new, method)
            elif method == 'D':
                w2_new = opinion_herding_pair(w2, w1_new)
            elif method == 'AS':
                w2_new = opinion_assertion(w2, w1_new)
            elif method == 'E':
                w2_new = opinion_encounter(w2, w1_new)
            if m != 'NU':
                ds = homophily_discount(w2_new, wf) - homophily_discount(w2, wf)
            G.nodes[i]['nash_utility'][strategy['A'].index(k)][strategy['U'].index(m)] = ds-gl #* j_strategy[m]

    for m in strategy['H']:
        if m != 'NU' and method != 'T':  # true informer never update
            for k in ['DG', 'C', 'DN', 'S']:
                w1_new = attacker_opinion(G, j, k)
                if method in ['U', 'H']:
                    w2_new = opinion_consensus(w2, w1_new, method)
                elif method == 'D':
                    w2_new = opinion_herding_pair(w2, w1_new)
                elif method == 'AS':
                    w2_new = opinion_assertion(w2, w1_new)
                elif method == 'E':
                    w2_new = opinion_encounter(w2, w1_new)
                else:
                    w2_new = w2[:]
                s = homophily_discount(w2_new, wf)
                G.nodes[j]['nash_utility'][strategy['A'].index(k)][strategy['U'].index(m)] = s #* pka[k]

    # make nash choices based on two matrix nash_utility, find a consensus between row and column players
    nash_list_row = []
    nash_list_col = []
    for p in range(4):
        row_max = max(G.nodes[j]['nash_utility'][p])
        for q in range(3):
            if G.nodes[j]['nash_utility'][p][q] == row_max:
                nash_list_row.append((p, q))
    for q in range(3):
        col_max = max(G.nodes[i]['nash_utility'][:, q])
        for p in range(4):
            if G.nodes[i]['nash_utility'][p][q] == col_max:
                if (p, q) in nash_list_row:
                    nash_list_col.append((p, q))
    if len(nash_list_col) == 0:
        nash = (3,2)
        if flag == 1:
            print("##empty ", nash)
    else:
        nash = random.choice(nash_list_col)
        if flag == 1:
            print("##noemp nash list: ", nash_list_col, nash)
    G.nodes[i]['choice'] = strategy['A'][nash[0]]
    G.nodes[j]['choice'] = strategy['U'][nash[1]]
    #update utility
    for k in strategy['A']:
        G.nodes[i]['utility_strategy'][k] += sum(G.nodes[i]['nash_utility'][strategy['A'].index(k)])
    for m in strategy['U']:
        G.nodes[j]['utility_strategy'][m] += sum(G.nodes[j]['nash_utility'][:,strategy['U'].index(m)])
    return

def update_attacker_user(G, i, j):
    '''attacker i uses strategy k, against a user j uses strategy m'''
    # choice is the decision, update is the new opinion at time t
    m = G.nodes[j]['choice']  # user
    if m == 'NU' or m == '':  # user not update, including true informer
        G.nodes[j]['update'] = 0
    else:  # user update
        w1 = G.nodes[i]['omega'][:]  # attacker
        w2 = G.nodes[j]['omega'][:]  # user
        k = G.nodes[i]['choice']  # attacker
        # 1.C attackers find an opinion
        w1_new = attacker_opinion(G, i, k)  # change to use last round opponent's opinion
        method = G.nodes[j]['user_type']
        # note true informer j will not update
        if method in ['H', 'U']:
            G.nodes[j]['update'] = opinion_consensus(w2, w1_new, method)
        elif method == 'D':
            G.nodes[j]['update'] = opinion_herding(G, j, w2)
        elif method == 'AS':
            G.nodes[j]['update'] = opinion_assertion(w2, w1_new)
        elif method == 'E':
            G.nodes[j]['update'] = opinion_encounter(w2, w1_new)
    G.nodes[i]['update'] = G.nodes[j]['omega'][:]
    return

# In[36]:

def strategy_user_nash(G, i, j, defender_strategy, flag, phos, pua):
    """User i meets attacker j or user j"""
    # save nash first fixed table
    fixed2 = {p:{q:0 for q in ['DG', 'C', 'DN', 'S', 'SU', 'U', 'NU'] } for p in strategy['H']} #i user np.zeros(21)
    fixed1 = {p:{q:0 for q in ['DG', 'C', 'DN', 'S', 'SU', 'U', 'NU'] } for p in strategy['H']} #j attacker/user
    w2 = G.nodes[i]['omega'][:]
    method2 = G.nodes[i]['user_type']
    w1 = G.nodes[j]['omega'][:]
    method = G.nodes[j]['user_type']
    if flag == 1 and method != 'T' and (w1[2] < 0.95 or w2[2] < 0.95) and method2 != 'T':
        flag = 1
    else:
        flag = 0

    # update fixed nash table for attacker j - i side
    for m in ['SU', 'U', 'NU']:
        if m != 'NU' and method != 'T':  # true informer never update
            # calculate the loss for each attacker type
            for k in ['DG', 'C', 'DN', 'S']:
                w1_new = attacker_opinion(G, j, k)
                if method in ['H', 'U']:
                    w2_new = opinion_consensus(w2, w1_new, method)
                elif method == 'D':
                    w2_new = opinion_herding_pair(w2, w1_new)
                elif method == 'AS':
                    w2_new = opinion_assertion(w2, w1_new)
                elif method == 'E':
                    w2_new = opinion_encounter(w2, w1_new)
                else:
                    w2_new = w2[:]
                s = homophily_discount(w2_new, wf)
                fixed2[m][k] = s #* pka[k]

    # update fixed nash table for attacker j - j side
    method = G.nodes[j]['user_type']
    for k in strategy['A']:
        w1_new = attacker_opinion(G, j, k)
        gl = all_similarity_attacks(G, w1_new)  # all users are attacked by attacker
        for m in ['SU', 'U', 'NU']: #
            ds = 0 #if m == 'NU'
            if method in ['H', 'U']:
                w2_new = opinion_consensus(w2, w1_new, method)
            elif method == 'D':
                w2_new = opinion_herding_pair(w2, w1_new)
            elif method == 'AS':
                w2_new = opinion_assertion(w2, w1_new)
            elif method == 'E':
                w2_new = opinion_encounter(w2, w1_new)
            else:
                w2_new = w2[:]
            if m != 'NU':
                ds = homophily_discount(w2_new, wf) - homophily_discount(w2, wf)
            fixed1[m][k] = ds-gl #* j_strategy[m]

    # update fixed nash table for user j
    method = G.nodes[j]['user_type']
    for m in strategy['U']:
        # updated i's opinion
        if m == 'NU' or method2 == 'T':  # i no update
            w2_new = w2[:]
        else:  # i update
            if method2 in ['H', 'U']:
                w2_new = opinion_consensus(w2, w1, method2)
            elif method2 == 'D':
                w2_new = opinion_herding_pair(w2, w1)
            elif method == 'AS':
                w2_new = opinion_assertion(w2, w1)
            elif method == 'E':
                w2_new = opinion_encounter(w2, w1)
         # update j's opinion
        if method == 'A' or method == 'T':
            method = method2
        for m1 in strategy['U']:
            if m1 == 'NU':  # j no update
                w1_new = w1[:]
            elif method in ['H', 'U']:
                w1_new = opinion_consensus(w1, w2, method)
            elif method == 'D':
                w1_new = opinion_herding_pair(w1, w2)
            elif method == 'AS':
                w1_new = opinion_assertion(w1, w2)
            elif method == 'E':
                w1_new = opinion_encounter(w1, w2)
            uu = 1
            if method2 == 'U':
                uu = uncertainty_discount(w2_new, w1_new)
            elif method2 in ['U', 'H', 'AS', 'E', 'D']:
                uu = homophily_discount(w2_new, w1_new)
            fixed2[m][m1] = uu
            fixed1[m][m1] = uu

    # make Bayesian nash matrix
    for iu in ['SU', 'U', 'NU']:
        for ja in ['DG', 'C', 'DN', 'S']:
            for ju in ['SU', 'U', 'NU']:
                G.nodes[j]['nash_utility'][strategy['H'].index(iu)][3*strategy['A'].index(ja)+strategy['H'].index(ju)] \
                    = pua * fixed1[iu][ja] + (1 - pua) * fixed1[iu][ju]
                G.nodes[i]['nash_utility'][strategy['H'].index(iu)][3*strategy['A'].index(ja)+strategy['H'].index(ju)] \
                    = pua * fixed2[iu][ja] + (1 - pua) * fixed2[iu][ju]

    # make nash choices based on two matrix nash_utility, find a consensus between row and column players
    nash_list_row = []
    nash_list_col = []
    for p in range(3):
        row_max = max(G.nodes[j]['nash_utility'][p])
        for q in range(12):
            if G.nodes[j]['nash_utility'][p][q] == row_max:
                nash_list_row.append((p, q))
    for q in range(12):
        col_max = max(G.nodes[i]['nash_utility'][:, q])
        for p in range(3):
            if G.nodes[i]['nash_utility'][p][q] == col_max:
                if (p, q) in nash_list_row:
                    nash_list_col.append((p, q))

    # make a final NE choice
    if len(nash_list_col) == 0: #no NE choice, use default
        G.nodes[i]['choice'], G.nodes[j]['choice'] = 'NU', 'NU'
        if G.nodes[j]['user_type'] == 'A':
            G.nodes[j]['choice'] = 'S'
    else:
        nash = random.choice(nash_list_col)
        G.nodes[i]['choice'] = strategy['U'][nash[0]]
        G.nodes[j]['choice'] = strategy['U'][nash[1] % 3]
        if G.nodes[j]['user_type'] == 'A':
            G.nodes[j]['choice'] = strategy['A'][nash[1] // 3]
        if flag == 1:
            print("###noemp nash list: ", nash_list_col, nash, G.nodes[i]['choice'], G.nodes[j]['choice'])

        #update utility
        for m in strategy['H']:
            G.nodes[i]['utility_strategy'][m] += G.nodes[i]['nash_utility'][strategy['H'].index(m)][nash[1]]
        if G.nodes[j]['user_type'] == 'A':
            for k in strategy['A']:
                G.nodes[j]['utility_strategy'][k] += G.nodes[i]['nash_utility'][nash[0]][
                    3*strategy['A'].index(k)+nash[1] % 3]
        else:
            for m in strategy['U']:
                G.nodes[j]['utility_strategy'][m] += G.nodes[i]['nash_utility'][nash[0]][
                    3 * (nash[1] // 3) + strategy['U'].index(m)]
    return

def update_user_user(G, i, j):
    '''User i uses strategy m1, user j uses strategy m2'''
    m1 = G.nodes[i]['choice']
    m2 = G.nodes[j]['choice']
    w1 = G.nodes[i]['omega'][:]
    w2 = G.nodes[j]['omega'][:]
    method1 = G.nodes[i]['user_type']
    method2 = G.nodes[j]['user_type']

    if m1 == 'NU' or m1 == '':
        G.nodes[i]['update'] = 0
    else:  # update i
        if method1 in ['H', 'U']:
            G.nodes[i]['update'] = opinion_consensus(w1, w2, method1)
        elif method1 == 'D':
            G.nodes[i]['update'] = opinion_herding(G, i, w1)
        elif method1 == 'AS':
            G.nodes[i]['update'] = opinion_assertion(w1, w2)
        elif method1 == 'E':
            G.nodes[i]['update'] = opinion_encounter(w1, w2)
    if m2 == 'NU' or m2 == '':
        G.nodes[j]['update'] = 0
    else:  # update j
        if method2 in ['H', 'U']:
            G.nodes[j]['update'] = opinion_consensus(w2, w1, method2)
        elif method2 == 'D':
            G.nodes[j]['update'] = opinion_herding(G, j, w2)
        elif method2 == 'AS':
            G.nodes[j]['update'] = opinion_assertion(w2, w1)
        elif method2 == 'E':
            G.nodes[j]['update'] = opinion_encounter(w2, w1)
    return

# In[38]:

def strategy_defender_nash(G, i, u, defender_cost, defender_observe, defender_utility, puh, pua, flag):
    '''Defender uses strategy l vs attacker i with all users u'''
    #save nash fixed table
    fixed1 = {p:{q:0 for q in ['DG', 'C', 'DN', 'S', 'SU', 'U', 'NU'] } for p in strategy['DF']} #defender
    fixed2 = {p:{q:0 for q in ['DG', 'C', 'DN', 'S', 'SU', 'U', 'NU'] } for p in strategy['DF']} #i as user or attacker
    defender_nash_utility = np.zeros((2, 12))
    oppo_nash_utility = np.zeros((2,12))

    # update fixed nash table for defender
    l = 'M' # defender has no effects
    ds = 0
    ulk = ds - defender_cost[l]  # M case are filled with 0 by default
    l = 'T'  # terminate the attackers
    w1 = G.nodes[i]['omega']
    obs_total = sum(defender_observe.values())
    obs_prob = {k: v / obs_total for (k, v) in defender_observe.items()}
    #defender side choose 'T'
    ep = 0
    for k in ['DG', 'C', 'DN', 'S']:  # the improvement is for all users
        # change all user's opinions, ds = original - changed
        sim = 0
        count = 0
        w1_new = attacker_opinion(G, i, k)
        for u in G.nodes():
            method = G.nodes[u]['user_type']  # estimate omega by U/H probability
            if method in ['U', 'H', 'AS', 'E', 'D']:
                w2 = G.nodes[u]['omega']
                if method in ['H', 'U']:
                    w2_new = opinion_consensus(w2, w1_new, method)
                elif method == 'D':
                    w2_new = opinion_herding_pair(w2, w1_new)
                elif method == 'AS':
                    w2_new = opinion_assertion(w2, w1_new)
                elif method == 'E':
                    w2_new = opinion_encounter(w2, w1_new)
                sim += homophily_discount(w2, wt) - homophily_discount(w2_new, wt)
                count += 1
        if count == 0:
            ds = 0
        else:
            ds = sim / count
        ulk = ds - defender_cost[l]  # level of terminating cost
        fixed1['T'][k] = ulk
        ep = ep + ulk * obs_prob[k] #s1[k]
    fixed1['T']['SU'], fixed1['T']['U'] = ep, ep

    # Bayesian nash matrix -- defender side
    for ja in ['DG', 'C', 'DN', 'S']:
        for ju in ['SU', 'U', 'NU']:
            defender_nash_utility[0][3*strategy['A'].index(ja)+strategy['H'].index(ju)] \
                = pua * fixed1['T'][ja] + (1 - pua) * fixed1['T'][ju]

    # Bayesian nash matrix from the 'nash_utility' -- attacker/user side, 3*12 matrix
    oppo = G.nodes[i]['nash_utility']
    if oppo.shape[1] == 12:
        oppo_nash_utility[0] = oppo[2]
        oppo_nash_utility[1] = oppo[0]
    else: # Bayesian nash matrix from the 'nash_utility' -- attacker/user side, 3*4 matrix
        # update fixed table, 'T' = 'NU', 'M' = 'U'
        for k in strategy['A']:
            fixed2['T'][k] = oppo[strategy['A'].index(k)][2] #'NU'
            fixed2['M'][k] = oppo[strategy['A'].index(k)][1] #'U'
        for m in strategy['H']:
            for k in strategy['A']:
                fixed2['M'][m] += oppo[strategy['A'].index(k)][strategy['H'].index(m)] * obs_prob[k]
                fixed2['T'][m] += oppo[strategy['A'].index(k)][2] * obs_prob[k]
        # make Bayesian nash matrix
        for ja in ['DG', 'C', 'DN', 'S']:
            for ju in ['SU', 'U', 'NU']:
                oppo_nash_utility[0][3*strategy['A'].index(ja)+strategy['H'].index(ju)] \
                    = pua * fixed2['T'][ja] + (1 - pua) * fixed1['T'][ju]
                oppo_nash_utility[1][3*strategy['A'].index(ja)+strategy['H'].index(ju)] \
                    = pua * fixed2['M'][ja] + (1 - pua) * fixed1['M'][ju]

    # make nash choices based on two matrix nash_utility, find a consensus between row and column players
    nash_list_row = []
    nash_list_col = []
    for p in range(2):
        row_max = max(oppo_nash_utility[p])
        for q in range(12):
            if oppo_nash_utility[p][q] == row_max:
                nash_list_row.append((p, q))
    for q in range(12):
        col_max = max(defender_nash_utility[:, q])
        for p in range(2):
            if defender_nash_utility[p][q] == col_max:
                if (p, q) in nash_list_row:
                    nash_list_col.append((p, q))

    # make a final NE choice
    if len(nash_list_col) == 0: #no NE choice, use default
        nash = (1,11)
    else:
        nash = random.choice(nash_list_col)
        if flag == 1:
            print("####noemp nash list: ", nash_list_col, nash)
    oppo_choice = strategy['A'][nash[1] // 3]
    if G.nodes[i]['user_type'] in ['U', 'H', 'AS', 'E', 'D', 'T']:
        oppo_choice = strategy['U'][nash[1] % 3]

    # save defender's utility value
    for l in strategy['DF']:
        defender_utility[l] += defender_nash_utility[strategy['DF'].index(l)][nash[1]]
    return strategy['DF'][nash[0]], defender_utility


# In[42]:

def game_consensus(G, dname, T, N, pa, phos, r, opinions_steps, between_steps, redundancy_steps, defender_cost,
                   defender_observe, defender_evidence, defender_strategy, defender_utility, para=True):
    '''Feeding and posting decisions and opinions update'''
    start_time = time.time()
    # print("seed: ", r)
    random.seed(start_time)  # r
    component_initial = [0]
    st_attacker, st_udm, st_hdm = [0, 0, 0, 0], [0, 0, 0], [0, 0, 0] #udm means all other types
    ut_attacker, ut_udm, ut_hdm = [0, 0, 0, 0], [0, 0, 0], [0, 0, 0] #udm means all other types
    strategy_steps = {x: {} for x in range(int(T / 5))}
    # adding a huge matrix payoff matrix or nash matrix
    for interaction in range(T):
        user_list = list(G.nodes())  # save the untouched nodes in each round
        shuffle = list(G.nodes())
        random.shuffle(shuffle)
        flag = 0
        if interaction == 30:
            flag = 1

        # attackers and users feeding game strategy choice decisions
        for n in shuffle:
            if n in user_list and len(user_list) == 1:
                user_list.remove(n)
                G.nodes[n]['target'] = -1
                G.nodes[n]['choice'] = ''
            elif n in user_list and len(user_list) > 1:  # find a target for n
                user_list.remove(n)
                neighbors = list(G.neighbors(n))
                if len(neighbors) == 0:
                    G.nodes[n]['target'] = -1
                    G.nodes[n]['choice'] = ''
                    continue
                target_list = [tar for tar in neighbors if tar in user_list]
                if len(target_list) == 0:
                    G.nodes[n]['target'] = -1
                    G.nodes[n]['choice'] = ''
                    continue
                weight_sharing = [G.nodes[k]['feeding'] + G.nodes[k]['posting'] for k in target_list]
                target = random.choices(target_list, weights=weight_sharing, k=1)[0]
                G.nodes[n]['target'] = target
                G.nodes[target]['target'] = n
                user_list.remove(target)

                if G.nodes[n]['user_type'] == 'A':  # target can be T, H or U
                    if interaction == 0:  # the first round, users choose random strategy
                        G.nodes[n]['choice'] = 'S'
                        if G.nodes[target]['user_type'] != 'T':
                            G.nodes[target]['choice'] = random.choice(strategy['U'])
                    else:  # game theory choice/payoffs to select strategy
                        # if True: calculate the huge table first, then choose a better choice for the two players
                        G.nodes[n]['nash_utility'] = np.zeros((4, 3)) #attacker vs user
                        G.nodes[target]['nash_utility'] = np.zeros((4, 3))
                        strategy_attacker_nash(G, n, target, defender_strategy, flag, phos)
                        obv = user_report(G, target, n, interaction)  # terminate friendship after each round and report
                        if obv == True:  # defender observe reported attacker's strategy
                            defender_observe[G.nodes[n]['choice']] += 1
                    update_attacker_user(G, n, target)
                elif G.nodes[target]['user_type'] == 'A':
                    if interaction == 0:  # the first round, users choose random strategy
                        G.nodes[target]['choice'] = 'S'
                        if G.nodes[n]['user_type'] != 'T':
                            G.nodes[n]['choice'] = random.choice(strategy['U'])
                    else:  # game theory choice/payoffs to select strategy
                        G.nodes[n]['nash_utility'] = np.zeros((3, 12)) #attacker vs user
                        G.nodes[target]['nash_utility'] = np.zeros((3, 12))
                        strategy_user_nash(G, n, target, defender_strategy, flag, phos, pa)
                        obv = user_report(G, n, target, interaction)
                        if obv == True:
                            defender_observe[G.nodes[target]['choice']] += 1
                    update_attacker_user(G, target, n)
                else:  # two users
                    if interaction == 0:  # the first round, users choose random strategy
                        if G.nodes[n]['user_type'] != 'T':
                            G.nodes[n]['choice'] = random.choice(strategy['U'])
                        if G.nodes[target]['user_type'] != 'T':
                            G.nodes[target]['choice'] = random.choice(strategy['U'])
                    else:  # game theory choice/payoffs to select strategy
                        G.nodes[n]['nash_utility'] = np.zeros((3, 12))  # user vs user
                        G.nodes[target]['nash_utility'] = np.zeros((3, 12))
                        strategy_user_nash(G, n, target, defender_strategy, flag, phos, pa)
                    user_report(G, target, n, interaction)
                    user_report(G, n, target, interaction)
                    update_user_user(G, n, target)


        # record the choices count/ distribution
        subrem = (interaction + 1) % 5
        subtime = (interaction + 1) / 5 - 1
        if subrem == 0:
            strategy_steps[subtime] = np.zeros(12) #att: DG-C-DN-S; udm/hdm: SU-U-NU; def: T-M
            print('initial 12 0s:', strategy_steps[subtime])
            strategy_att = [y for (x, y) in list(G.nodes(data='choice')) if G.nodes[x]['user_type'] == 'A']
            print('strategy_att:', strategy_att)
            if len(strategy_att) == 0:
                strategy_steps[subtime][0], strategy_steps[subtime][1], \
                strategy_steps[subtime][2], strategy_steps[subtime][3] = 0, 0, 0, 0
            else:
                strategy_steps[subtime][0] = strategy_att.count('DG')
                strategy_steps[subtime][1] = strategy_att.count('C')
                strategy_steps[subtime][2] = strategy_att.count('DN')
                strategy_steps[subtime][3] = strategy_att.count('S')
            if phos != 1:
                strategy_udm = [y for (x, y) in list(G.nodes(data='choice'))
                                if G.nodes[x]['user_type'] in ['U','AS','D','E']]
                if len(strategy_udm) == 0:
                    strategy_steps[subtime][4], strategy_steps[subtime][5], \
                    strategy_steps[subtime][6] = 0, 0, 0
                else:
                    strategy_steps[subtime][4] = strategy_udm.count('SU')
                    strategy_steps[subtime][5] = strategy_udm.count('U')
                    strategy_steps[subtime][6] = strategy_udm.count('NU')
            if phos != 0:
                strategy_hdm = [y for (x, y) in list(G.nodes(data='choice')) if G.nodes[x]['user_type'] == 'H']
                if len(strategy_hdm) == 0:
                    strategy_steps[subtime][7], strategy_steps[subtime][8], \
                    strategy_steps[subtime][9] = 0, 0, 0
                else:
                    strategy_steps[subtime][7] = strategy_hdm.count('SU')
                    strategy_steps[subtime][8] = strategy_hdm.count('U')
                    strategy_steps[subtime][9] = strategy_hdm.count('NU')
            print('after 10 vals:', strategy_steps[subtime])

        if interaction >= 20 and interaction  <= 40:
            flag = 1
        for n in shuffle:  #adding more sections to calculate utility of users/ attackers
            defender_nash_utility = np.empty(0)
            target = G.nodes[n]['target']
            if n in G.nodes() and G.nodes[n]['report'] >= 3 and target > -1:  # defender's threshold 5%?
                defender_choice = 'M'
                defender_choice, defender_utility = strategy_defender_nash(G, n, target,
                        defender_cost, defender_observe, defender_utility, phos, pa, flag)
                defender_evidence[defender_choice] += 1  # update the expeience of defender
                if defender_evidence[defender_choice] <= 10:
                    print("#defender evidence test: ", defender_evidence, interaction)
                if defender_choice == 'T':  ###defender remove attackers
                    G.nodes[n]['remove_node'] = True
                    G.remove_node(n)
                #for defender's current choices
                if subrem == 0:
                    print('defender choices', strategy_steps[subtime])
                    if defender_choice == 'T':
                        strategy_steps[subtime][10] = strategy_steps[subtime][10] + 1
                    else:
                        strategy_steps[subtime][11] = strategy_steps[subtime][11] + 1

        # update all (opinion update, Pf/Pp, choice) for current step after decisions
        for n in G.nodes():
            target = G.nodes[n]['target']
            if target not in G.nodes() or target == -1:  # no update for current interaction
                continue
            elif G.nodes[n]['user_type'] == 'A':
                G.nodes[n]['omega'] = G.nodes[n]['update'][:]
                G.nodes[n]['evidence_strategy'][G.nodes[n]['choice']] += 1
            else:  # users
                if G.nodes[n]['choice'] in ['SU', 'U']:
                    if G.nodes[n]['user_type'] != 'T':
                        G.nodes[n]['omega'] = G.nodes[n]['update'][:]
                    G.nodes[n]['evidence_share'][2] += 1  # update Pf
                if G.nodes[n]['choice'] == 'SU':
                    G.nodes[n]['evidence_share'][3] += 1  # update Pp
                if G.nodes[n]['user_type'] != 'T':  # update evidence_strategy
                    G.nodes[n]['evidence_strategy'][G.nodes[n]['choice']] += 1

            # update probability of strategies (AS, H, U, D) and sharing (H, U)
            if G.nodes[n]['user_type'] != 'T':
                sum_strategy = sum(G.nodes[n]['evidence_strategy'].values())
                for m in strategy[G.nodes[n]['user_type']]:
                    G.nodes[n]['strategy'][m] = G.nodes[n]['evidence_strategy'][m] / sum_strategy
            if G.nodes[n]['user_type'] in ['U', 'H', 'AS','D','E']:
                G.nodes[n]['feeding'] = (G.nodes[n]['evidence_share'][0] + G.nodes[n]['evidence_share'][2]) \
                                        / (N + G.nodes[n]['evidence_share'][2])
                G.nodes[n]['posting'] = (G.nodes[n]['evidence_share'][1] + G.nodes[n]['evidence_share'][3]) \
                                        / (N + G.nodes[n]['evidence_share'][3])

        # update defender_strategy based on defender_evidence
        sum_def = sum(defender_evidence.values())
        for e in strategy['DF']:
            defender_strategy[e] = max(defender_evidence[e] / sum_def, 0.1)

        #count the strategy choices/utilities of each user
        if (interaction + 1) % 50 == 0:
            print(interaction, "#size and order: ", G.size(), G.order())
            # print(G)
            stat_at_1 = [y['S'] for (x, y) in list(G.nodes(data='evidence_strategy')) if 'S' in y]
            stat_at_2 = [y['DN'] for (x, y) in list(G.nodes(data='evidence_strategy')) if 'DN' in y]
            stat_at_3 = [y['C'] for (x, y) in list(G.nodes(data='evidence_strategy')) if 'C' in y]
            stat_at_4 = [y['DG'] for (x, y) in list(G.nodes(data='evidence_strategy')) if 'DG' in y]
            stat_u_1 = [y['U'] for (x, y) in list(G.nodes(data='evidence_strategy')) if
                        ('U' in y and G.nodes[x]['user_type'] in ['U','AS','D','E'])]
            stat_u_2 = [y['SU'] for (x, y) in list(G.nodes(data='evidence_strategy')) if
                        ('SU' in y and G.nodes[x]['user_type'] in ['U','AS','D','E'])]
            stat_u_3 = [y['NU'] for (x, y) in list(G.nodes(data='evidence_strategy')) if
                        ('NU' in y and G.nodes[x]['user_type'] in ['U','AS','D','E'])]
            stat_h_1 = [y['U'] for (x, y) in list(G.nodes(data='evidence_strategy')) if
                        ('U' in y and G.nodes[x]['user_type'] == 'H')]
            stat_h_2 = [y['SU'] for (x, y) in list(G.nodes(data='evidence_strategy')) if
                        ('SU' in y and G.nodes[x]['user_type'] == 'H')]
            stat_h_3 = [y['NU'] for (x, y) in list(G.nodes(data='evidence_strategy')) if
                        ('NU' in y and G.nodes[x]['user_type'] == 'H')]
            print("#i ", interaction, "stats S", sum(stat_at_1) - len(stat_at_1), "DN", sum(stat_at_2) - len(stat_at_2),
                  "C", sum(stat_at_3) - len(stat_at_3), "DG", sum(stat_at_4) - len(stat_at_4),
                  "U", sum(stat_u_1) - len(stat_u_1), "SU", sum(stat_u_2) - len(stat_u_2))

            # utility value for each user
            ut_at_1 = [y['S'] / (sum(G.nodes[x]['evidence_strategy'].values()) - 4) for (x, y) in
                       list(G.nodes(data='utility_strategy')) if 'S' in y]
            ut_at_2 = [y['DN'] / (sum(G.nodes[x]['evidence_strategy'].values()) - 4) for (x, y) in
                       list(G.nodes(data='utility_strategy')) if 'DN' in y]
            ut_at_3 = [y['C'] / (sum(G.nodes[x]['evidence_strategy'].values()) - 4) for (x, y) in
                       list(G.nodes(data='utility_strategy')) if 'C' in y]
            ut_at_4 = [y['DG'] / (sum(G.nodes[x]['evidence_strategy'].values()) - 4) for (x, y) in
                       list(G.nodes(data='utility_strategy')) if 'DG' in y]
            if len([x for x in list(G.nodes()) if G.nodes[x]['user_type'] in ['U','AS','D','E']]) == 0:
                ut_u_1, ut_u_2, ut_u_3 = [0], [0], [0]
            elif sum([sum(G.nodes[x]['evidence_strategy'].values()) - 3
                for (x, y) in list(G.nodes(data='utility_strategy'))
                     if ('U' in y and G.nodes[x]['user_type'] in ['U','AS','D','E'])]) == 0:
                ut_u_1, ut_u_2, ut_u_3 = [0], [0], [0]
            else:
                ut_u_1 = [y['U'] / (sum(G.nodes[x]['evidence_strategy'].values()) - 2)
                          for (x, y) in list(G.nodes(data='utility_strategy')) if
                          ('U' in y and G.nodes[x]['user_type'] in ['U','AS','D','E']
                           and sum(G.nodes[x]['evidence_strategy'].values()) > 3)]
                ut_u_2 = [y['SU'] / (sum(G.nodes[x]['evidence_strategy'].values()) - 2)
                          for (x, y) in list(G.nodes(data='utility_strategy')) if
                          ('SU' in y and G.nodes[x]['user_type'] in ['U','AS','D','E']
                           and sum(G.nodes[x]['evidence_strategy'].values()) > 3)]
                ut_u_3 = [y['NU'] / (sum(G.nodes[x]['evidence_strategy'].values()) - 2)
                          for (x, y) in list(G.nodes(data='utility_strategy')) if
                          ('NU' in y and G.nodes[x]['user_type'] in ['U','AS','D','E']
                           and sum(G.nodes[x]['evidence_strategy'].values()) > 3)]
            if len([x for x in list(G.nodes()) if G.nodes[x]['user_type'] == 'H']) == 0:
                ut_h_1, ut_h_2, ut_h_3 = [0], [0], [0]
            elif sum([sum(G.nodes[x]['evidence_strategy'].values()) - 3
                for (x, y) in list(G.nodes(data='utility_strategy'))
                      if ('U' in y and G.nodes[x]['user_type'] == 'H')]) == 0:
                ut_h_1, ut_h_2, ut_h_3 = [0], [0], [0]
            else:
                ut_h_1, ut_h_2, ut_h_3 = [0], [0], [0]
                ut_h_1 = [y['U'] / (sum(G.nodes[x]['evidence_strategy'].values()) - 2)
                          for (x, y) in list(G.nodes(data='utility_strategy')) if
                          ('U' in y and G.nodes[x]['user_type'] == 'H'
                           and sum(G.nodes[x]['evidence_strategy'].values()) > 3)]
                ut_h_2 = [y['SU'] / (sum(G.nodes[x]['evidence_strategy'].values()) - 2)
                          for (x, y) in list(G.nodes(data='utility_strategy')) if
                          ('SU' in y and G.nodes[x]['user_type'] == 'H'
                           and sum(G.nodes[x]['evidence_strategy'].values()) > 3)]
                ut_h_3 = [y['NU'] / (sum(G.nodes[x]['evidence_strategy'].values()) - 2)
                          for (x, y) in list(G.nodes(data='utility_strategy')) if
                          ('NU' in y and G.nodes[x]['user_type'] == 'H'
                           and sum(G.nodes[x]['evidence_strategy'].values()) > 3)]
            print("#utility: ", sum(ut_at_1) / max(1, len(ut_at_1)), sum(ut_at_2) / max(1, len(ut_at_1)),
                  sum(ut_at_3) / max(1, len(ut_at_1)), sum(ut_at_4) / max(1, len(ut_at_1)),
                  sum(ut_u_1) / max(1, len(ut_u_1)), sum(ut_u_2) / max(1, len(ut_u_1)),
                  sum(ut_u_3) / max(1, len(ut_u_1)), sum(ut_h_1) / max(1, len(ut_h_1)),
                  sum(ut_h_2) / max(1, len(ut_h_1)), sum(ut_h_2) / max(1, len(ut_h_1)))

            #count the average count of strategies of all users
            if interaction == T - 1:
                if len(stat_at_1) != 0:
                    st_attacker[0] = sum(stat_at_1) / len(stat_at_1)
                    st_attacker[1] = sum(stat_at_2) / len(stat_at_2)
                    st_attacker[2] = sum(stat_at_3) / len(stat_at_3)
                    st_attacker[3] = sum(stat_at_4) / len(stat_at_4)
                if phos != 1:
                    if len(stat_u_1) != 0:
                        st_udm[0] = sum(stat_u_1) / len(stat_u_1)
                        st_udm[1] = sum(stat_u_2) / len(stat_u_2)
                        st_udm[2] = sum(stat_u_3) / len(stat_u_3)
                if phos != 0:
                    if len(stat_h_1) != 0:
                        st_hdm[0] = sum(stat_h_1) / len(stat_h_1)
                        st_hdm[1] = sum(stat_h_2) / len(stat_h_2)
                        st_hdm[2] = sum(stat_h_3) / len(stat_h_3)

                if len(ut_at_1) != 0:
                    ut_attacker[0] = sum(ut_at_1) / len(ut_at_1)
                    ut_attacker[1] = sum(ut_at_2) / len(ut_at_2)
                    ut_attacker[2] = sum(ut_at_3) / len(ut_at_3)
                    ut_attacker[3] = sum(ut_at_4) / len(ut_at_4)
                if phos != 1:
                    if len(ut_u_1) != 0:
                        ut_udm[0] = sum(ut_u_1) / len(ut_u_1)
                        ut_udm[1] = sum(ut_u_2) / len(ut_u_2)
                        ut_udm[2] = sum(ut_u_3) / len(ut_u_3)
                if phos != 0:
                    if len(ut_h_1) != 0:
                        ut_hdm[0] = sum(ut_h_1) / len(ut_h_1)
                        ut_hdm[1] = sum(ut_h_2) / len(ut_h_2)
                        ut_hdm[2] = sum(ut_h_3) / len(ut_h_3)

        # adding new friends
        user_list = list(G.nodes())  # save the untouched nodes in each round
        shuffle = list(G.nodes())
        random.shuffle(shuffle)
        node_degree = [x for (x, y) in list(G.degree())]
        degrees = [y for (x, y) in list(G.degree())]  # probability of each degree
        occur_degree = []
        for d in degrees:
            occur_degree.append(sum([d == x for x in degrees]) / len(degrees))
        d_time_p = [degrees[x] * occur_degree[x] for x in range(len(degrees))]
        for n in shuffle:
            if n in user_list and len(user_list) == 1:
                user_list.remove(n)
            elif n in user_list and len(user_list) > 1:  # find a target for n
                user_list.remove(n)
                neighbors = list(G.neighbors(n))
                if G.nodes[n]['user_type'] == 'A':
                    target_list = [tar for tar in user_list if (tar not in neighbors and G.nodes[tar] != 'A')]
                else:
                    target_list = [tar for tar in user_list if tar not in neighbors]
                if len(target_list) == 0:
                    continue
                # adding friend rule
                weight_sharing = [(degrees[node_degree.index(k)] + 1) * occur_degree[node_degree.index(k)]
                                  / (sum(d_time_p) + 1) for k in target_list]
                target = random.choices(target_list, weights=weight_sharing, k=1)[0]
                user_list.remove(target)
                if G.nodes[target]['user_type'] == 'U':
                    if G.nodes[n]['omega'][2] < G.nodes[target]['phi']:
                        new_friend(G, n, target)
                elif G.nodes[target]['user_type'] in ['H','AS','D','E'] and \
                        opinion_difference(G.nodes[n]['omega'], G.nodes[target]['omega']) < \
                        G.nodes[target]['phi']:

        capital_between(G)
        capital_redundancy(G)
        # save the result of each step for metrics
        for n in G.nodes():
            opinions_steps[interaction][n] = G.nodes[n]['omega'][:]
            between_steps[interaction][n] = G.nodes[n]['capital']
            redundancy_steps[interaction][n] = G.nodes[n]['redundancy']

        if interaction == 0:
            component_initial = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
        if interaction == 30:
            print("end of run 30")
    print((time.time() - start_time) / 60)

    # write to file for parallel computing
    if para: #1ks10kn
        folder = 'res1ks'
        fname = '%s/tmpnash_%d/%d' % (folder, int(phos * 100), r)

        with open(fname+'.txt', 'w') as f:
            f.write("%s\n" % list(component_initial))
            f.write("%s\n" % list([len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]))
            for item in opinions_steps:
                f.write("INTER %s\n" % item)
                for node in opinions_steps[item]:
                    f.write("%d %s\n" % (node, list(opinions_steps[item][node])))
            for item in between_steps:
                f.write("INTER %s\n" % item)
                for node in between_steps[item]:
                    f.write("%d %f\n" % (node, between_steps[item][node]))
            for item in redundancy_steps:
                f.write("INTER %s\n" % item)
                for node in redundancy_steps[item]:
                    f.write("%d %f\n" % (node, redundancy_steps[item][node]))
            f.write("INTER 0\n")
            f.write("%s\n" % st_attacker)
            f.write("%s\n" % list(defender_evidence.values()))
            f.write("%s\n" % st_udm)
            f.write("%s\n" % st_hdm)
            # utility
            print("utility defender", defender_utility)
            f.write("%s\n" % ut_attacker)
            if sum(defender_evidence.values()) - 2 == 0:
                f.write("%s\n" % list([0, 0]))
            else:
                f.write("%s\n" % list([y / (sum(defender_evidence.values()) - 2) for y in defender_utility.values()]))
            f.write("%s\n" % ut_udm)
            f.write("%s\n" % ut_hdm)
            # strategy by steps
            for item in strategy_steps:
                f.write("INTER %d %s\n" % (item, list(strategy_steps[item])))
    return


# In[43]:

def user_report(G, u, r, interaction):
    '''User u report another user r as attacker'''
    p = False
    w1 = G.nodes[u]['omega'][:]
    w2 = G.nodes[r]['omega'][:]
    phi = G.nodes[u]['phi']
    rho = G.nodes[u]['rho']

    if (u, r) in G.edges():
        diff = opinion_difference(w1, w2)
        if G.nodes[r]['user_type'] != 'T' and diff > rho and w2[2] < 0.5:  # and w2[1] > rho: #report
            if G.edges[u, r]["weight"] < 2:  # one user can only report once
                G.edges[u, r]["weight"] += 1
                G.nodes[r]['report'] += 1
                p = True

        # new removing edge rule:
        if G.nodes[r]['user_type'] == 'U':
            if w2[2] > phi and w2[2] < 0.5:
                G.nodes[u]['remove_edge'] = r
                if (u, r) in G.edges():
                    G.remove_edge(u, r)
        elif diff > phi and w2[2] < 0.5:
            G.nodes[u]['remove_edge'] = r
            if (u, r) in G.edges():
                G.remove_edge(u, r)
    return p


def new_friend(G, u, v):
    '''Adding an new edge'''
    G.add_edge(u, v, weight=1, f=1, b=1) #initial feeding and feedback count
    for i in [u, v]:  # update  full
        if G.degree(i) >= G.nodes[i]['friends']:
            G.nodes[i]['full'] = True


def friend_pending(G):
    '''Return list of nodes need friends'''
    pending = []
    for (i, x) in G.nodes(data='full'):
        if x == False:
            pending.append(i)
    return pending


def friendship_TP(G, dname, topics, run=10):
    '''Simulate adding friends of users-- Topics similarity'''
    start_time = time.time()
    rnd = 375
    random.seed(rnd)
    print("run seed:", (date.today() - date(2020, 7, 1)).days)

    # adjacency matrix to save similarity
    size = max(G.nodes()) + 1
    product_matrix = np.zeros((size, size))
    for x in G.nodes():
        dot_x = topics.loc[x]
        for y in G.nodes():
            if y > x:
                dot_y = topics.loc[y]
                mut = np.dot(dot_x, dot_y) / np.linalg.norm(dot_x) / np.linalg.norm(dot_y)
                product_matrix[x][y] = mut

    for i in range(100):
        pending = friend_pending(G)
        random.shuffle(pending)
        for j in pending:
            idx_j = pending.index(j)
            if G.nodes[j]['full'] == True:
                continue

            a = random.random()
            if G.nodes[j]['inviting'] >= a:
                # find all possible k
                friend_j = list(G.neighbors(j))
                list_k = pending[:]
                list_k.remove(j)
                list_k = list(set(list_k) - set(friend_j))
                if G.nodes[j]['user_type'] == 'A':  # attacker not choose attacker as friends
                    atk = [x for x in G.nodes() if G.nodes[x]['user_type'] == 'A']
                    list_k = list(set(list_k) - set(atk))
                if len(list_k) == 0:
                    G.nodes[j]['full'] = True
                    continue
                node_list0 = {}
                for k in list_k:
                    if G.nodes[k]['full'] == False:
                        node_list0[k] = product_matrix[min(j, k)][max(j, k)]
                if len(node_list0) == 0:
                    G.nodes[j]['full'] = True
                    continue
                node_list = {}
                for k in node_list0.keys():
                    node_list[k] = node_list0[k]
                if len(node_list) == 0:
                    continue

                # rank node_list and pick top 1
                node_list = {k: v for k, v in sorted(node_list.items(), key=lambda item: item[1])}
                flag = 0
                while flag == 0:
                    if len(node_list) == 0 or G.nodes[j]['full'] == True:
                        break
                    ranked = list(node_list.keys())  # ascending
                    f = ranked[-1]
                    if G.nodes[f]['full'] == False:
                        new_friend(G, j, f)
                        flag = 1
                    del node_list[f]

    print((time.time() - start_time) / 60)
    return G


def game_run(G, dname, N, T, pa, phos, r, defender_cost):
    '''Run parallel jobs for each run'''
    print("subjob", r, "graph size initial:", G.size(), G.order())
    defender_strategy = {'T': 0.5, 'M': 0.5}
    defender_evidence = {'T': 1, 'M': 1}
    defender_utility = {'T': 0, 'M': 0}
    defender_observe = {x: 1 for x in strategy['A']}  # defender observe pkA from reports
    # sharing and posting stats
    opinions_steps = {x: {} for x in range(T)}
    between_steps = {x: {} for x in range(T)}
    redundancy_steps = {x: {} for x in range(T)}
    game_consensus(G, dname, T, N, pa, phos, r, opinions_steps, between_steps, redundancy_steps,
                   defender_cost, defender_observe, defender_evidence, defender_strategy, defender_utility, True)
    print(defender_evidence)
    print("graph size remove:", G.size(), G.order())
    return opinions_steps, between_steps, redundancy_steps