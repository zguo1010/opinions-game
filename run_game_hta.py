#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.community as nx_m
import warnings
import math
import time
import random
import json
from numpy.random import randint, choice, seed
from datetime import date
import sys
from multiprocessing import Process, Value, Array, Pool
from game_functions_hta import *
#very important for cluster to see real-time result
plt.switch_backend('agg')

def main(argv):
    #parameters
    dname = "1ks10kn"
    N = 1000
    T = int(argv[0]) #1000
    pa = float(argv[1]) #0.1
    pt = float(argv[2]) #0.1
    phos = float(argv[3]) #1.0 for H, 0.5 for H and other, 0 for others
    mup, sigmap = float(argv[4]), 0.1
    mur, sigmar = float(argv[5]), 0.05
    run = int(argv[6])

    print("N: %d T: %d pa: %f pt: %f phos: %f run: %d" %(N, T, pa, pt, phos, run))
    #strategy choices for attackers, defenders, and users
    #choices: A-k, D-l, U-m
    strategy = {"A": ['DG', 'C', 'DN', 'S'], "D":['T', 'M'],
                "U":['SU', 'U', 'NU'], "H":['SU', 'U', 'NU'], "T":['SU', 'U', 'NU']}
    #cost of defender's strategy
    defender_cost = {'T':0.1, 'M':0} #0.001

    import pandas as pd
    #read data from files
    likers = pd.read_csv("1ks10kn.csv")
    print(likers.columns) #'num_hashtags' replaces 'category' as support received from friends/followers
    likers = pd.read_csv("1ks10kn.csv",
                         names=['user_id', 'followers', 'friends', 'age', 'lines',
                               'len_name', 'category', #'num_hashtags'
                               'num_urls', 'num_mentions', 'favorite_tweets', 'total_posts',
                               'total_replies', 'freq_posts', 'freq_replies', 'label', 'net_posts', 'freq_np', 'verified'],
                         header=0)
    full_size = likers.shape[0]
    topics = pd.read_csv("TopicsMalNormal.csv", names=range(20), header=None)
    likers.loc[likers['total_replies'] == 0, 'total_replies'] = 1
    print(likers.shape)

    # set max friend, cut the extra
    topics = topics.loc[likers['friends'] <= 80000]
    likers = likers.loc[likers['friends'] <= 80000]
    likers.loc[likers['friends']==0, 'friends'] = 1

    # reduce friend number by scale
    scale = 0.01
    likers['friends'] = np.ceil(likers['friends'] * scale)

    # normalize large values
    likers.loc[likers['favorite_tweets'] > 20, 'favorite_tweets'] = 20  # 2106
    maximum_cap(likers, 'freq_posts', 0.1)#83
    likers.loc[likers['freq_replies']>0.5, 'freq_replies'] = 0.5 #952

    # features P^f P^p and friends
    ld = likers['label'] == 'Legit'
    likers.loc[ld, 'feeding'] = (likers.loc[ld, 'freq_replies'] / max(likers.loc[ld, 'freq_replies'])
                                 + likers.loc[ld, 'favorite_tweets'] / max(likers.loc[ld, 'favorite_tweets'])) / 2
    likers.loc[ld, 'posting'] = likers.loc[ld, 'freq_posts'] / max(likers.loc[ld, 'freq_posts'])
    likers.loc[ld, 'inviting'] = likers.loc[ld, 'friends'] / np.percentile(likers.loc[ld, 'friends'], 85)
    #0.142, 0.186, xx
    print('friend max: ', max(likers['friends']))#reduced max(number of friend)
    f = int(np.percentile(likers['friends'], 90))
    print('friend 90%:', f)

    # start game decision
    start_time_i = time.time()
    opinions_runs = {x:{} for x in range(T)} #save each user's opinions for runs/interactions
    sir_runs = {x:{} for x in range(T)} #count the sir status for each fun
    between_runs = {x:{} for x in range(T)} #save betweenness metric for runs
    redundancy_runs = {x:{} for x in range(T)} #save redundancy metric for runs
    strategy_runs_att = {x: {} for x in range(int(T / 5))}
    strategy_runs_udm = {x: {} for x in range(int(T / 5))}
    strategy_runs_hdm = {x: {} for x in range(int(T / 5))}
    strategy_runs_def = {x: {} for x in range(int(T / 5))}
    opinions_std, between_std, redundancy_std = {x:{} for x in range(T)}, {x:{} for x in range(T)}, {x:{} for x in range(T)}
    components = {x:{} for x in range(run)}
    components_st = {x:{} for x in range(run)} #initial components number
    choice_at, choice_df, choice_u, choice_h = np.empty([0,4]),  np.empty([0,2]), np.empty([0,3]),  np.empty([0,3])
    utility_at, utility_df, utility_u, utility_h = np.empty([0, 4]), np.empty([0, 2]), np.empty([0, 3]), np.empty([0, 3])
    community_bi, community_mo, community_lp = np.empty([0, 6]), np.empty([0, 6]), np.empty([0, 6])
    nodes = {'T':[], 'A':[], 'U':[], 'H':[], 'AS':[], 'D':[], 'E':[]} #save the user types
    nodes_total = []
    degrees = {'b': [], 'a': []}

    G, topics_s, nodes, nodes_total = initialization(likers, topics, N, pa, pt, phos, mup, sigmap, mur, sigmar)
    # make friend connection
    friendship_TP(G, dname, topics_s)  # same friend work
    # if r == 0:
    print(G.degree())
    print([len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)])
    #save betweeness and redundancy to file
    capital_between(G)
    capital_between_friend(G)
    # capital_redundancy(G)
    trust_update(G)
    between_0 = list(G.nodes(data='bridging'))
    redundancy_0 = list(G.nodes(data='trust'))
    degrees['b'] = sum([y for (x,y) in G.degree])/G.order()
    print("Mean degree of the network: ", degrees)

    # Communities- Modularity
    comm_bi_val = {'b': [], 'a':[]}
    comm_lp_val = {'b': [], 'a': []}
    comm_mo = nx_m.greedy_modularity_communities(G)
    print("Modularities", len(comm_mo))
    comm_mo = sorted(comm_mo, key=len, reverse=True)
    print([len(x) for x in comm_mo])
    comm_mo_val = {'b': [], 'a': []}  # modularity, bp, random walk scores, performance
    comm_mo_val['b'].append(len(comm_mo))
    comm_mo_val['b'].append(sum([len(x) for x in comm_mo]) / len(comm_mo))
    comm_mo_val['b'].append(modularity_normalize(nx_m.modularity(G, comm_mo)))
    lines1 = [str(x) + '\n' for x in comm_mo[0]]
    lines2 = [str(x) + '\n' for x in comm_mo[1]]
    comm_mo_val['b'].append(boundary_polarization(G, lines1, lines2))
    comm_mo_val['b'].append(random_walk_polarization(G, lines1, lines2))
    comm_mo_val['b'].append(nx_m.performance(G, comm_mo))
    draw_community(G, comm_mo, 'results/community_mo_' + str(int(phos * 100)) )
    print(comm_mo_val)

    # parallel running
    directory = 'res1ks/tmp%d_%d/' % (int(pa * 100), int(phos * 100))
    files_in_directory = os.listdir(directory)
    filtered_files = sorted([file for file in files_in_directory if file.endswith(".txt")])
    for file in filtered_files:
        path_to_file = os.path.join(directory, file)
        os.remove(path_to_file)
    filtered_files = sorted([file for file in files_in_directory if file.endswith(".png")])
    for file in filtered_files:
        path_to_file = os.path.join(directory, file)
        os.remove(path_to_file)

    threads = [0 for r in range(run)]
    for r in range(run): #four combining methods?
        threads[r] = Process(target=game_run, args=(G, dname, N, T, pa, phos, r, defender_cost))
        threads[r].start()
    for thread in threads:
        try:
            thread.join()
        except Exception as e:
            print("Exception Handled with Details of the Exception:", r, e)
    start_time_r = time.time()
    print("running time for parallel", run, (start_time_r - start_time_i) / 60)

    #collect all results in opinions_runs
    directory = 'res1ks/tmp%d_%d/' % (int(pa * 100), int(phos * 100))
    files_in_directory = os.listdir(directory)
    filtered_files = sorted([file for file in files_in_directory if file.endswith(".txt")])
    for filename in filtered_files:
        file = open(directory + filename, 'r')
        print("open a new file", filename)
        counter = 0
        #read first interaction
        line = file.readline()
        components_st[int(filename.split('.')[0])] = [int(x) for x in line[1:-2].split(', ')]
        #read T interaction
        line = file.readline()
        components[int(filename.split('.')[0])] = [int(x) for x in line[1:-2].split(', ')]
        line = file.readline()
        while counter <= T:
            if line.startswith("INTER "):
                t = int(line.split(' ')[1])
                counter += 1
            else:
                linesplit = line.split(' [')
                node = int(linesplit[0])
                new_op = np.array([float(x) for x in linesplit[1][:-2].split(', ')]).reshape((1,4))
                if node not in opinions_runs[t]:
                    opinions_runs[t][node] = new_op
                else:
                    opinions_runs[t][node] = np.append(opinions_runs[t][node], new_op, axis=0)
            line = file.readline()
        counter = 1
        while counter <= T:
            if line.startswith("INTER "):
                t = int(line[:-1].split(' ')[1])
                counter += 1
            else:
                linesplit = line.split(' ')
                node = int(linesplit[0])
                if node not in between_runs[t]:
                    between_runs[t][node] = [float(linesplit[1])]
                else:
                    between_runs[t][node].append(float(linesplit[1]))
            line = file.readline()
        counter = 1
        while line and counter <= T:
            if line.startswith("INTER "):
                t = int(line[:-1].split(' ')[1])
                counter += 1
            else:
                linesplit = line.split(' ')
                node = int(linesplit[0])
                if node not in redundancy_runs[t]:
                    redundancy_runs[t][node] = [float(linesplit[1])]
                else:
                    redundancy_runs[t][node].append(float(linesplit[1]))
            line = file.readline()
        if sum([float(x) for x in line[1:-2].split(', ')]) > 0:
            choice_at = np.append(choice_at, np.array([float(x) for x in line[1:-2].split(', ')]).reshape((1,4)), axis=0)
        line = file.readline()
        choice_df = np.append(choice_df, np.array([int(x) for x in line[1:-2].split(', ')]).reshape((1,2)), axis=0)
        line = file.readline()
        if phos != 1.0:
            choice_u = np.append(choice_u, np.array([float(x) for x in line[1:-2].split(', ')]).reshape((1,3)), axis=0)
        line = file.readline()
        if phos != 0.0:
            choice_h = np.append(choice_h, np.array([float(x) for x in line[1:-2].split(', ')]).reshape((1,3)), axis=0)
        #utility
        line = file.readline()
        if sum([float(x) for x in line[1:-2].split(', ')]) != 0:
            utility_at = np.append(utility_at, np.array([float(x) for x in line[1:-2].split(', ')]).reshape((1,4)), axis=0)
            print("###", utility_at)
        line = file.readline()
        utility_df = np.append(utility_df, np.array([float(x) for x in line[1:-2].split(', ')]).reshape((1,2)), axis=0)
        line = file.readline()
        if phos != 1.0:
            utility_u = np.append(utility_u, np.array([float(x) for x in line[1:-2].split(', ')]).reshape((1,3)), axis=0)
        line = file.readline()
        if phos != 0.0:
            utility_h = np.append(utility_h, np.array([float(x) for x in line[1:-2].split(', ')]).reshape((1,3)), axis=0)
        #polarization
        line = file.readline()
        community_bi = np.append(community_bi, np.array([float(x) for x in line[1:-2].split(', ')]).reshape((1, 6)),
                                 axis=0)
        line = file.readline()
        community_mo = np.append(community_mo, np.array([float(x) for x in line[1:-2].split(', ')]).reshape((1, 6)),
                                 axis=0)
        line = file.readline()
        community_lp = np.append(community_lp, np.array([float(x) for x in line[1:-2].split(', ')]).reshape((1, 6)),
                                 axis=0)
        line = file.readline() #skip for stat
        degrees['a'].append(float(line[:-1]))
        #strategy of each step
        counter = 0
        while counter < int(T/5):
            line = file.readline()
            t = int(line.split(' ')[1])
            counter += 1
            linesplit = line.split(' [')
            strategy_new = [float(x) for x in linesplit[1][:-2].split(', ')]
            if len(strategy_runs_att[t]) == 0:
                strategy_runs_att[t] = np.array(strategy_new[:4]).reshape((1, 4))
                strategy_runs_udm[t] = np.array(strategy_new[4:7]).reshape((1, 3))
                strategy_runs_hdm[t] = np.array(strategy_new[7:10]).reshape((1, 3))
                strategy_runs_def[t] = np.array(strategy_new[-2:]).reshape((1, 2))
            else:
                strategy_runs_att[t] = np.append(strategy_runs_att[t], np.array(strategy_new[:4]).reshape((1, 4)),
                                                 axis=0)
                strategy_runs_udm[t] = np.append(strategy_runs_udm[t], np.array(strategy_new[4:7]).reshape((1, 3)),
                                                 axis=0)
                strategy_runs_hdm[t] = np.append(strategy_runs_hdm[t], np.array(strategy_new[7:10]).reshape((1, 3)),
                                                 axis=0)
                strategy_runs_def[t] = np.append(strategy_runs_def[t], np.array(strategy_new[-2:]).reshape((1, 2)),
                                                 axis=0)
        file.close()
        print("file", filename, "closed")
    print(components)
    print("running time data collection", (time.time() - start_time_i)/60) #start_time_r

    #collect information for SIR
    for t in range(T):
        sir_count = np.zeros(5) 
        for n in nodes_total:
            if n in opinions_runs[t]:
                sir_runnum = opinions_runs[t][n].shape[0]
                if n in nodes['T']:
                    sir_count[4] += 1
                elif n in nodes['A']:
                    sir_count[3] += sir_runnum / len(filtered_files)
                    # print(sir_count[t])
                else:
                    pb = opinions_runs[t][n][:,0] + opinions_runs[t][n][:,3] * opinions_runs[t][n][:,2]
                    pd = opinions_runs[t][n][:,1] + (1-opinions_runs[t][n][:,3]) * opinions_runs[t][n][:,2]
                    pbtrue = pb <= 0.5
                    pbfalse = pb > 0.5
                    pdtrue = pd <= 0.5
                    pdfalse = pd > 0.5
                    if True in pbtrue * pdtrue:
                        sir_count[0] += sum(pbtrue * pdtrue) / len(filtered_files)  # S
                    if True in pdfalse:
                        sir_count[1] += sum(pdfalse) / len(filtered_files)  # I
                    if True in pbfalse:
                        sir_count[2] += sum(pbfalse) / len(filtered_files)  # R
                    # print(sir_count)
        sir_runs[t] = sir_count

    #result for runs and interactions
    for t in range(T):
        for n in nodes_total:
            if n in opinions_runs[t]:
                opinions_std[t][n] = np.std(opinions_runs[t][n], axis = 0)
                opinions_runs[t][n] = np.mean(opinions_runs[t][n], axis = 0)
                between_std[t][n] = np.std(np.array(between_runs[t][n]))
                between_runs[t][n] = np.mean(np.array(between_runs[t][n]))
                redundancy_std[t][n] = np.std(np.array(redundancy_runs[t][n]))
                redundancy_runs[t][n] = np.mean(np.array(redundancy_runs[t][n]))
    #calculate strategy choices ratio
    print(choice_at)
    print(choice_df)
    print(choice_u)
    print(choice_h)
    print(utility_at)
    print(utility_df)
    print(utility_u)
    print(utility_h)
    choice_at = choice_at / np.sum(choice_at, axis=1, keepdims=True)
    choice_df = choice_df / np.sum(choice_df, axis=1, keepdims=True)
    if phos != 1:
        choice_u = choice_u / np.sum(choice_u, axis=1, keepdims=True)
    if phos != 0:
        choice_h = choice_h / np.sum(choice_h, axis=1, keepdims=True)

    comm_bi_val['a'] = np.mean(community_bi, axis=0)
    comm_mo_val['a'] = np.mean(community_mo, axis=0)
    comm_lp_val['a'] = np.mean(community_lp, axis=0)
    print(comm_bi_val)
    print(comm_mo_val)
    print(comm_lp_val)
    degrees['a'] = sum(degrees['a'])/len(degrees['a'])
    print(degrees)

    #calculate strategy steps
    for t in range(int(T/5)):
        strategy_runs_att[t] = np.mean(strategy_runs_att[t], axis=0)
        strategy_runs_udm[t] = np.mean(strategy_runs_udm[t], axis=0)
        strategy_runs_hdm[t] = np.mean(strategy_runs_hdm[t], axis=0)
        strategy_runs_def[t] = np.mean(strategy_runs_def[t], axis=0)
    print(strategy_runs_att[int(T / 5) - 1])
    print(strategy_runs_def[int(T / 5) - 1])

    #save to file -- opinions_runs
    with open('node_hta_'+ str(int(phos*100)) + '_' + str(int(pa*100))+'.txt', 'w') as f:
        #write parameters
        f.write("%s\n" % list(argv))
        f.write("%s\n" % list(nodes['A']))
        f.write("%s\n" % list(nodes['T']))
        f.write("%s\n" % list(nodes['H']))
        f.write("%s\n" % list(nodes['U']))
        f.write("%s\n" % list(nodes['AS']))
        f.write("%s\n" % list(nodes['D']))
        f.write("%s\n" % list(nodes['E']))
        f.write("%s\n" % list(np.mean(choice_at, axis = 0)))
        f.write("%s\n" % list(np.mean(choice_df, axis = 0)))
        if phos != 1:
            f.write("%s\n" % list(np.mean(choice_u, axis = 0)))
        else:
            f.write("%s\n" % list(choice_u))
        if phos != 0.0:
            f.write("%s\n" % list(np.mean(choice_h, axis = 0)))
        else:
            f.write("%s\n" % list(choice_h))
        f.write("%s\n" % list(np.mean(utility_at, axis = 0)))
        f.write("%s\n" % list(np.mean(utility_df, axis = 0)))
        if phos != 1:
            f.write("%s\n" % list(np.mean(utility_u, axis = 0)))
        else:
            f.write("%s\n" % list(utility_u))
        if phos != 0:
            f.write("%s\n" % list(np.mean(utility_h, axis = 0)))
        else:
            f.write("%s\n" % list(utility_h))

        for item in components: #run lines
            f.write("%s\n" % components_st[item])
        for item in components: #run lines
            f.write("%s\n" % components[item])
        # save betweeness and redundancy of the initial network
        for (x,y) in between_0: #N lines
            f.write("%d %f\n" % (x, y))
        for (x,y) in redundancy_0: #N lines
            f.write("%d %f\n" % (x, y))
        # save strategy of each step
        for item in strategy_runs_att:
            f.write("%s\n" % list(strategy_runs_att[item]))
        for item in strategy_runs_udm:
            f.write("%s\n" % list(strategy_runs_udm[item]))
        for item in strategy_runs_hdm:
            f.write("%s\n" % list(strategy_runs_hdm[item]))
        for item in strategy_runs_def:
            f.write("%s\n" % list(strategy_runs_def[item]))

        f.write("%s %s\n" % (comm_bi_val['b'], list(np.mean(community_bi, axis=0))))
        f.write("%s %s\n" % (comm_mo_val['b'], list(np.mean(community_mo, axis=0))))
        f.write("%s %s\n" % (comm_lp_val['b'], list(np.mean(community_lp, axis=0))))
        f.write("%s\n" % degrees)

    with open('op_hta_'+ str(int(phos*100)) +'_' + str(int(pa*100))+'.txt', 'w') as f:
        for item in opinions_runs:
            f.write("INTER %s\n" % item)
            for node in opinions_runs[item]:
                f.write("%d %s\n" % (node, list(opinions_runs[item][node])))
        for item in opinions_std:
            f.write("INTER %s\n" % item)
            for node in opinions_std[item]:
                f.write("%d %s\n" % (node, list(opinions_std[item][node])))

        for item in between_runs:
            f.write("INTER %s\n" % item)
            for node in between_runs[item]:
                f.write("%d %f\n" % (node, between_runs[item][node]))
        for item in between_std:
            f.write("INTER %s\n" % item)
            for node in between_std[item]:
                f.write("%d %f\n" % (node, between_std[item][node]))

        for item in redundancy_runs:
            f.write("INTER %s\n" % item)
            for node in redundancy_runs[item]:
                f.write("%d %f\n" % (node, redundancy_runs[item][node]))
        for item in redundancy_std:
            f.write("INTER %s\n" % item)
            for node in redundancy_std[item]:
                f.write("%d %f\n" % (node, redundancy_std[item][node]))

    with open('sir_hta_' + str(int(phos * 100)) + '_' + str(int(pa*100))+ '.txt', 'w') as f:
        for item in sir_runs:
            f.write("INTER %s %s\n" % (item, list(sir_runs[item])))

    print("total time: ", (time.time() - start_time_i)/60)

if __name__ == '__main__':
    main(sys.argv[1:])