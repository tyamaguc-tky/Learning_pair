# -*- coding: utf-8 -*-
"""
Codes for Figures 1-3
"""
import numpy as np
import pickle
import random
listlen = 2#4 or 8 in Figure 2B,C
listIndex = 1#1 in Fig 1. 1,2,3,6 in Figure 2. 12 in Figure 3.
gene_num = int(listlen**listIndex)
gene = np.ones(gene_num, int)#indicate [xA, xB] in Figure 1
target = np.arange(gene_num) + 1
'''In Figure 3B-F, 4096 high expression genes are selected from GSE96706. 
The imported csv file is used as target. 
target = np.exp2('data')'''
targetratio = target/np.sum(target)#Target ratio is [1/3, 2/3]
pair_num = int((gene_num - 1)/(listlen - 1))#number of pairs
pair_list = []
for j in range(pair_num):
    pair_list.append([j, np.ones(listlen, int)])
pair_dict = dict(pair_list)#dictionary of pairs. Initial state is [1, 1]
'''#In Figure 3C-D, pickled pair_dict at the end of Figure 3B is loaded as the initial state.
f = open('pair_dict_file','rb')
pair_dict = pickle.load(f)
f.close()
'''
randlist = list(range(gene_num))
#random.shuffle(randlist)#Activate to shuffle in Figure 2E,H
'''1 and 0 in pair_matrix indicates whether the gene is in the pair or not'''
pair_matrix = np.zeros((pair_num, listlen, gene_num), int)
k = 0
for j in range(int(gene_num/listlen)):
    for i in range(listlen):
        pair_matrix[j, i, randlist[k]] += 1
        k += 1
k = int(gene_num/listlen)
k0 = 0
k1 = int(gene_num/listlen)
while k1/listlen >= 1:
    k1 = int(k1/listlen)
    for j in range(k1):
        for i in range(listlen):
            pair_matrix[k + j, i, :] = np.sum(pair_matrix[k0 + j*listlen + i, :, :], axis=0)
    k0 = int(k)
    k += k1

def hier_pairing(x_list):#Convert a list to dictionary of pairs as a ratio
    rlist = []
    for jj in range(pair_num):
        ratio = np.zeros(listlen, float)
        for ii in range(listlen):
            ratio[ii] = np.dot(x_list, pair_matrix[jj, ii, :])
        rlist.append([jj, (ratio + 1e-7)/np.sum(ratio + 1e-7)])
    return dict(rlist)
target_dict = hier_pairing(target)
add_dict = hier_pairing(gene)#[0.5, 0.5]

def mean_square(xratio, target_p):
    error = sum((xratio - target_p)**2)/len(xratio)
    return error
def stepwise_MSE(xratio, target_p):
    error = float('1'+'{:.0E}'.format(sum((xratio - target_p)**2)/len(xratio))[-4:])
    return error
def step_error(xratio, target_p):
    error = sum((xratio - target_p)**2)/len(xratio)
    # if error < 1E-5:
    #     decay = 1E-6
    # elif error < 1E-4:
    #     decay = 1E-5
    # elif error < 1E-3:
    #     decay = 1E-4
    if error < 1E-2:#3-step error. Modefy this part in other step error
        decay = 1E-3
    elif error < 1E-1:
        decay = 1E-2
    else:
        decay = 1E-1
    return decay
a_inc = 1/10
gamma = 0#Change to 0~1 in Figure 1L,M
bias = 1#1e-7 in Figure 1K-M
def x_change(add_ratio, xD, target_p, mse=1):#Change in each pair
    if np.random.rand() < a_inc:
        if np.random.rand() < gamma:#Additive increae
            rc = np.random.choice(range(listlen), p=add_ratio)
            xD[rc] += 1
        else:#Competitive amplification
            rc = np.random.choice(range(listlen), p=(xD + bias)/np.sum(xD + bias))
            xD[rc] += 1
    tot = np.sum(xD)
    if tot > 0:
        if mse == 0:
            dec = mean_square(xD/tot, target_p)/10
        elif mse == 1:
            dec = stepwise_MSE(xD/tot, target_p)/10
        elif mse == 2:
            dec = step_error(xD/tot, target_p)/10
        elif mse == 3:
            dec = 0.0001#decay probability
        xD = np.random.binomial(xD, 1 - dec)
    return xD
t_index_max = 5
tmax = 10**t_index_max
tbin = 1000# for recording
data_all = np.zeros((gene_num, tbin + 1), float)
r_dyn = np.zeros(tbin, float)
ratio_total = np.ones(gene_num)
for t in range(tmax):
    if t%(tmax/tbin) == 0:#for recording
        ratio_total = np.ones(gene_num)#Calculate ratio in total from ratios in pairs
        for j in pair_dict:
            TF_ratio = (pair_dict[j] + 1e-7)/np.sum((pair_dict[j] + 1e-7))
            for i in range(listlen):
                ratio_total = ratio_total * (np.ones(gene_num) * TF_ratio[i])**pair_matrix[j, i, :]
        data_all[:, int(t*tbin/tmax)] = ratio_total#pair_dict[0] in Figure 1
        r_dyn[int(t*tbin/tmax)] = np.corrcoef(ratio_total/np.sum(ratio_total), targetratio)[0, 1]
    for jk in pair_dict:
        pair_dict[jk] = x_change(add_ratio=add_dict[jk], xD=pair_dict[jk], target_p=target_dict[jk], mse=0)

ratio_total = np.ones(gene_num)
for j in pair_dict:
    TF_ratio = (pair_dict[j] + 1e-7)/np.sum((pair_dict[j] + 1e-7))
    for i in range(listlen):
        ratio_total = ratio_total * (np.ones(gene_num)*TF_ratio[i])**pair_matrix[j, i, :]
data_all[:, -1] = ratio_total#pair_dict[0] in Figure 1
#f = open('pair_dict_file','wb')
#pickle.dump(pair_dict, f)
#f.close()
