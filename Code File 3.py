# -*- coding: utf-8 -*-
'''For Figure 3'''
import pandas as pd
import numpy as np
import pickle
'''
#test data file is saved in advance by merging to gene list used for clustering as followings
testdata = pd.merge(gene_name_file, scRNAseq, on ='gene_name', how = 'left')
testdata = testdata.fillna(0)
f = open('C:/Users/testdata_16921','wb')
pickle.dump(testdata,f)
f.close()
'''

f = open('C:/Users/testdata_16921', 'rb')
testdata = pickle.load(f)
f.close()
gene_num = len(testdata)

init_data = np.array(testdata['z1_1'])#Select an appropreate scRNAseq data for initial condition
init_ratio = init_data/np.sum(init_data)
target = np.array(testdata['4c1_1'])#Select an appropreate scRNAseq data for target
targetratio = target/np.sum(target)
'''#Activate in Figure 3f
import random
randlist = np.array(range(gene_num), dtype=int)
random.shuffle(randlist)
init_data = np.array(testdata['z1_1'])[randlist]
init_ratio = init_data/np.sum(init_data)
random.shuffle(randlist)
target = np.array(testdata['4c1_1'])[randlist]
targetratio = target/np.sum(target)
'''
f = open('C:/Users/pair_matirix_16921', 'rb')#Select an appropreate file for pair_matirix
pair_matirix3 = pickle.load(f)
f.close()
pair_matirix = [np.mod(pair_matirix3, 2), np.array(pair_matirix3 // 2)]

a_max = 1/10
a_min = 1/1000
a_dec = 1/10

def hier_pairing(x_list):#calculating ratios in pairs from ratio in total
    rlist = []
    for jj in range(int(gene_num - 1)):
        ratio = np.zeros(2, float)
        for ii in range(2):
            ratio[ii] = np.dot(x_list, pair_matirix[ii][jj, :])
        rlist.append([jj, (ratio + 1e-7)/np.sum(ratio + 1e-7)])
    return dict(rlist)

def cal_all_ratio(curr_pairs, n, bias=1e-7):#calculating ratio in total from ratios in pairs
    ratio_list = np.ones(n)
    for jj in range(n - 1):
        curr_ratio = (curr_pairs[jj] + bias)/np.sum((curr_pairs[jj] + bias))
        for ii in range(2):
            ratio_list = ratio_list * (np.ones(n) * curr_ratio[ii])**pair_matirix[ii][jj, :]
    return ratio_list

def mean_square(xratio, target_p):
    error = sum((xratio - target_p)**2)/2
    return error
def stepwise_MSE(xratio, target_p):
    error = float('1'+'{:.0E}'.format(sum((xratio - target_p)**2)/2)[-4:])
    return error
def x_change(xD, target_p, bs, bias=1):
    if np.random.rand() < (a_max*np.exp(np.log10(bs + 1E-7)) + a_min):
        rc = np.random.choice(range(2), p=(xD + bias)/np.sum(xD + bias))
        xD[rc] += 1
    if np.random.rand() < a_dec:
        tot = np.sum(xD)
        if tot > 0:
            dec = stepwise_MSE(xD/tot, target_p)
            xD = np.random.binomial(xD, 1 - dec)
    return xD

init_dict_ratio = hier_pairing(init_data)
target_dict = hier_pairing(target)

#setting the initial condition in pairs
pair_list = []
for j in range(int(gene_num - 1)):
    pair_list.append([j, np.ones(2, int)])
pair_dict = dict(pair_list)
for j in init_dict_ratio:
    bs0 = (np.exp(np.log10(np.dot(init_ratio, pair_matirix[0][j, :] + pair_matirix[1][j, :])+1E-7))*a_max + a_min)*gene_num*a_max
    for i in range(2):
        pair_dict[j][i] = int(round(init_dict_ratio[j][i]*bs0))

tmax = 10**5
tbin = 1000
r_dyn = np.zeros(tbin, float)
ratio_total = np.ones(gene_num, float)
bsize = np.ones(gene_num - 1, float)#coverage of each pair in total is used  instead of a_inc

for t in range(tmax):
    if t%(tmax/tbin) == 0:
        ratio_total = cal_all_ratio(pair_dict, gene_num, bias=1)
        for j in pair_dict:
            bsize[j] = np.dot(ratio_total, pair_matirix[0][j, :] + pair_matirix[1][j, :])
        r_dyn[int(t*tbin/tmax)] = np.corrcoef(ratio_total/np.sum(ratio_total), targetratio)[0, 1]
    for j in pair_dict:
        pair_dict[j] = x_change(xD=pair_dict[j], target_p=target_dict[j], bs=bsize[j], bias=1)

ratio_total = cal_all_ratio(pair_dict, gene_num, bias=1)
