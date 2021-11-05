# -*- coding: utf-8 -*-
'''#For the model with an mRNA pool used in Figures 4G-O and 5'''
import numpy as np
import pickle
import random
'''
#test data file is saved in advance by merging to gene list used for clustering as followings
testdata = pd.merge(gene_name_file, scRNAseq, on ='gene_name', how = 'left')
testdata = testdata.fillna(0)
f = open('C:/Users/testdata','wb')
pickle.dump(testdata,f)
f.close()
'''
f = open('C:/Users/testdata', 'rb')
testdata = pickle.load(f)
f.close()
gene_num = len(testdata)
randlist = np.array(range(gene_num), dtype=int)
#random.shuffle(randlist)#Activate this in Figure 4I
init_data = np.array(testdata['z1_1'])#[randlist]
#Select an appropreate scRNAseq data for initial condition
init_ratio = init_data/np.sum(init_data)
#random.shuffle(randlist)#Activate this in Figure 4I
target = np.array(testdata['4c1_1'])#[randlist]
#Select an appropreate scRNAseq data for target.
targetratio = target/np.sum(target)
bias_test = 1#1e-7#Select appropreate bias value
f = open('C:/Users/pair_matirix_No', 'rb')#Select pair_matirix generated in Code File 2
pair_matrix3 = pickle.load(f)
f.close()
pair_matrix = [np.mod(pair_matrix3, 2), np.array(pair_matrix3 // 2)]
f = open('C:/Users/result1_No', 'rb')#Select an appropreate file for result1 in clustering (Code File 2)
pairs_array = np.array(pickle.load(f)[:, 0:3], dtype=int)
#Indexes of left branch on column0, index of right branch on column1 and layer on column2. Indexes from 0 to gene_num - 1 indicates a gene. The row of a pair is calculated by Index - gene_num
f.close()
def hier_pairing(x_list):#Calculate ratios in pairs from ratio in total
    rlist = []
    for jj in range(int(gene_num - 1)):
        ratio = np.zeros(2, float)
        for ii in range(2):
            ratio[ii] = np.dot(x_list, pair_matrix[ii][jj, :])
        rlist.append([jj, (ratio + 1e-7)/np.sum(ratio + 1e-7)])
    return dict(rlist)

def cal_all_ratio(curr_pairs, bias=1e-7):#Calculate ratio in total from ratios in pairs
    ratio_list = np.ones(gene_num)
    for jj in range(gene_num - 1):
        curr_ratio = (curr_pairs[jj] + bias)/np.sum((curr_pairs[jj] + bias))
        for ii in range(2):
            ratio_list = ratio_list * (np.ones(gene_num) * curr_ratio[ii])**pair_matrix[ii][jj, :]
    return ratio_list

def stepwise_MSE(xratio, target_p):
    error = float('1'+'{:.0E}'.format(sum((xratio - target_p)**2)/2)[-4:])
    return error
def step4_error(xratio, target_p):
    error = sum((xratio - target_p)**2)/2
    if error < 1E-3:
        decay = 1E-4
    elif error < 1E-2:
        decay = 1E-3
    elif error < 1E-1:
        decay = 1E-2
    else:
        decay = 1E-1
    return decay
def step3_error(xratio, target_p):
    error = sum((xratio - target_p)**2)/2
    if error < 1E-2:
        decay = 1E-3
    elif error < 1E-1:
        decay = 1E-2
    else:
        decay = 1E-1
    return decay

init_dict_ratio = hier_pairing(init_data)
target_dict = hier_pairing(target)
#setting the initial condition in pairs
pair_list = []
for j in range(int(gene_num - 1)):
    pair_list.append([j, np.ones(2, int)])
pair_dict = dict(pair_list)
for j in init_dict_ratio:
    bs0 = np.dot(init_ratio, pair_matrix[0][j, :] + pair_matrix[1][j, :]) * gene_num
    for i in range(2):
        pair_dict[j][i] = int(round(init_dict_ratio[j][i]*bs0))

mRNA_pool = np.zeros(gene_num, dtype=np.int)
mRNA_num = int(36 * 10**4)
for t in range(mRNA_num):
    i = np.random.choice(range(gene_num), p=init_ratio)
    mRNA_pool[i] += 1
mRNA_init = np.array(mRNA_pool, dtype=np.int)#For recording of initial state

#To select the starting layer in Monte Carlo tree search
layer_num = 6#Select one pair among top 7 layers
top_layers = np.zeros((2**(layer_num + 1) - 1, 2), dtype=int)
top_layers[0, 0] = gene_num - 2#index of the top layer
top_genes = np.ones((2**(layer_num + 1) - 1, 2), dtype=int)*10**6#10**6 indicates not-annotated yet
i = 0
ij = 0
for j in range(layer_num):
    layers = np.ones(2**j, dtype=int)*10**6
    layers1 = np.where(pairs_array[:, 2] == j)[0]#pairs_array[:, 2] indicates layer of the pair
    layers[:len(layers1)] = layers1#list of index of pairs on j layer
    for k in range(2**j):
        top_layers[(i+k) * 2 + 1, 1] = layers[k]#index of the parent pair
        top_layers[(i+k) * 2 + 2, 1] = layers[k]
        if layers[k] == 10**6:#when gene is selected in a pair, the number of pairs is fewer than 2**j
            top_layers[(i+k) * 2 + 1, 0] = 10**6
            top_layers[(i+k) * 2 + 2, 0] = 10**6
        else:
            top_layers[(i+k) * 2 + 1, 0] = pairs_array[layers[k], 0] - gene_num#index of left branch
            top_layers[(i+k) * 2 + 2, 0] = pairs_array[layers[k], 1] - gene_num#index of right branch
            if j < layer_num - 1:#if gene is selected in the pair
                if top_layers[(i+k) * 2 + 1, 0] < 0:#the gene is recorded in top_genes
                    top_genes[ij, 0] = top_layers[(i+k) * 2 + 1, 0]#to add them in the lower layers
                    top_genes[ij, 1] = j
                    ij += 1
                if top_layers[(i+k) * 2 + 2, 0] < 0:
                    top_genes[ij, 0] = top_layers[(i+k) * 2 + 1, 0]
                    top_genes[ij, 1] = j
                    ij += 1
    i += 2**j
top_genes = np.delete(top_genes, np.where(top_genes == 10**6)[0], axis=0)
top_g_n = 0
for i in range(len(top_genes[:, 0])):
    j = top_genes[i, 1]
    for k in range(layer_num - 1 - j):#to add genes in the lower layers
        index_g = np.where(top_layers[:, 0] == top_genes[i, 0])[0][0]
        top_layers[np.where(top_layers[:, 0] == 10**6)[0][-1], :] = [top_layers[index_g, 0], index_g]
        top_g_n += 1
top_layers = np.delete(top_layers, np.where(top_layers == 10**6)[0], axis=0)

top_num = len(top_layers[:, 0])
psize = np.ones(top_num, dtype=float)#indicates the coverage of each pair in top_layes

def cal_psize(curr_pairs, bias=1):#to calcurate the coverage of each pair
    tp_s = np.ones(top_num, dtype=float)
    for ii in range(1, int(top_num - top_g_n)):
        pair = np.array(curr_pairs[top_layers[ii, 1]])
        if ii%2 == 1:
            tp_s[ii] = (pair[0] + bias)/np.sum(pair + bias)*tp_s[np.where(top_layers[:, 0] == top_layers[ii, 1])[0][0]]
        else:
            tp_s[ii] = (pair[1] + bias)/np.sum(pair + bias)*tp_s[np.where(top_layers[:, 0] == top_layers[ii, 1])[0][0]]
    for ii in range(top_g_n):
        tp_s[top_num - top_g_n + ii] = tp_s[top_layers[top_num - top_g_n + ii, 1]]
    return tp_s#np.sum(tp_s) = 7

a_dec = 1/10
def rand_forest(curr_pairs, tp_s, bias=1):#Monte Carlo Tree search
    ii = top_layers[np.random.choice(range(len(psize)), p=tp_s/np.sum(tp_s)), 0]
    while True:
        if ii < 0:#if gene is selected
            ii += gene_num
            break
        xD = np.array(curr_pairs[ii], dtype=int)
        rc = np.random.choice(range(2), p=(xD + bias)/np.sum(xD + bias))
        curr_pairs[ii][rc] += 1#competitive amplification
        ii = pairs_array[int(ii), rc]#index of the next down stream pair
        if ii > gene_num - 1:
            ii -= gene_num
        else:
            break
    return [curr_pairs, int(ii)]
def pairs_dec(curr_pairs, step=4):#MSE dependent decay
    for ii in curr_pairs:
        if np.random.rand() < a_dec:
            xD = np.array(curr_pairs[ii], dtype=int)
            tot = np.sum(xD)
            target_p = np.array(target_dict[ii], dtype=float)
            if tot > 0:
                if step == 4:
                    dec = step4_error(xD/tot, target_p)
                elif step == 3:
                    dec = step3_error(xD/tot, target_p)
                elif step == 1:
                    dec = stepwise_MSE(xD/tot, target_p)
                curr_pairs[ii] = np.random.binomial(xD, 1 - dec)
    return curr_pairs

tmax = int(10**6 / 2)#10**6 in the case with shuffle and Figure 4J
tbin = 2000#for recording
mRNA_dyn = np.zeros((tbin + 1, 2), float)
gxp_dyn = np.zeros((tbin + 1, 2), float)
ratio_total = np.ones(gene_num, float)

'''#To continue the differentiation process in Figure 5, repeat the followings after changing target
target = np.array(testdata['8c1_1'])#Select an appropreate scRNAseq data for target
targetratio = target/np.sum(target)
target_dict = hier_pairing(target)
'''
for t in range(tmax):
    if t%(tmax/tbin) == 0:
        ratio_total = cal_all_ratio(curr_pairs=pair_dict, bias=1e-7)
        mRNA_dyn[int(t*tbin/tmax), 0] = np.corrcoef(mRNA_pool/mRNA_num, init_ratio)[0, 1]
        mRNA_dyn[int(t*tbin/tmax), 1] = np.corrcoef(mRNA_pool/mRNA_num, targetratio)[0, 1]
        gxp_dyn[int(t*tbin/tmax), 0] = np.corrcoef(ratio_total, init_ratio)[0, 1]
        gxp_dyn[int(t*tbin/tmax), 1] = np.corrcoef(ratio_total, targetratio)[0, 1]

    psize = cal_psize(pair_dict, bias=bias_test)
    [pair_dict, sel_gene] = rand_forest(curr_pairs=pair_dict, tp_s=psize, bias=bias_test)
    i = np.random.choice(range(gene_num), p=mRNA_pool/mRNA_num)
    mRNA_pool[i] -= 1
    mRNA_pool[sel_gene] += 1
    pair_dict = pairs_dec(curr_pairs=pair_dict, step=4)#step=3 in 3-step error

ratio_total = cal_all_ratio(curr_pairs=pair_dict, bias=1e-7)
mRNA_dyn[-1, 0] = np.corrcoef(mRNA_pool/mRNA_num, init_ratio)[0, 1]
mRNA_dyn[-1, 1] = np.corrcoef(mRNA_pool/mRNA_num, targetratio)[0, 1]
gxp_dyn[-1, 0] = np.corrcoef(ratio_total, init_ratio)[0, 1]
gxp_dyn[-1, 1] = np.corrcoef(ratio_total, targetratio)[0, 1]
