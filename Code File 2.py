# -*- coding: utf-8 -*-
"""
Codes for hierarchical clustering analysis used in Figures 3-5
"""
import pickle
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, cdist
#import scRNA-seq data
'''
For Figure 3-4 clustering, E-MTAB-3929, GSM2257302, and GSE75748 were used, which are available from the following web site;
https://www.ebi.ac.uk/arrayexpress/experiments/E-MTAB-3929/
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE85066
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE75748
['APS.p1c1r2','D2_25somitomere.p9c1r1','DLL1PXM.p8c1r1','Earlysomite.p10c2r8','H7hESC.p7c1r4','LatM.p3c1r1','MPS3.p5c1r1','Sclerotome.p2c1r1','cDM.p4c1r1','max_ES']
were selected from GSM2257302_All_samples_sc_tpm.txt
['H9.00hb4s_001','H9.12h_001','H9.24h_013','H9.36h_001','H9.72h_001','H9.96h_001']
were selected from GSE75748_sc_time_course_ec.csv
['E3.1.443','E4.1.1','E5.1.26','E6.1.72','E7.2.138']
were selected from E-MTAB-3929
Data was merged using gene name
Data = Data.fillna(0)
Data_sum = Data.sum(axis=0).values
for i in range(len(Data_sum)):
    Data.iloc[:, i] = Data.iloc[:, i]/Data_sum[i]*10**6
Data['max'] = SumList=Data.iloc[:, 1:].max(axis=1)
Data = Data[Data['max']>10]#Data contained 16921 genes
f = open('C:/Users/TPM_DATA10','wb') #read_geneID
pickle.dump(Data, f)
f.close()
'''
'''
#If gene name is not available, following code is applied to identify gene name from gene ID available in scRNAseq
scRNAseq = pd.read_csv('C:/Users/file_name.txt',sep='\t')
scRNAseq.insert(0, 'gene_name', 0)
import mygene
mg = mygene.MyGeneInfo()
#import types
for i in range(len(scRNAseq)):
    if mg.getgene(scRNAseq.iloc[i,1].split('.')[0], fields='symbol') is None:
        scRNAseq.iloc[i,0] = None
    else:
        scRNAseq.iloc[i,0] = mg.getgene(scRNAseq.iloc[i,1].split('.')[0], fields='symbol')['symbol']
Data1 = scRNAseq.groupby('gene_name').sum()
'''
''' For Figure 5
GSE97531_norm_gene_count_table for 515 peripheral blood cells
HTSeq_counts_donor_207B.txt and HTSeq_counts_donor_313C from GSE143567 for 836 hematopoietic stem cell
GSE89497_Human_Placenta_TMP_V2.txt for 1567 trophoblast and stromal cells of placenta
GSE95140_human_single-cardiomyocyte_RNA-seq.txt for 559 cardiomyocytes
GSE111976_ct.csv for 2148 endometrium cells from uterus
GSE160048_human_glom_single_cell_rpkms.txt for 766 renal cells from kidney biopsy
GSE132149_sc16_counts.txt for 91 fallopian tube epithelial cells
GSE133707_P1_Mac.txt and GSE133707_P1_Per.txt for 2036 retina cells from eyes
GSM2295850_F_10W_embryo1_gene_expression.txt and
GSM2306040_M_25W_embryo1_101_gene_expression.txt for 134 primordial germ cell
GSE52529_fpkm_matrix.txt for 372 in vitro cultured primary myoblasts
GSM2257302_All_samples_sc_tpm.txt for 498 in vitro cultured embryonic stem cells and early mesoderm progenitors
GSE75748_sc_time_course_ec.csv for 758 in vitro cultured embryonic stem cells and endoderm progenitors
rpkm file in E-MTAB-3929 for 1529 cells from early preimplantation embryos
were merged and get Data as pd.DataFrame format
Datasets are available from the ncbi web site, by changing GSE***** to dataID;
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE*****
'''
'''
Data_n_nan = Data.iloc[:5000, :].dropna(how='any')
Data_n_nan = Data_n_nan.append(Data.iloc[5000:10000, :].dropna(how='any'))#separate data to save memory
Data_n_nan = Data_n_nan.append(Data.iloc[10000:15000, :].dropna(how='any'))
Data_n_nan = Data_n_nan.append(Data.iloc[15000:20000, :].dropna(how='any'))
Data_n_nan = Data_n_nan.append(Data.iloc[20000:25000, :].dropna(how='any'))
Data_n_nan = Data_n_nan.append(Data.iloc[25000:30000, :].dropna(how='any'))
Data_n_nan = Data_n_nan.groupby('gene_name').max()
Data_n_nan = Data_n_nan.drop('geneID', axis=1)
f = open('C:/Users/DATA_all', 'wb') #read_geneID_201127
pickle.dump(Data_n_nan, f)#All_nonnan30
f.close()
'''

#clustering
f = open('C:/Users/TPM_DATA10', 'rb')#import scRNAseq data normalized by TPM in pd.DataFrame format
Data = pickle.load(f).iloc[:, 1:-1]#colume 0 is gene name. colume -1 is max
f.close()
gene_num = len(Data)
cell_num = len(Data.columns)
mData = np.array(Data/10**3)
'''#For clustering used in Figure 5
f = open('C:/Users/DATA_all','rb')
Data = np.array(pickle.load(f).iloc[:,1:], dtype=int)
f.close()
total = np.sum(Data, axis=0)
mData = np.array(Data[:,total!=0]/total[total!=0], dtype = np.float16)
gene_num = len(mData[:,0])
cell_num = len(mData[0,:])
'''
result1 = np.zeros((gene_num-1, 5), int)

def condensed_index(n, i, j):
    """
    Calculate the condensed index of element (i, j) in an n x n condensed
    matrix.
    """
    if i < j:
        return np.int(n * i - (i * (i + 1) / 2) + (j - i - 1))
    elif i > j:
        return np.int(n * j - (j * (j + 1) / 2) + (i - j - 1))
"""Perform hierarchy clustering using nearest-neighbor chain algorithm in the python tool."""
def uncondensed(n, cmin):
    i = 0
    while cmin >= n * i - (i * (i + 1) / 2):
        i += 1
    i -= 1
    j = cmin - n * i + (i * (i + 1) / 2) + i + 1
    return np.array([i, j], dtype=np.int)
def find_min(dists, data_array, n):
    Z_arr = np.empty((n - 1, 4))#The number of observations. n : int
    Z = Z_arr
    D = dists.copy()# Distances between clusters with condensed index.
    size = np.ones(n, dtype=np.int)  # Sizes of clusters.
    index_list = np.array(range(n), int)
    for k in range(n - 1):
        cmin = np.argmin(D)
        [x, y] = uncondensed(n=n, cmin=cmin)
        nx = size[x]
        ny = size[y]
        Z[k, 0] = index_list[x]
        Z[k, 1] = index_list[y]
        Z[k, 2] = D[cmin]
        Z[k, 3] = nx + ny
        size[x] = 0# Cluster x is dropped.
        size[y] = nx + ny# Cluster y is replaced with the new cluster
        data_array[y, :] = data_array[x, :] + data_array[y, :]# Sum
        index_list[y] = int(n + k)
        # Update the distance matrix.
        for i in range(n):
            ni = size[i]
            if ni == 0 or i == x or i == y:
                continue
            #x is not changed but has be dropped. y gets new distance
            D[condensed_index(n, i, y)] = cdist(data_array[i, :].reshape(1, cell_num), data_array[y, :].reshape(1, cell_num), lambda u, v: (np.max([np.sum(u**2) * np.sum(v**2) - np.dot(u, v)**2, 0]))**(1/2))[0, 0]
            D[condensed_index(n, i, x)] = np.inf
        D[condensed_index(n, x, y)] = np.inf
    return Z_arr

def pair_matirix_cal(result1, inf_num=0):
   # gene_num = len(result1) + 1
    matrix = [np.zeros((gene_num - 1, gene_num), dtype='int8'), np.zeros((gene_num-1, gene_num), dtype='int8')]
    for i in range(gene_num - 1 - inf_num):
        for j in range(2):
            k = int(result1[i, j])
            if k < gene_num:
                matrix[j][i, k] += 1
            else:
                matrix[j][i, :] += matrix[0][k - gene_num, :] + matrix[1][k - gene_num, :]
    matrix3 = matrix[0] + matrix[1] * 2
    return matrix3

#pdist calculates distance between all pairs (using AreaSum method)
cond_dist = pdist(mData, lambda u, v: (np.max([np.sum(u**2) * np.sum(v**2) - np.dot(u, v)**2, 0]))**(1/2))
result1 = find_min(dists=cond_dist, data_array=mData, n=gene_num)

'''#for clustering with Ward, WCO, or Single, choose the relevant metric and method
#Activate metric = 'euclidean', and method='ward') for Ward
#Activate metric = = 'cosine', and method='weighted') for WCO
#Activate metric = 'euclidean', and method='single') for Single

from scipy.cluster.hierarchy import linkage#, dendrogram
result1 = linkage(mData,
                  #metric = 'correlation',
                  #metric = 'cosine',
                  metric = 'euclidean',
                  #method= 'single')
                  #method = 'average')
                  #method='weighted')
                  method='ward')
'''
'''#for clustering with CvSum or Cvarea, modify the function (u,v) in pdist and cdist in find_min() as follows
from statistics import variance, stdev, mean
# for CvSum method
cond_dist = pdist(mData, lambda u, v: stdev(u + v))
D[condensed_index(n, i, y)] = cdist(data_array.iloc[i, :].values.reshape(1,cell_num), data_array.iloc[y, :].values.reshape(1,cell_num), lambda u, v: stdev(u + v))[0,0]
# for Cvarea method
cond_dist = pdist(mData, lambda u, v: stdev(u + v)/mean(u + v) * (np.max([np.sum(u**2) * np.sum(v**2) - np.dot(u, v)**2,0]))**(1/2))
D[condensed_index(n, i, y)] = cdist(data_array.iloc[i, :].values.reshape(1,cell_num), data_array.iloc[y, :].values.reshape(1,cell_num), lambda u, v: (stdev(u + v))/mean(u + v) * (np.max([np.sum(u**2) * np.sum(v**2) - np.dot(u, v)**2,0]))**(1/2))[0,0]
'''
pair_matirix = pair_matirix_cal(result1)
#result1 and pair_matirix is in Supplementary Table1
f = open('C:/Users/result1_No','wb')
pickle.dump(result1,f)
f.close()
f = open('C:/Users/pair_matirix_No','wb')
pickle.dump(pair_matirix,f)
f.close()
