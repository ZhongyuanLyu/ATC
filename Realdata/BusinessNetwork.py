#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import json
import pandas as pd
import numpy as np
import pyreadstat
from collections import defaultdict
import os
import matplotlib.pyplot as plt
from collections import Counter
from algo_new import *
from sparsebm import SBM
from sparsebm import generate_SBM_dataset


# In[2]:


# Read data
with open('BRN.json', 'r') as file:
    text = file.read()
dict_1 = json.loads(text)
A = np.array(dict_1.get('network'))
X = np.array(dict_1.get('feature'))
label = np.array(dict_1.get('labels'))
ID = np.array([i for i in range(A.shape[1])]).tolist()
K = 5
A_mat = A[4,:,:]
res_A = fit_SBM(A_mat, K, sparse_method = False, num_refinement = 0)
res_X = fit_GMM(X, K)


print(len(label),
      Hamming_aligned(res_A['labels'],label)/len(label),
      Hamming_aligned(res_X['labels'],label)/len(label))

# Adaptive clustering
list_lbd = np.arange(0, 60, 3)
[paras_hat, errs, ARIs]= TL_demo(A_mat, X, label, K,  ['Network', 'Gaussian'], list_lbd, show = True, sparse_method = False, num_refinement = 0)
B_bootstrap = 1000
list_q = [0.8, 0.9, 0.95, 0.99]
# Error estimation
start = time.time()
[errs_hat, phi_hat, psi_hat]= estimate_error(A_mat, X, K = K, dist = ['Network', 'Gaussian'], list_lbd = list_lbd, B_bootstrap = B_bootstrap, list_q = list_q, sparse_method = False, num_refinement = 0)
duration = time.time() - start
print('time elapsed: ', duration)
print('Oracle error: {}'.format(np.min(errs)/len(label)))


# Adaptive selection
print('Adaptive transfer clustering: ')
list_selected_idx = np.argmin(errs_hat, axis = 1)
for (i, q) in enumerate(list_q):
    print('q = {}: {}, {}'.format(q, errs[list_selected_idx[i]]/len(label), ARIs[list_selected_idx[i]]))


# Plot
fig = plt.figure(figsize=(6, 6))
plot_labels = ['ATC: q=0.8', 'ATC: q=0.9',  'ATC: q=0.95',  'ATC: q=0.99']
plt.plot(list_lbd, np.array(errs[:-1])/len(label), 'k', label = 'Ground Truth')
colors = ['b', 'g', 'r', 'c']
for i in range(errs_hat.shape[0]):
    plt.plot(list_lbd, errs_hat[i]/len(label), label=plot_labels[i], color=colors[i])
plt.legend(fontsize=14, loc='lower right', bbox_to_anchor=(1, 0.2))
plt.title('Clustering error v.s. penalty $\\lambda$', fontsize=22)
plt.xlabel(' $\\lambda$', fontsize=22)
plt.ylabel('Clustering Error', fontsize=22)
# plt.savefig('Business_lambda.png', bbox_inches='tight')





