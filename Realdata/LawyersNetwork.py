#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pyreadstat
from collections import defaultdict
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import time
from algo_new import *
from LawyersNetwork_preprocess import *
import warnings


# In[2]:


target, name_label, name_network, list_cov, remove_isolated, remove_office_3 = ['X', 'status', 'work', ['years w/ firm'], True, True]

# preprocess data
[A, X, Z] = preprocess_lawyer(name_label = name_label, name_network = name_network, list_cov = list_cov, remove_isolated = remove_isolated, remove_office_3 = remove_office_3)
print(Counter(Z)) # class sizes


# In[4]:


K = 2
res_A = fit_SBM(A, K)
res_X = fit_GMM(X, K)
print('Network only:{}'.format(Hamming_aligned(res_A['labels'],Z)/len(Z)))
print('Covariate only:{}'.format(Hamming_aligned(res_X['labels'],Z)/len(Z)))


# In[6]:


list_lbd = 0.5 * np.arange(21)
seed = 1000
dist = ['Gaussian','Network']
# dist = ['Network','Gaussian']
list_lbd = np.arange(0, 20, 0.2)
[paras_hat, errs, AMIs]= TL_demo(X, A, Z, K, dist = dist, list_lbd = list_lbd, show = False)

plt.plot(list_lbd, np.array(errs[:len(list_lbd)])/len(Z), 'k')
# Adaptive clustering
B_bootstrap = 1000
list_q = [0.8, 0.9, 0.95]
# Error estimation
start = time.time()
[errs_hat, phi_hat, psi_hat]= estimate_error(X, A, K = K, dist = dist, list_lbd = list_lbd, B_bootstrap = B_bootstrap, list_q = list_q)
duration = time.time() - start
print('time elapsed: ', duration)
print('Oracle error: {}'.format(np.min(errs)/len(Z)))
# Adaptive selection
print('Adaptive transfer clustering: ')
list_selected_idx = np.argmin(errs_hat, axis = 1)
for (i, q) in enumerate(list_q):
    print('q = {}: {}'.format(q, errs[list_selected_idx[i]]/len(Z)))
fig = plt.figure(figsize=(6, 6))
plot_labels = ['ATC(0.8)', 'ATC(0.9)',  'ATC(0.95)']
plt.plot(list_lbd, np.array(errs[:len(list_lbd)])/len(Z), 'k', label = 'Ground Truth')
colors = ['b', 'g', 'r']
for i in range(errs_hat.shape[0]):
    plt.plot(list_lbd, errs_hat[i]/len(Z), label=plot_labels[i], color=colors[i])

# Adding legend
plt.legend(fontsize=14)

# Adding titles and labels
plt.title('Clustering error v.s. penalty $\\lambda$', fontsize=22)
plt.xlabel(' $\\lambda$', fontsize=22)
plt.ylabel('Clustering Error', fontsize=22)
# plt.savefig('Layers_lambda.png', bbox_inches='tight')


# In[ ]:




