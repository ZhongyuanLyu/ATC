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
from TIMSS_preprocess import *
import warnings
import seaborn as sns

def ATC_TIMSS(country_code, label_char, group_id, B_bootstrap = 10, resample = 0):
    response_mat_ME = pd.read_csv(f'TIMSS/df_mc_{country_code}_ME_group_{group_id}.csv').to_numpy()
    response_mat_SE = pd.read_csv(f'TIMSS/df_mc_{country_code}_SE_group_{group_id}.csv').to_numpy()
    label = pd.read_csv(f'TIMSS/df_label_{country_code}_{label_char[0]}_group_{group_id}.csv').to_numpy()
    ############## Label setting: K = 2 #################
    K = 2
    set_1 = {1}
    new_labels = np.array([
        1 if all(val in set_1 for val in sample) else 0
        for sample in label
    ])
    ############ Parameter setting ############
    list_q = [0.8, 0.9, 0.95, 0.99]
    list_lbd = np.arange(0, 20, 0.2)
    warnings.filterwarnings('ignore')
    ###############  Point estimation ###############
    if resample == 0:
        bootstrap_errs = np.ones(8)
        res_response_SE = fit_LCM_softEM(response_mat_SE, K)
        res_response_ME = fit_LCM_softEM(response_mat_ME, K)
        bootstrap_errs[1] = Hamming_aligned(res_response_SE['labels'],new_labels)/len(new_labels)
        bootstrap_errs[2] = Hamming_aligned(res_response_ME['labels'],new_labels)/len(new_labels)
        [_, errs, _]= TL_demo(response_mat_SE, response_mat_ME, new_labels, K, ['Bernoulli', 'Bernoulli'], list_lbd, show = False, spectral = False)
        bootstrap_errs[3] = np.min(errs[:len(list_lbd)])/len(new_labels)
        bootstrap_errs[0] = errs[len(list_lbd)]/len(new_labels)
        # Adaptive clustering
        [errs_hat, _, _]= estimate_error(response_mat_SE, response_mat_ME, K = K, dist = ['Bernoulli', 'Bernoulli'], list_lbd = list_lbd, B_bootstrap = B_bootstrap, list_q = list_q, spectral = False)
        # Adaptive selection
        list_selected_idx = np.argmin(errs_hat, axis = 1)
        for i in range(len(list_q)):
            bootstrap_errs[i+4] = errs[list_selected_idx[i]]/len(new_labels)
    ###############  Bootstrap resample ###############
    elif resample > 0:
        bootstrap_errs = np.zeros((resample, 8))
        n_samples = response_mat_ME.shape[0]
        for i in range(resample):
            # Resample indices
            indices = np.random.choice(n_samples, n_samples, replace=True)
            response_mat_ME_resampled = response_mat_ME[indices, :]
            response_mat_SE_resampled = response_mat_SE[indices, :]
            new_labels_resampled = new_labels[indices]
            
            # Compute errors for resampled data
            res_response_SE = fit_LCM_softEM(response_mat_SE_resampled, K)
            res_response_ME = fit_LCM_softEM(response_mat_ME_resampled, K)
            bootstrap_errs[i, 1] = Hamming_aligned(res_response_SE['labels'], new_labels_resampled) / len(new_labels_resampled)
            bootstrap_errs[i, 2] = Hamming_aligned(res_response_ME['labels'], new_labels_resampled) / len(new_labels_resampled)
            _, errs, _ = TL_demo(response_mat_SE_resampled, response_mat_ME_resampled, new_labels_resampled, K, ['Bernoulli', 'Bernoulli'], list_lbd, show=False, spectral=False)
            bootstrap_errs[i, 3] = np.min(errs[:len(list_lbd)]) / len(new_labels_resampled)
            bootstrap_errs[i, 0] = errs[len(list_lbd)] / len(new_labels_resampled)
            
            # Adaptive clustering
            errs_hat, _, _ = estimate_error(response_mat_SE_resampled, response_mat_ME_resampled, K=K, dist=['Bernoulli', 'Bernoulli'], list_lbd=list_lbd, B_bootstrap=B_bootstrap, list_q=list_q, spectral=False)
            list_selected_idx = np.argmin(errs_hat, axis=1)
            for j in range(len(list_q)):
                bootstrap_errs[i, j+4] = errs[list_selected_idx[j]] / len(new_labels_resampled)
    return bootstrap_errs


# In[14]:


############## Country: SGP #################
country_code = 'SGP'
############## Target data: Science #################
label_char = ['BSBS24F']
item_char = 'BSBS'
num_resample = 2
grp0_err = np.zeros((num_resample, 8))
grp0_err = ATC_TIMSS(country_code, label_char, 0, B_bootstrap = 1, resample = num_resample)

bootstrap_errs = grp0_err[:,(0,1,2,3,4,5,6)]
plt.figure(figsize=(6, 6))
ax = sns.violinplot(data=bootstrap_errs)
# Set the tick positions and labels
ax.set_xticks(range(bootstrap_errs.shape[1]))
ax.set_xticklabels(['DP', 
                        'Target', 
                        # 'Source', 
                        'Oracle', 
                        'ATC\n(0.8)', 
                        'ATC\n(0.9)', 
                        'ATC\n(0.95)', 
                        'ATC\n(0.99)'], fontsize=12)
# ax.set_xticklabels(['DP', 'Target only', 'Source only', 'Oracle', 
#                     'ATC: q=0.8', 'ATC: q=0.9', 'ATC: q=0.95', 'ATC: q=0.99'], fontsize=12)
plt.title('Clustering Error via Resampling', fontsize=22)
plt.xlabel('Type', fontsize=22)
plt.ylabel('Clustering Error', fontsize=22)
plt.tight_layout()

# plt.savefig('TIMSS_resample.png', bbox_inches='tight')


# In[ ]:




