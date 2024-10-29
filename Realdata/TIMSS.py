#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


### BSBS24F	SCI\AGREE\I AM GOOD AT SCIENCE ###

country_code = 'SGP'
############## Target data: Science #################
label_char = ['BSBS24F']
item_char = 'BSBS'

num_group_ME = summary_data(country_code, 'ME')
num_group_SE = summary_data(country_code, 'SE')

for i in range(num_group_ME):
    preprocess_data(country_code, label_char, 'ME', item_char, 'MAT', i)
for i in range(num_group_SE):
    preprocess_data(country_code, label_char, 'SE', item_char, 'SCI', i)


group_id = 0
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

warnings.filterwarnings('ignore')
res_response_SE = fit_LCM_softEM(response_mat_SE, K)
res_response_ME = fit_LCM_softEM(response_mat_ME, K)
print('Number of samples:{}'.format(len(new_labels)))
print('Science only:{}'.format(Hamming_aligned(res_response_SE['labels'],new_labels)/len(new_labels)))
print('Math only:{}'.format(Hamming_aligned(res_response_ME['labels'],new_labels)/len(new_labels)))
     
list_lbd = np.arange(0, 20, 0.2)

[paras_hat, errs, ARIs]= TL_demo(response_mat_SE, response_mat_ME, new_labels, K, ['Bernoulli', 'Bernoulli'], list_lbd, show = False, spectral = False)
# plt.plot(list_lbd, np.array(errs[:len(list_lbd)])/len(new_labels), 'k')

# Adaptive clustering
B_bootstrap = 1000
list_q = [0.8, 0.9, 0.95, 0.99]
# Error estimation
start = time.time()
[errs_hat, phi_hat, psi_hat]= estimate_error(response_mat_SE, response_mat_ME, K = K, dist = ['Bernoulli', 'Bernoulli'], list_lbd = list_lbd, B_bootstrap = B_bootstrap, list_q = list_q, spectral = False)
duration = time.time() - start
print('time elapsed: ', duration)
print('Oracle error: {}'.format(np.min(errs)/len(new_labels)))
# Adaptive selection
print('Adaptive transfer clustering: ')
list_selected_idx = np.argmin(errs_hat, axis = 1)
for (i, q) in enumerate(list_q):
    print('q = {}: {}'.format(q, errs[list_selected_idx[i]]/len(new_labels)))
fig = plt.figure(figsize=(6, 6))
plot_labels = ['ATC: q=0.8', 'ATC: q=0.9',  'ATC: q=0.95',  'ATC: q=0.99']
plt.plot(list_lbd, np.array(errs[:len(list_lbd)])/len(new_labels), 'k', label = 'Ground Truth')
colors = ['b', 'g', 'r', 'c']
for i in range(errs_hat.shape[0]):
    plt.plot(list_lbd, errs_hat[i]/len(new_labels), label=plot_labels[i], color=colors[i])

# Adding legend
plt.legend(fontsize=14)

# Adding titles and labels
plt.title('Clustering error v.s. penalty $\\lambda$', fontsize=22)
plt.xlabel(' $\\lambda$', fontsize=22)
plt.ylabel('Clustering Error', fontsize=22)
# plt.savefig('TIMSS_lambda_{group_id}.png', bbox_inches='tight')



## Plot of \hat\Theta ##

[paras_hat, list_Z_hat] = TL_candidates(response_mat_SE, response_mat_ME, K, ['Bernoulli', 'Bernoulli'], list_lbd, show = False, spectral = False)
best_Z_099 = list_Z_hat[list_selected_idx[3]]
theta_1 = response_mat_SE[best_Z_099==0].mean(axis=0)
theta_2 = response_mat_SE[best_Z_099==1].mean(axis=0)
Theta_hat = np.vstack((theta_1, theta_2)).T

response_SE = pd.read_csv(f'TIMSS/df_mc_{country_code}_SE_group_{group_id}.csv')
SE_items = response_SE.columns.tolist()
file_path = 'T19_G8_Item Information/eT19_G8_Item Information.xlsx'
df = pd.read_excel(file_path, sheet_name='SCI')
filtered_df = df[df['Item ID'].isin(SE_items)]
result_df = filtered_df[['Item ID', 'Topic Area', 'Cognitive Domain', 'Content Domain', 'Label']]

content_domain = result_df['Content Domain'].tolist()
topic_area = result_df['Topic Area'].tolist()
item_labels = [f'{cd}, {ta}' for cd, ta in zip(content_domain, topic_area)]
row_variances = np.var(Theta_hat, axis=1)
sorted_indices = np.argsort(row_variances)[::-1]
theta_matrix_sorted = Theta_hat[sorted_indices]
item_labels_sorted = [item_labels[i] for i in sorted_indices]

fig, ax = plt.subplots(figsize=(16, 12))  
cax = ax.matshow(theta_matrix_sorted, cmap='coolwarm', aspect=0.1)  
ax.set_yticks(range(len(item_labels_sorted)))
ax.set_yticklabels(item_labels_sorted, fontsize=14, fontweight='normal')
ax.set_xticks(range(theta_matrix_sorted.shape[1]))
ax.set_xticklabels([f'Class {i+1}' for i in range(theta_matrix_sorted.shape[1])], fontsize=14, fontweight='bold')
for (i, j), val in np.ndenumerate(theta_matrix_sorted):
    ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=14, fontweight='normal', color='black')

# Set axis labels and title with increased and bold font
ax.set_xlabel('Classes', fontsize=16, fontweight='bold')
ax.set_ylabel('Cognitive Domain, Topic Area', fontsize=16, fontweight='bold')
ax.set_title('Item parameters by ATC', fontsize=18,  fontweight='bold')
plt.tight_layout()

# Save the plot to a file
# plt.savefig('theta_matrix_plot_with_domain_and_area.png')

# Display the plot
plt.show()

