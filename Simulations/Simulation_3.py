#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import time
from algo_new import *
from realdata import *
import warnings


# In[6]:


###############################
### Gaussian Mixture Model ####
###############################

######################## Time v.s. n ################################

# parameters for generating models
dist = ['Gaussian','Gaussian']
d = 10   # dimension 
K = 2    # number of clusters
tmp = np.random.normal(0, 1, size=d)

# parameters for bootstrap 
B_bootstrap = 1000
list_q = [0.8, 0.9, 0.95]
seed = 1000
np.random.seed(seed)

# parameters for simulation
nsim = 50
n_list = np.arange(400, 850, 50)  # number of samples
epsilon = 0.2

# to store runtime per (n, sim)
time_eps = np.zeros((len(n_list), nsim))

# start simulations
for i in range(nsim):
    print(f'simulation iteration = {i+1}\n')
    for j, n in enumerate(n_list):
        mean = 0.2 * np.log(n) * tmp / np.linalg.norm(tmp)
        pars = {'mean': mean, 'covariances': np.identity(d)}
        t0 = time.perf_counter()
        _ = ATC_err_lbd(
            K, n, dist, pars, epsilon,
            B_bootstrap, list_q,
            list_lbd=0.3 * np.arange(20)
        )
        time_eps[j, i] = time.perf_counter() - t0

# average runtime across simulations for each n
time_mean = time_eps.mean(axis=1)

# save a 2-column file: n, mean_time_seconds
out = np.column_stack([n_list, time_mean])
np.savetxt('time_gmm_eps_mu0.2_n_varying.txt', out, fmt='%.6f', header='n mean_time_seconds', comments='')



# load the saved (n, mean_time_seconds)
data = np.loadtxt('time_gmm_eps_mu0.2_n_varying.txt', skiprows=1)
n_list = data[:, 0]
time_mean = data[:, 1]

plt.figure(figsize=(10, 10))
plt.plot(n_list, time_mean, marker='o', linewidth=2, markersize=8)

plt.title('Computing time vs. number of samples $n$', fontsize=22)
plt.xlabel('Number of samples $n$', fontsize=22)
plt.ylabel('Computing time (seconds)', fontsize=22)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# optional: if times vary a lot, uncomment the next line
# plt.yscale('log')

plt.grid(True, alpha=0.3)
plt.savefig('gmm_time_vs_eps_mu0.2_n_varying.png', bbox_inches='tight')
plt.show()


###############################
### Bernoulli-Gaussian model ##
###############################

######################## Err v.s. n ################################
warnings.filterwarnings('ignore')
# Setting parameters
dist = ['Bernoulli','Gaussian']
K = 2   # number of clusters
d = 10   # dimension 
p = 20 
tmp = np.random.normal(0, 1, size=d)

# delta = 0.1
# Theta = np.zeros((p,K))
# Theta[:, 0] = 0.5-delta
# Theta[:, 1] = 0.5+delta
alpha = np.array([6,10])
beta = np.array([14,10])
Theta = np.zeros((p,K))
Theta[:, 0] = np.random.beta(alpha[0], beta[0], size=p)
Theta[:, 1] = np.random.beta(alpha[1], beta[1], size=p)


# parameters for bootstrap 
B_bootstrap = 500
list_q = [0.8, 0.9, 0.95]

# parameters for simulation
nsim = 50
n_list = np.arange(100, 550, 50)

epsilon = 0.2 # similarity control
err_eps = np.zeros([len(n_list), len(list_q)+2, nsim])
# start simulations
for i in range(nsim):
    print('simulation iteration = {}\n'.format(i+1))
    for j in range(len(err_eps)):
        mean = np.sqrt(0.25*np.log(n_list[j]))*tmp/np.linalg.norm(tmp)
        pars = {'Theta': Theta, 'mean': mean, 'covariances': np.identity(d)}
        err_eps[j, :, i] = ATC_err_lbd(K, n_list[j], dist, pars, epsilon, B_bootstrap, list_q, list_lbd = 0.3 * np.arange(20))

err = err_eps.mean(axis = 2)
np.savetxt('err_bgm_eps_mu0.25_n_varying.txt', err, fmt='%f')


# In[38]:


###############################
### Contextual SBM model  ####
###############################

######################## Err v.s. n ################################

# Setting parameters
dist = ['Network','Gaussian']
K = 2   # number of clusters
d = 10   # dimension 
tmp = np.random.normal(0, 1, size=d)

# parameters for bootstrap 
B_bootstrap = 1000
list_q = [0.6, 0.7, 0.75]
seed = 1000

# parameters for simulation
nsim = 50
n_list = np.arange(200, 650, 50)
epsilon = 0.2 # similarity control
err_eps = np.zeros([len(n_list), len(list_q)+2, nsim])
# start simulations
for i in range(nsim):
    print('simulation iteration = {}\n'.format(i+1))
    for j in range(len(err_eps)):
        mean = np.sqrt(0.25*np.log(n_list[j]))*tmp/np.linalg.norm(tmp)
        pars = {'p': 0.4, 'q': 0.25, 'mean': mean, 'covariances': np.identity(d)}
        err_eps[j, :, i] = ATC_err_lbd(K, n_list[j], dist, pars, epsilon, B_bootstrap, list_q, list_lbd = 0.3 * np.arange(20))

err = err_eps.mean(axis = 2)
np.savetxt('err_csbm_eps_mu0.25_n_varying.txt', err, fmt='%f')


# In[39]:


err_eps


# In[8]:


###############################
### Gaussian Mixture Model ####
###############################

######################## Err v.s. n ################################

# parameters for generating models
dist = ['Gaussian','Gaussian']
d = 10   # dimension 
K = 2   # number of clusters
tmp = np.random.normal(0, 1, size=d)

# parameters for bootstrap 
B_bootstrap = 1000
list_q = [0.8, 0.9, 0.95]
seed = 1000

# parameters for simulation
nsim = 50
n_list = np.arange(400, 850, 50) # similarity control
epsilon = 0.2
err_eps = np.zeros([len(n_list), len(list_q)+2, nsim])
# start simulations
for i in range(nsim):
    print('simulation iteration = {}\n'.format(i+1))
    for j in range(len(err_eps)):
        mean = 0.2*np.log(n_list[j])*tmp/np.linalg.norm(tmp)
        pars = {'mean': mean, 'covariances': np.identity(d)}
        err_eps[j, :, i] = ATC_err_lbd(K, n_list[j], dist, pars, epsilon, B_bootstrap, list_q, list_lbd = 0.3 * np.arange(20))

err = err_eps.mean(axis = 2)
np.savetxt('err_gmm_eps_mu0.2_n_varying.txt', err, fmt='%f')


