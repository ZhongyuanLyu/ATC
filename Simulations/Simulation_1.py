#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import time
from algo_new import *
import warnings



###############################
### Bernoulli-Gaussian model ##
###############################

######################## Err v.s. epsilon ################################
warnings.filterwarnings('ignore')
# Setting parameters
dist = ['Bernoulli','Gaussian']
K = 2   # number of clusters
n = 200  # number of samples
d = 10   # dimension 
p = 20 
tmp = np.random.normal(0, 1, size=d)
mean = np.sqrt(0.25*np.log(n))*tmp/np.linalg.norm(tmp)
# delta = 0.1
# Theta = np.zeros((p,K))
# Theta[:, 0] = 0.5-delta
# Theta[:, 1] = 0.5+delta
alpha = np.array([6,10])
beta = np.array([14,10])
Theta = np.zeros((p,K))
Theta[:, 0] = np.random.beta(alpha[0], beta[0], size=p)
Theta[:, 1] = np.random.beta(alpha[1], beta[1], size=p)
pars = {'Theta': Theta, 'mean': mean, 'covariances': np.identity(d)}

# parameters for bootstrap 
B_bootstrap = 500
list_q = [0.8, 0.9, 0.95]

# parameters for simulation
nsim = 50
epsilon_list = np.arange(0.05, 0.3, 0.02) # similarity control
err_eps = np.zeros([len(epsilon_list), len(list_q)+2, nsim])
# start simulations
for i in range(nsim):
    print('simulation iteration = {}\n'.format(i+1))
    for j in range(len(err_eps)):
        err_eps[j, :, i] = ATC_err_lbd(K, n, dist, pars, epsilon_list[j], B_bootstrap, list_q, list_lbd = 0.3 * np.arange(20))

err = err_eps.mean(axis = 2)
# np.savetxt('err_bgm_eps_mu0.25.txt', err, fmt='%f')



###############################
### Contextual SBM model  ####
###############################

######################## Err v.s. epsilon ################################

# Setting parameters
dist = ['Network','Gaussian']
K = 2   # number of clusters
n = 300  # number of samples
d = 10   # dimension 
tmp = np.random.normal(0, 1, size=d)
mean = np.sqrt(0.25*np.log(n))*tmp/np.linalg.norm(tmp)
pars = {'p': 0.4, 'q': 0.25, 'mean': mean, 'covariances': np.identity(d)}

# parameters for bootstrap 
B_bootstrap = 500
list_q = [0.6, 0.7, 0.75]
seed = 1000

# parameters for simulation
nsim = 50
epsilon_list = np.arange(0.02, 0.35, 0.02) # similarity control
err_eps = np.zeros([len(epsilon_list), len(list_q)+2, nsim])
# start simulations
for i in range(nsim):
    print('simulation iteration = {}\n'.format(i+1))
    for j in range(len(err_eps)):
        err_eps[j, :, i] = ATC_err_lbd(K, n, dist, pars, epsilon_list[j], B_bootstrap, list_q)

err = err_eps.mean(axis = 2)
# np.savetxt('err_csbm_eps_mu0.25.txt', err, fmt='%f')



###############################
### Gaussian Mixture Model ####
###############################

######################## Err v.s. epsilon ################################

# parameters for generating models
dist = ['Gaussian','Gaussian']
d = 10   # dimension 
K = 2   # number of clusters
n = 500  # number of samples
tmp = np.random.normal(0, 1, size=d)
mean = 0.2*np.log(n)*tmp/np.linalg.norm(tmp)
pars = {'mean': mean, 'covariances': np.identity(d)}
# parameters for bootstrap 
B_bootstrap = 500
list_q = [0.8, 0.9, 0.95]
seed = 1000

# parameters for simulation
nsim = 50
epsilon_list = np.arange(0.02, 0.3, 0.02) # similarity control
err_eps = np.zeros([len(epsilon_list), len(list_q)+2, nsim])
# start simulations
for i in range(nsim):
    print('simulation iteration = {}\n'.format(i+1))
    for j in range(len(err_eps)):
        err_eps[j, :, i] = ATC_err_lbd(K, n, dist, pars, epsilon_list[j], B_bootstrap, list_q)

err = err_eps.mean(axis = 2)
# np.savetxt('err_gmm_eps_mu0.2.txt', err, fmt='%f')
