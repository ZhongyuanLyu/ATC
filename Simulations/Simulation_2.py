#!/usr/bin/env python
# coding: utf-8



import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import time
from algo_new import *
import warnings




###############################
### Bernoulli-Bernoulli model ##
###############################

######################## Err v.s. epsilon ################################
warnings.filterwarnings('ignore')
# Setting parameters
dist = ['Bernoulli','Bernoulli']
K = 2   # number of clusters
n = 300  # number of samples
p = 15 

######################## Fixed design ################################
delta = 0.1
Theta = np.zeros((p,K))
Theta[:, 0] = 0.5-delta
Theta[:, 1] = 0.5+delta

######################## Random design ################################
# alpha = np.array([6,10])
# beta = np.array([14,10])
# Theta = np.zeros((p,K))
# Theta[:, 0] = np.random.beta(alpha[0], beta[0], size=p)
# Theta[:, 1] = np.random.beta(alpha[1], beta[1], size=p)


pars = {'Theta': Theta}

# parameters for bootstrap 
B_bootstrap = 1000
list_q = [0.8, 0.9, 0.95, 0.99]
list_lbd = np.arange(0, 20, 0.3)
# parameters for simulation
nsim = 50
epsilon_list = np.arange(0, 0.501, 0.025) # similarity control
err_eps = np.zeros([len(epsilon_list), len(list_q)+2, nsim])
# start simulations
for i in range(nsim):
    print('simulation iteration = {}\n'.format(i+1))
    for j in range(len(err_eps)):
        err_eps[j, :, i] = ATC_err_lbd(K, n, dist, pars, epsilon_list[j], B_bootstrap, list_q, list_lbd)

err = err_eps.mean(axis = 2)
# np.savetxt('err_lcm_eps_theta0.5_new.txt', err, fmt='%f')
print(err)





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

######################## Fixed design ################################
delta = 0.1
Theta = np.zeros((p,K))
Theta[:, 0] = 0.5-delta
Theta[:, 1] = 0.5+delta

######################## Random design ################################
# alpha = np.array([6,10])
# beta = np.array([14,10])
# Theta = np.zeros((p,K))
# Theta[:, 0] = np.random.beta(alpha[0], beta[0], size=p)
# Theta[:, 1] = np.random.beta(alpha[1], beta[1], size=p)


pars = {'Theta': Theta, 'mean': mean, 'covariances': np.identity(d)}

# parameters for bootstrap 
B_bootstrap = 500
list_q = [0.8, 0.9, 0.95, 0.99]
list_lbd = np.arange(0, 20, 0.3)
# parameters for simulation
nsim = 50
epsilon_list = np.arange(0, 0.501, 0.025) # similarity control
err_eps = np.zeros([len(epsilon_list), len(list_q)+2, nsim])
# start simulations
for i in range(nsim):
    print('simulation iteration = {}\n'.format(i+1))
    for j in range(len(err_eps)):
        err_eps[j, :, i] = ATC_err_lbd(K, n, dist, pars, epsilon_list[j], B_bootstrap, list_q, list_lbd)

err = err_eps.mean(axis = 2)
print(err)
# np.savetxt('err_bgm_eps_mu0.25_new.txt', err, fmt='%f')


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
list_q = [0.8, 0.9, 0.95, 0.99]
list_lbd = np.arange(0, 20, 0.3)

# parameters for simulation
nsim = 50
epsilon_list = np.arange(0, 0.501, 0.025) # similarity control
err_eps = np.zeros([len(epsilon_list), len(list_q)+2, nsim])
# start simulations
for i in range(nsim):
    print('simulation iteration = {}\n'.format(i+1))
    for j in range(len(err_eps)):
        err_eps[j, :, i] = ATC_err_lbd(K, n, dist, pars, epsilon_list[j], B_bootstrap, list_q, list_lbd, sparse_method = False)

err = err_eps.mean(axis = 2)
print(err)
# np.savetxt('err_csbm_eps_mu0.25_new.txt', err, fmt='%f')



#########################################
### Contextual SBM model: Target GMM ####
#########################################

######################## Err v.s. epsilon ################################

# Setting parameters
dist = ['Gaussian','Network']
K = 2   # number of clusters
n = 300  # number of samples
d = 10   # dimension 
tmp = np.random.normal(0, 1, size=d)
mean = np.sqrt(0.23*np.log(n))*tmp/np.linalg.norm(tmp)
pars = {'p': 0.5, 'q': 0.3, 'mean': mean, 'covariances': np.identity(d)}

# parameters for bootstrap 
B_bootstrap = 500
list_q = [0.8, 0.9, 0.95, 0.99]
list_lbd = np.arange(0, 20, 0.3)

# parameters for simulation
nsim = 50
epsilon_list = np.arange(0, 0.501, 0.025) # similarity control
err_eps = np.zeros([len(epsilon_list), len(list_q)+2, nsim])
# start simulations
for i in range(nsim):
    print('simulation iteration = {}\n'.format(i+1))
    for j in range(len(err_eps)):
        err_eps[j, :, i] = ATC_err_lbd(K, n, dist, pars, epsilon_list[j], B_bootstrap, list_q, list_lbd, sparse_method = False)

err = err_eps.mean(axis = 2)
print(err)
# np.savetxt('err_csbm_targetgmm_eps_mu0.2_new.txt', err, fmt='%f')




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
list_q = [0.8, 0.9, 0.95, 0.99]
list_lbd = np.arange(0, 20, 0.3)
seed = 1000

# parameters for simulation
nsim = 50
epsilon_list = np.arange(0, 0.501, 0.025) # similarity control
err_eps = np.zeros([len(epsilon_list), len(list_q)+2, nsim])
# start simulations
for i in range(nsim):
    print('simulation iteration = {}\n'.format(i+1))
    for j in range(len(err_eps)):
        err_eps[j, :, i] = ATC_err_lbd(K, n, dist, pars, epsilon_list[j], B_bootstrap, list_q, list_lbd)

err = err_eps.mean(axis = 2)
print(err)
# np.savetxt('err_gmm_eps_mu0.2_new.txt', err, fmt='%f')



###############################
### Gaussian Mixture Model ####
###############################

######################## Err v.s. epsilon ################################

# parameters for generating models
dist = ['Gaussian','Gaussian']
d = 10   # dimension 
K = 2   # number of clusters
n = 500  # number of samples
epsilon = 0.2 # similarity control
covariances = np.identity(d)

# parameters for bootstrap 
B_bootstrap = 500
list_q = [0.8, 0.9, 0.95]
seed = 1000

# parameters for simulation
nsim = 50
signal = np.arange(0.1, 0.4, 0.02)
tmp = np.random.normal(0, 3, size=d)
tmp /= np.linalg.norm(tmp)
mean_mat = np.outer(signal*np.log(n), tmp)
err_signal = np.zeros([len(signal), len(list_q)+2, nsim])
# start simulations
for i in range(nsim):
    print('simulation iteration = {}\n'.format(i+1))
    for j in range(len(signal)):
        err_signal[j, :, i] = one_sim(K, n, dist, mean_mat[j,:], covariances, epsilon, B_bootstrap, list_q)

err = err_signal.mean(axis = 2)
# np.savetxt('err_gmm_mu_eps0.2.txt', err, fmt='%f')



