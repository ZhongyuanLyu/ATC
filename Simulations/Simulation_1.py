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
        err_eps[j, :, i] = ATL_err_lbd(K, n, dist, pars, epsilon_list[j], B_bootstrap, list_q, list_lbd = 0.3 * np.arange(20))

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
pars = {'p': 0.4, 'q': 0.3, 'mean': mean, 'covariances': np.identity(d)}

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
        err_eps[j, :, i] = ATL_err_lbd(K, n, dist, pars, epsilon_list[j], B_bootstrap, list_q)

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
        err_eps[j, :, i] = ATL_err_lbd(K, n, dist, pars, epsilon_list[j], B_bootstrap, list_q)

err = err_eps.mean(axis = 2)
# np.savetxt('err_gmm_eps_mu0.2.txt', err, fmt='%f')



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




###############################
### Gaussian Mixture Model ####
###############################

######################## One iteration plot ################################
# Parameters
dist = ['Gaussian','Gaussian']
d = 10   # dimension 
K = 2   # number of clusters
n = 200  # number of samples
epsilon = 0.3 # similarity control
tmp = np.random.normal(0, 3, size=d)
mean = 0.3*np.log(n)*tmp/np.linalg.norm(tmp)
covariances = np.identity(d)
[X0, X1, Z0, Z1, parGMM] = generate_GMM_samples(K,n, mean, covariances, epsilon)
list_lbd = 0.3 * np.arange(20)

# performance of all candidates
[paras_hat, errs, AMIs] = TL_demo(X0, X1, Z0, K, dist, list_lbd)

seed = 1000
B_bootstrap = 1000
list_q = [0.8, 0.9, 0.95]
refit = False
# error estimation
start = time.time()
errs_hat = estimate_error(X0, X1, K = K, dist = dist, list_lbd = list_lbd, B_bootstrap = B_bootstrap, list_q = list_q, seed = seed, refit = refit, num_refinement = 10, matrix = 'adjacency', regularizer = 'null')
duration = time.time() - start
print('time elapsed: ', duration)
print('Oracle error: {}'.format(np.min(errs)))
# adaptive selection
print('Adaptive transfer clustering: ')
list_selected_idx = np.argmin(errs_hat, axis = 1)
for (i, q) in enumerate(list_q):
    print('q = {}: {}'.format(q, errs[list_selected_idx[i]]))
list_lbd_2 = 0.3 * np.arange(21)
fig = plt.figure(figsize=(6, 6))
plt.plot(list_lbd_2, errs, 'k')
plt.plot(list_lbd, errs_hat[0], 'r--', alpha = 0.7)
plt.plot(list_lbd, errs_hat[1], 'b--', alpha = 0.7)
plt.plot(list_lbd, errs_hat[2], 'c--', alpha = 0.7)

#plt.savefig('curves_{}.pdf'.format(target))

# plot AMIs
# print('Oracle AMI: {}'.format(np.max(AMIs)))
# plt.plot(list_lbd, AMIs, 'k')

# print('Adaptive transfer clustering: ')
# for (i, q) in enumerate(list_q):
#     print('q = {}: {}'.format(q, AMIs[list_selected_idx[i]]))





###############################
### Bernoulli-Gaussian model ##
###############################

######################## One iteration plot ################################

dist = ['Bernoulli','Gaussian']
n = 200  # Number of samples
p = 20   # dimension of Bernoulli
d = 10   # dimension of Gaussian
K = 2   # number of clusters
epsilon = 0.1 # similarity control
tmp = np.random.normal(0, 1, size=d)
mean = np.sqrt(0.25*np.log(n))*tmp/np.linalg.norm(tmp)
covariances = np.identity(d)
# delta = 0.08
# Theta = np.zeros((p,K))
# Theta[:, 0] = 0.2-delta
# Theta[:, 1] = 0.2+delta
alpha = np.array([6,10])
beta = np.array([14,10])
Theta = np.zeros((p,K))
Theta[:, 0] = np.random.beta(alpha[0], beta[0], size=p)
Theta[:, 1] = np.random.beta(alpha[1], beta[1], size=p)
[X0, X1, Z0, Z1, parBMM, parGMM] = generate_BGMM_samples(K, n, Theta, mean, covariances,  epsilon)
list_lbd = 0.2 * np.arange(20)

# performance of all candidates
[paras_hat, errs, AMIs] = TL_demo(X0, X1, Z0, K, dist, list_lbd)

# Adaptive clustering
B_bootstrap = 500
list_q = [0.8, 0.9, 0.95]
# Error estimation
start = time.time()
errs_hat = estimate_error(X0, X1, K = K, dist = dist, list_lbd = list_lbd, B_bootstrap = B_bootstrap, list_q = list_q, seed = 1000, refit = False, num_refinement = 10, matrix = 'adjacency', regularizer = 'null')
duration = time.time() - start
print('time elapsed: ', duration)
print('Oracle error: {}'.format(np.min(errs)))
# Adaptive selection
print('Adaptive transfer clustering: ')
list_selected_idx = np.argmin(errs_hat, axis = 1)
for (i, q) in enumerate(list_q):
    print('q = {}: {}'.format(q, errs[list_selected_idx[i]]))
fig = plt.figure(figsize=(6, 6))
plt.plot(list_lbd, errs, 'k')
plt.plot(list_lbd, errs_hat[0], 'r--', alpha = 0.7)
plt.plot(list_lbd, errs_hat[1], 'b--', alpha = 0.7)
plt.plot(list_lbd, errs_hat[2], 'c--', alpha = 0.7)







