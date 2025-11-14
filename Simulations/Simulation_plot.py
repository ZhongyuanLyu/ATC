#!/usr/bin/env python
# coding: utf-8




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import time
from algo_new import *
from stepmix.stepmix import StepMix


err = np.loadtxt('err_lcm_eps_theta0.5_new.txt')[0:13]
epsilon_list = np.arange(0, 0.501, 0.025)[0:13] # similarity control
# Plotting the data
plt.figure(figsize=(10, 10))

# Define labels for methods
labels = ['ATC(0.8)', 'ATC(0.9)', 'ATC(0.95)', 'ATC(0.99)', 'ITL', 'DP']
# labels = ['Method 0.9', 'Independent Task Learning', 'Data Pooling']

# Plot each method with different colors
colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#8c564b']
# colors = ['#00441b', '#006d2c', '#238b45', '#41ae76', '#66c2a4', '#99d8c9']   
# colors = ['b', 'g', 'r', 'c', 'm', 'y']
markers = ['o', 's', 'D', '^', 'v', '*']
for i in range(err.shape[1]):
    plt.plot(epsilon_list, err[:, i], label=labels[i], color=colors[i], marker=markers[i], markersize = 8)


# Adding titles and labels
plt.title('Clustering error v.s. discrepancy $\\epsilon$', fontsize=22)
plt.xlabel('Similarity $\\epsilon$', fontsize=22)
plt.ylabel('Clustering error', fontsize=22)

# Adding legend
plt.legend(fontsize=20)

# Adjusting tick parameters
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Show plot
# plt.grid(True)
plt.savefig('lcm_err_vs_eps_n300_theta0.5_new.png', bbox_inches='tight')
plt.show()


err = np.loadtxt('err_bgm_eps_mu0.25_new.txt')[0:13]
epsilon_list = np.arange(0, 0.501, 0.025)[0:13] # similarity control
# Plotting the data
plt.figure(figsize=(10, 10))

# Define labels for methods
labels = ['ATC(0.8)', 'ATC(0.9)', 'ATC(0.95)', 'ATC(0.99)', 'ITL', 'DP']
# labels = ['Method 0.9', 'Independent Task Learning', 'Data Pooling']

# Plot each method with different colors
# colors = ['b', 'g', 'r']
# colors = ['b', 'g', 'r', 'c', 'm', 'y']
colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#8c564b']
markers = ['o', 's', 'D', '^', 'v', '*']
for i in range(err.shape[1]):
    plt.plot(epsilon_list, err[:, i], label=labels[i], color=colors[i], marker=markers[i], markersize = 8)


# Adding titles and labels
plt.title('Clustering error v.s. discrepancy $\\epsilon$', fontsize=22)
plt.xlabel('Similarity $\\epsilon$', fontsize=22)
plt.ylabel('Clustering error', fontsize=22)

# Adding legend
plt.legend(fontsize=20)

# Adjusting tick parameters
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Show plot
# plt.grid(True)
plt.savefig('bgm_err_vs_eps_n200_mu0.25_new.png', bbox_inches='tight')
plt.show()




err = np.loadtxt("err_csbm_eps_mu0.25_new.txt")[0:13]
epsilon_list = np.arange(0, 0.501, 0.025)[0:13]
# Plotting the data
plt.figure(figsize=(10, 10))

# Define labels for methods
labels = ['ATC(0.8)', 'ATC(0.9)', 'ATC(0.95)', 'ATC(0.99)', 'ITL', 'DP']
# labels = ['Method 0.9', 'Independent Task Learning', 'Data Pooling']

# Plot each method with different colors
# colors = ['b', 'g', 'r']
colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#8c564b']
markers = ['o', 's', 'D', '^', 'v', '*']
for i in range(err.shape[1]):
    plt.plot(epsilon_list, err[:, i], label=labels[i], color=colors[i], marker=markers[i])


# Adding titles and labels
plt.title('Clustering error v.s. discrepancy $\\epsilon$', fontsize=22)
plt.xlabel('Similarity $\\epsilon$', fontsize=22)
plt.ylabel('Clustering error', fontsize=22)

# Adding legend
plt.legend(fontsize=20)

# Adjusting tick parameters
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Show plot
# plt.grid(True)
plt.savefig('csbm_err_vs_eps_n500_mu0.25_new.png', bbox_inches='tight')
plt.show()




err = np.loadtxt("err_csbm_targetgmm_eps_mu0.23_newnew.txt")[0:13]
epsilon_list = np.arange(0, 0.501, 0.025)[0:13]
# Plotting the data
plt.figure(figsize=(10, 10))

# Define labels for methods
labels = ['ATC(0.8)', 'ATC(0.9)', 'ATC(0.95)', 'ATC(0.99)', 'ITL', 'DP']
# labels = ['Method 0.9', 'Independent Task Learning', 'Data Pooling']

# Plot each method with different colors
# colors = ['b', 'g', 'r']
colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#8c564b']
markers = ['o', 's', 'D', '^', 'v', '*']
for i in range(err.shape[1]):
    plt.plot(epsilon_list, err[:, i], label=labels[i], color=colors[i], marker=markers[i], markersize = 8)


# Adding titles and labels
plt.title('Clustering error v.s. discrepancy $\\epsilon$', fontsize=22)
plt.xlabel('Similarity $\\epsilon$', fontsize=22)
plt.ylabel('Clustering error', fontsize=22)

# Adding legend
plt.legend(fontsize=20)

# Adjusting tick parameters
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Show plot
# plt.grid(True)
plt.savefig('csbm_targetgmm_err_vs_eps_n500_mu0.23_new.png', bbox_inches='tight')
plt.show()




err = np.loadtxt("err_gmm_eps_mu0.2_new.txt")[0:13]
epsilon_list = np.arange(0, 0.501, 0.025)[0:13]
# Plotting the data
plt.figure(figsize=(10, 10))

# Define labels for methods
labels = ['ATC(0.8)', 'ATC(0.9)', 'ATC(0.95)', 'ATC(0.99)', 'ITL', 'DP']
# labels = ['Method 0.9', 'Independent Task Learning', 'Data Pooling']

# Plot each method with different colors
# colors = ['b', 'g', 'r']
colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#8c564b']
markers = ['o', 's', 'D', '^', 'v', '*']
for i in range(err.shape[1]):
    plt.plot(epsilon_list, err[:, i], label=labels[i], color=colors[i], marker=markers[i], markersize = 8)


# Adding titles and labels
plt.title('Clustering error v.s. discrepancy $\\epsilon$', fontsize=22)
plt.xlabel('Similarity $\\epsilon$', fontsize=22)
plt.ylabel('Clustering error', fontsize=22)

# Adding legend
plt.legend(fontsize=20)

# Adjusting tick parameters
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Show plot
# plt.grid(True)
plt.savefig('gmm_err_vs_eps_n500_mu0.108_new.png', bbox_inches='tight')
plt.show()







