#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 10:42:17 2021

@author: Sol
"""

from FourAFC_taskModels import StimHist_constant2
from psychrnn.backend.models.basic import Basic
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import model_functions as mf
import visfunctions as vf

N_rec = 100
N_batch = 100
dt = 10
tau = 100
T = 1000
K = 10
vis_noise = 0.9
mem_noise = 0.3
rec_noise = 0.5
in_noise= 1.0
tParams = scipy.io.loadmat('./mat_files/tParams.mat')

# Load the model

pd1 = StimHist_constant2(dt, tau, T, N_batch, K, dat1=tParams)
network_params1 = pd1.get_task_params() # get the params passed in and defined in the task
network_params1['name'] = 'hn' # name the model uniquely if running mult models in unison
network_params1['N_rec'] = N_rec # set the number of recurrent units in the model
network_params1['load_weights_path'] = './saved_weights/correctHighNoise.npz'
network_params1['rec_noise'] = rec_noise
model1 = Basic(network_params1)

nmax = 7
iters = 5
model_taskaccuracies = np.zeros((iters, nmax))
model_accuracies = np.zeros((iters, nmax))
model_switchprobs = np.zeros((iters, nmax))
monkey_taskaccuracies = np.zeros((iters, nmax))
monkey_accuracies = np.zeros((iters, nmax))
monkey_switchprobs = np.zeros((iters, nmax))

testdat = []
for n in range(1, nmax+1):
    testdat.append(mf.gendat_undetectedSwitch_nBefore(tParams, n, K, cond_needSwitch=True))

for i in range(iters):
    print('iteration ', i+1)
    for n in range(1, nmax+1):
        
        td = testdat[n-1]
        
        pd_test = StimHist_constant2(dt, tau, T, N_batch, K, dat1=td, in_noise=in_noise, vis_noise=vis_noise, mem_noise=mem_noise, gendat=True)
        
        x_in, target_output, _, trial_params = pd_test.get_trial_batch()
        model_output, _ = model1.test(x_in)
        
        model_accuracies[i][n-1] = mf.get_modelAccuracy(model_output, target_output)
        model_switchprobs[i][n-1] = mf.modelSwitchPercentage(model_output, trial_params)
        monkey_accuracies[i][n-1] = mf.get_monkeyAccuracy(trial_params)
        monkey_switchprobs[i][n-1] = mf.monkeySwitchPercentage(trial_params)

avg_accuracies = np.mean(model_accuracies, axis=0)
avg_switchpercentage = np.mean(model_switchprobs, axis=0)
std_accuracies = np.std(model_accuracies, axis=0)
std_switchpercentage = np.std(model_switchprobs, axis=0)
avg_accuracies2 = np.mean(monkey_accuracies, axis=0)
avg_switchpercentage2 = np.mean(monkey_switchprobs, axis=0)
std_accuracies2 = np.std(monkey_accuracies, axis=0)
std_switchpercentage2 = np.std(monkey_switchprobs, axis=0)

# MODEL VS MONKEY side by side with labels

x = np.arange(nmax)
width = 0.4

# Plot switch percentages
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'

fig, ax = plt.subplots()
ax.bar(x-width/2-.01, avg_switchpercentage, yerr=std_switchpercentage, width=width, label='Model', color='tab:blue', ecolor='darkgrey')
ax.bar(x+width/2+.01, avg_switchpercentage2, yerr=std_switchpercentage2, width=width, label='Monkey', color='firebrick', ecolor='darkgrey')
ax.set_xticks(x)
ax.set_xticklabels(['1', '2', '3', '4', '5', '6', '7'], fontsize=18)
ax.set_title('Conditional switch probabilities', fontsize=18)
ax.set_ylabel('P(switch | wrong task since switch)', fontsize=18)
ax.set_xlabel('Number of trials after task switch', fontsize=18)
ax.tick_params(axis='y', labelsize=12)
vf.add_value_labels(ax, fontsize=16)
plt.legend(loc = 'upper left', fontsize=14)
fig.tight_layout()
plt.savefig('./Figures/trainCorrect/highnoise_condSwitchProbs', bbox_inches='tight')

'''
# Plot overall accuracies
fig, ax = plt.subplots()
ax.bar(x-width/2-.01, avg_accuracies, yerr=std_accuracies, width=width, label='Model', color='tab:blue', ecolor='darkgrey')
ax.bar(x+width/2+.01, avg_accuracies2, yerr=std_accuracies2, width=width, label='Monkey', color='firebrick', ecolor='darkgrey')
ax.set_xticks(x)
ax.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8'])
ax.set_title('Conditional accuracies')
ax.set_ylabel('P(correct choice | monkey has not yet switched)')
ax.set_xlabel('Number of trials after task switch')
vf.add_value_labels(ax)
plt.legend()
#plt.savefig('./Figures/')
'''