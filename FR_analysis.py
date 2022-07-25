#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 16:40:46 2021

@author: Sol
"""
from FourAFC_taskModels import StimHist_constant2
from psychrnn.backend.models.basic import Basic
import numpy as np
import pandas as pd
from statsmodels.stats.weightstats import ztest
import matplotlib.pyplot as plt
import behavior_analysis as ba

# Define model parameters and load weights
N_rec = 200
N_batch = 100
dt = 10
tau = 100
T = 1000
K = 20
rec_noise = 0.1
in_noise = 0.1
vis_noise = 0.0
mem_noise = 0.0
tParams_df = pd.read_csv('./tParams_df.csv')

task = StimHist_constant2(dt, tau, T, N_batch, K, dat1=tParams_df)
network_params = task.get_task_params()
network_params['name'] = 'choice_v4_2_correctctrl' # CHANGE MODEL NAME
network_params['N_rec'] = N_rec
network_params['load_weights_path'] = './saved_weights/trainChoice_v4_2_correctctrl.npz'
network_params['rec_noise'] = rec_noise
model = Basic(network_params)

# Test the model, produce as many test batches as desired for analysis
pd_test = StimHist_constant2(dt, tau, T, N_batch, K, dat1=tParams_df, in_noise=in_noise, vis_noise=vis_noise, mem_noise=mem_noise)

x_in1, target_output1, mask1, trial_params1 = pd_test.get_trial_batch()
model_output1, state_var1 = model.test(x_in1)

trial_params_large = trial_params1.copy()
model_output_large = model_output1.copy()
state_var_large = state_var1
batches = 100
for i in range(batches-1):
    print('test batch ', i+2)
    x_in, _, _, trial_params = pd_test.get_trial_batch()
    model_output, state_var = model.test(x_in)
    trial_params_large = np.concatenate((trial_params_large, trial_params), axis=0)
    model_output_large = np.concatenate((model_output_large, model_output), axis=0)
    state_var_large = np.concatenate((state_var_large, state_var), axis=0)

# Compute the average firing rate for each unit across trials and timesteps:
fr_large = np.maximum(state_var_large, 0)

avg_unit_fr = np.mean(fr_large.reshape((fr_large.shape[0]*fr_large.shape[1], N_rec)), axis=0)

plt.figure()
plt.hist(avg_unit_fr, bins=50)
plt.xlabel('Trial and time-averaged firing rates')
plt.ylabel('Number of units')
#plt.savefig('./Figures/')

avg_fr = np.mean(avg_unit_fr)
print('Average firing rate: ',avg_fr)

df = pd.DataFrame()
df['model'] = [network_params['name']]
df['avg_fr'] = [avg_fr]

df.to_csv('./FR_analysis.csv', mode='a', index=False, header=False)
