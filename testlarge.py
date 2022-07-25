#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 20:47:12 2021

@author: Sol
"""

import numpy as np
import pandas as pd
from FourAFC_taskModels import StimHist_constant2
from psychrnn.backend.models.basic import Basic
import scipy.io
import behavior_analysis as ba


N_rec = 100
dt = 10
tau = 100
T = 1000
K = 10
in_noise = 1.0
rec_noise = 0.5
vis_noise = 0.9
mem_noise = 0.3

tParams_df = pd.read_csv('./tParams_df.csv')
N_tot = tParams_df.shape[0]
tParams_mat = scipy.io.loadmat('./Cheng_recTrainTest/tParams.mat')
t_ind = np.arange(N_tot)
# remove indices that overlap sessions
sess_ind = np.array(tParams_mat['r_indinit'][0], dtype=int)-1
N_s = len(sess_ind)
del_ind = np.empty((N_s-1)*(K-1), dtype=int)
for j in range(0, N_s-1):
    temp = np.arange(sess_ind[j+1]-K+1, sess_ind[j+1], dtype=int)
    del_ind[j*(K-1):(j+1)*(K-1)] = temp
    
t_ind2 = np.delete(t_ind, del_ind)
t_ind2 = t_ind2[t_ind2<N_tot-K+1]
np.random.seed(1234)
N_batch = 20000
test_inds = np.random.choice(t_ind2, N_batch, replace=False)

s = './saved_weights/'
weights_paths = [s+'correctHighNoise.npz']
model_names = ['correctHighNoise']

for i in range(len(weights_paths)):
    
    weights_path = weights_paths[i]
    model_name = model_names[i]

    task1 = StimHist_constant2(dt, tau, T, N_batch, K, dat1=tParams_df, dat1_inds=test_inds, testall=True, \
                               in_noise=in_noise, vis_noise=vis_noise, mem_noise=mem_noise)
    network_params1 = task1.get_task_params()
    network_params1['name'] = model_name
    network_params1['N_rec'] = N_rec
    network_params1['load_weights_path'] = weights_path
    network_params1['rec_noise'] = rec_noise
    model1 = Basic(network_params1)

    x1, target_output1, _, trial_params1 = task1.get_trial_batch()
    model_output1, state_var1 = model1.test(x1)
    '''
    sp, spErr, spErrAr = ba.exploreSwitch(trial_params1, model_output1)
    print(model_name, 'sp:', sp, 'spErr:', spErr, 'spErrAr', spErrAr, 'spErr/sp:', spErr/sp)
    
    df_a = pd.DataFrame([[model_name,sp,spErr,spErrAr,spErr/sp]],columns=['model','sp','spErr','spErrAr','spErr/sp'])
    df_a.to_csv('./exploreSwitch.csv', mode='a', index=False, header=False)
    '''
    dsl_perf_aR, dsl_perf_aNR, dsf_perf_aR, dsf_perf_aNR, p_dsl, p_dsf = ba.percPerfSameStim(trial_params1, model_output1)
    
    df_b = pd.DataFrame([[model_name, N_batch, np.mean(dsl_perf_aR), np.mean(dsl_perf_aNR), np.mean(dsf_perf_aR), np.mean(dsf_perf_aNR), p_dsl, p_dsf]], \
                        columns=['model','N_test','dsl_paR', 'dsl_paNR','dsf_paR','dsf_paNR','p_dsl','p_dsf'])
    df_b.to_csv('./percPerfSameStim.csv', index=False, header=False, mode='a')