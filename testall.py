#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 19:27:23 2021

@author: Sol
"""
from FourAFC_taskModels import StimHist_constant2
from psychrnn.backend.models.basic import Basic
import numpy as np
import pandas as pd
from scipy.io import loadmat
import pickle
#%%
# Define model parameters and load weights
N_rec = 100
dt = 10
tau = 100
T = 1000
K = 10
rec_noise = 0.2
in_noise = 0.5
vis_noise = 0.8
mem_noise = 0.3

tParams_df = pd.read_csv('./tParams_df.csv')
N_tot = tParams_df.shape[0]
tParams_mat = loadmat('./Cheng_recTrainTest/tParams.mat')
t_ind = np.arange(N_tot)
# remove indices that overlap sessions
sess_ind = np.array(tParams_mat['r_indinit'][0], dtype=int)-1
N_s = len(sess_ind)
del_ind = np.empty((N_s-1)*(K-1), dtype=int)
for j in range(0, N_s-1):
    temp = np.arange(sess_ind[j+1]-K+1, sess_ind[j+1], dtype=int)
    del_ind[j*(K-1):(j+1)*(K-1)] = temp
t_ind2 = np.delete(t_ind, del_ind)

dat1_inds_m1 = t_ind2[t_ind2<91493] # monkey 1 only
dat1_inds_m2 = t_ind2[(91493<=t_ind2) & (t_ind2<N_tot-K+1)] # monkey 2 only

for dat1_inds, savename in [(dat1_inds_m1, './pickled_testdata/correctFullConn2_testm1'),(dat1_inds_m2, './pickled_testdata/correctFullConn2_testm2')]: #change savenames

    testall = True
    N_batch = dat1_inds.shape[0]
    task = StimHist_constant2(dt, tau, T, N_batch, K, dat1=tParams_df)
    network_params = task.get_task_params()
    network_params['name'] = 'fullconn2'
    network_params['N_rec'] = N_rec
    network_params['load_weights_path'] = './saved_weights/correctFullConn2.npz'
    network_params['rec_noise'] = rec_noise
    model = Basic(network_params)
    
    # Test the model
    task_test = StimHist_constant2(dt, tau, T, N_batch, K, dat1=tParams_df, dat1_inds=dat1_inds, in_noise=in_noise, vis_noise=vis_noise, mem_noise=mem_noise, testall=testall)
    
    x_in1, target_output1, mask1, trial_params1 = task_test.get_trial_batch()
    model_output1, state_var1 = model.test(x_in1)
    
    # save the test data
    if savename is not None:
        savefile = open(savename+'_modeloutput.pickle','wb')
        pickle.dump(model_output1, savefile, protocol=4)
        savefile.close()

        savefile = open(savename+'_trialparams.pickle','wb')
        pickle.dump(trial_params1, savefile, protocol=4)
        savefile.close()
    
    model.destruct()