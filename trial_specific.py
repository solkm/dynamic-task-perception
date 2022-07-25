#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 16:30:06 2021

@author: Sol
"""

from FourAFC_taskModels import StimHist_constant3
from psychrnn.backend.models.basic import Basic
import numpy as np
import matplotlib.pyplot as plt
import model_functions as mf
import pandas as pd
import scipy.io

K=20

tParams_df = pd.read_csv('./tParams_df.csv')
N_tot = tParams_df.shape[0]
tParams_mat = scipy.io.loadmat('./Cheng_recTrainTest/tParams.mat')
t_ind = np.arange(N_tot)

# remove indices that overlap sessions
sess_ind = np.array(tParams_mat['r_indinit'][0], dtype=int)-1
N_s = len(sess_ind)
del_ind = np.empty((N_s)*(K-1), dtype=int)
for j in range(0, N_s):
    temp = np.arange(sess_ind[j], sess_ind[j]+K-1, dtype=int)
    del_ind[j*(K-1):(j+1)*(K-1)] = temp
    
t_ind2 = np.delete(t_ind, del_ind)

t_ind_m1 = t_ind2[t_ind2<91493] # monkey 1 only
t_ind_m2 = t_ind2[t_ind2>91493] # monkey 2 only

err_inds = tParams_df[tParams_df['err']==1].index.to_numpy()
ae_inds = err_inds[err_inds < N_tot-1] + 1

ae_inds_m1 = ae_inds[np.isin(ae_inds,t_ind_m1)]
ae_inds_m2 = ae_inds[np.isin(ae_inds,t_ind_m2)]

ac_inds_m1 = t_ind_m1[np.isin(t_ind_m1,ae_inds_m1,invert=True)]
ac_inds_m2 = t_ind_m2[np.isin(t_ind_m2,ae_inds_m2,invert=True)]

# Holdout test set. Using the same number of test and train trials for all models.
N_train = int(ae_inds_m2.shape[0]*0.9)
N_test = ae_inds_m2.shape[0] - N_train

np.random.seed(411)
np.random.shuffle(ae_inds_m1)
np.random.shuffle(ae_inds_m2)

ae_train_m1 = ae_inds_m1[:N_train]
ae_test_m1 = ae_inds_m1[-N_test:]
ae_train_m2 = ae_inds_m2[:N_train]
ae_test_m2 = ae_inds_m2[-N_test:]

np.random.shuffle(ac_inds_m1)
np.random.shuffle(ac_inds_m2)

ac_train_m1 = ac_inds_m1[:N_train]
ac_test_m1 = ac_inds_m1[-N_test:]
ac_train_m2 = ac_inds_m2[:N_train]
ac_test_m2 = ac_inds_m2[-N_test:]

#%% PRE-TRAINING on trials where the monkey makes a correct choice
'''
corr_inds_m1 = ac_inds_m1-1
corr_inds_m1_train = np.delete(corr_inds_m1, np.concatenate(ae_test_m1, ac_test_m1))
corr_inds_m1_test = corr_inds_m1[np.isin(corr_inds_m1,corr_inds_m1_train,invert=True)]
corr_inds_m2 = ac_inds_m2-1
corr_inds_m2_train = np.delete(corr_inds_m2, np.concatenate(ae_test_m2, ac_test_m2))
corr_inds_m2_test = corr_inds_m2[np.isin(corr_inds_m2,corr_inds_m2_train,invert=True)]

dt = 10
tau = 100
T = 1000
N_batch = 100
dat = tParams_df
N_rec = 200
rec_noise = 0.1
in_noise = 0.1

L2_in = 0.1
L2_rec = 0.01
L2_out = 0.01
L2_FR = 0.01

train_iters = 100000

# monkey 1
preTrainCorrect_m1 = StimHist_constant3(dt, tau, T, N_batch, K, dat, dat_inds=corr_inds_m1_train, in_noise=in_noise)
network_params = preTrainCorrect_m1.get_task_params()
network_params['name'] = 'pretraincorrectm1'
network_params['N_rec'] = 200
network_params['rec_noise'] = rec_noise
network_params['L2_in'] = L2_in
network_params['L2_rec'] = L2_rec
network_params['L2_out'] = L2_out
network_params['L2_firing_rate'] = L2_FR
train_params = {}
train_params['training_iters'] = train_iters
train_params['learning_rate'] = 0.003

pt_m1 = Basic(network_params)
losses, initialTime, trainTime = pt_m1.train(preTrainCorrect_m1, train_params)
print('initialTime:', initialTime, 'trainTime:',trainTime)
plt.figure() # plot loss
plt.plot(losses)
plt.title('Loss during training')
plt.ylabel('Minibatch loss')
plt.xlabel('Batch number')
plt.legend()
plt.savefig('./trial_specific_models/'+network_params['name']+'_loss')
pt_m1.save('./trial_specific_models/'+network_params['name'])

# monkey 2
preTrainCorrect_m2 = StimHist_constant3(dt, tau, T, N_batch, K, dat, dat_inds=corr_inds_m2_train, in_noise=in_noise)
network_params = preTrainCorrect_m1.get_task_params()
network_params['name'] = 'pretraincorrectm2'
network_params['N_rec'] = 200
network_params['rec_noise'] = rec_noise
network_params['L2_in'] = L2_in
network_params['L2_rec'] = L2_rec
network_params['L2_out'] = L2_out
network_params['L2_firing_rate'] = L2_FR
train_params = {}
train_params['training_iters'] = train_iters
train_params['learning_rate'] = 0.003

pt_m2 = Basic(network_params)
losses, initialTime, trainTime = pt_m2.train(preTrainCorrect_m2, train_params)
print('initialTime:', initialTime, 'trainTime:',trainTime)
plt.figure() # plot loss
plt.plot(losses)
plt.title('Loss during training')
plt.ylabel('Minibatch loss')
plt.xlabel('Batch number')
plt.legend()
plt.savefig('./trial_specific_models/'+network_params['name']+'_loss')
pt_m1.save('./trial_specific_models/'+network_params['name'])
'''
#%%

# ----- Set up models and train -----
dt = 10
tau = 100
T = 1000
N_batch = 100
dat = tParams_df
N_rec = 200
rec_noise = 0.1
in_noise = 0.1

L2_in = 0.1
L2_rec = 0.01
L2_out = 0.01
L2_FR = 0.01

train_iters = 300000

model_names = ['ae_m1','ae_m2','ac_m1','ac_m2']

#%% monkey 1 after error model

afterError_m1 = StimHist_constant3(dt, tau, T, N_batch, K, dat, dat_inds=ae_train_m1, in_noise=in_noise)
network_params = afterError_m1.get_task_params()
network_params['name'] = model_names[0]
network_params['N_rec'] = 200
network_params['rec_noise'] = rec_noise
network_params['L2_in'] = L2_in
network_params['L2_rec'] = L2_rec
network_params['L2_out'] = L2_out
network_params['L2_firing_rate'] = L2_FR

train_params = {}
train_params['training_iters'] = train_iters
train_params['learning_rate'] = 0.003

ae_m1 = Basic(network_params)
losses, initialTime, trainTime = ae_m1.train(afterError_m1, train_params) # train model to perform pd task

print('initialTime:', initialTime, 'trainTime:',trainTime)
plt.figure() # plot loss
plt.plot(losses)
plt.title('Loss during training')
plt.ylabel('Minibatch loss')
plt.xlabel('Batch number')
plt.legend()
plt.savefig('./trial_specific_models/'+network_params['name']+'_loss')

ae_m1.save('./trial_specific_models/'+network_params['name'])
#%% test after error m1
test = StimHist_constant3(dt, tau, T, N_test, K, dat, in_noise=in_noise)
network_params = test.get_task_params()
network_params['name'] = 'test_'+model_names[0]
network_params['N_rec'] = 200
network_params['rec_noise'] = rec_noise
network_params['load_weights_path'] = './trial_specific_models/'+model_names[0]+'.npz'

ae_m1_testmodel = Basic(network_params)
ae_m1_test = StimHist_constant3(dt, tau, T, N_test, K, dat, dat_inds=ae_test_m1, in_noise=in_noise, testall=True)
test_inputs, target_output, mask, trial_params = ae_m1_test.get_trial_batch()
model_output, state_var = ae_m1_testmodel.test(test_inputs)

accuracy = mf.get_modelAccuracy(model_output,target_output)
print(network_params['name']+' accuracy: ', accuracy)
#%% monkey 2 after error model

afterError_m2 = StimHist_constant3(dt, tau, T, N_batch, K, dat, dat_inds=ae_train_m2, in_noise=in_noise)
network_params = afterError_m2.get_task_params()
network_params['name'] = model_names[1]
network_params['N_rec'] = 200
network_params['rec_noise'] = rec_noise
network_params['L2_in'] = L2_in
network_params['L2_rec'] = L2_rec
network_params['L2_out'] = L2_out
network_params['L2_firing_rate'] = L2_FR
train_params = {}
train_params['training_iters'] = train_iters
train_params['learning_rate'] = 0.003

ae_m2 = Basic(network_params)
losses, initialTime, trainTime = ae_m2.train(afterError_m2, train_params) # train model to perform pd task

print('initialTime:', initialTime, 'trainTime:',trainTime)
plt.figure() # plot loss
plt.plot(losses)
plt.title('Loss during training')
plt.ylabel('Minibatch loss')
plt.xlabel('Batch number')
plt.legend()
plt.savefig('./trial_specific_models/'+network_params['name']+'_loss')

ae_m2.save('./trial_specific_models/'+network_params['name'])
#%% test after error m2
network_params = test.get_task_params()
network_params['name'] = 'test_'+model_names[1]
network_params['N_rec'] = 200
network_params['rec_noise'] = rec_noise
network_params['load_weights_path'] = './trial_specific_models/'+model_names[1]+'.npz'

ae_m2_testmodel = Basic(network_params)
ae_m2_test = StimHist_constant3(dt, tau, T, N_test, K, dat, dat_inds=ae_test_m2, in_noise=in_noise, testall=True)
test_inputs, target_output, mask, trial_params = ae_m2_test.get_trial_batch()
model_output, state_var = ae_m2_testmodel.test(test_inputs)

accuracy = mf.get_modelAccuracy(model_output,target_output)
print(network_params['name']+' accuracy: ', accuracy)
#%% monkey 1 after correct model

afterCorrect_m1 = StimHist_constant3(dt, tau, T, N_batch, K, dat, dat_inds=ac_train_m1, in_noise=in_noise)
network_params = afterCorrect_m1.get_task_params()
network_params['name'] = model_names[2]
network_params['N_rec'] = 200
network_params['rec_noise'] = rec_noise
network_params['L2_in'] = L2_in
network_params['L2_rec'] = L2_rec
network_params['L2_out'] = L2_out
network_params['L2_firing_rate'] = L2_FR
train_params = {}
train_params['training_iters'] = train_iters
train_params['learning_rate'] = 0.003

ac_m1 = Basic(network_params)
losses, initialTime, trainTime = ac_m1.train(afterCorrect_m1, train_params) # train model to perform pd task

print('initialTime:', initialTime, 'trainTime:',trainTime)
plt.figure() # plot loss
plt.plot(losses)
plt.title('Loss during training')
plt.ylabel('Minibatch loss')
plt.xlabel('Batch number')
plt.legend()
plt.savefig('./trial_specific_models/'+network_params['name']+'_loss')

ac_m1.save('./trial_specific_models/'+network_params['name'])
#%% test after correct m1
network_params = test.get_task_params()
network_params['name'] = 'test_'+model_names[2]
network_params['N_rec'] = 200
network_params['rec_noise'] = rec_noise
network_params['load_weights_path'] = './trial_specific_models/'+model_names[2]+'.npz'

ac_m1_testmodel = Basic(network_params)
ac_m1_test = StimHist_constant3(dt, tau, T, N_test, K, dat, dat_inds=ac_test_m1, in_noise=in_noise, testall=True)
test_inputs, target_output, mask, trial_params = ac_m1_test.get_trial_batch()
model_output, state_var = ac_m1_testmodel.test(test_inputs)

accuracy = mf.get_modelAccuracy(model_output,target_output)
print(network_params['name']+' accuracy: ', accuracy)
#%% monkey 2 after correct model

afterCorrect_m2 = StimHist_constant3(dt, tau, T, N_batch, K, dat, dat_inds=ac_train_m2, in_noise=in_noise)
network_params = afterCorrect_m2.get_task_params()
network_params['name'] = model_names[3]
network_params['N_rec'] = 200
network_params['rec_noise'] = rec_noise
network_params['L2_in'] = L2_in
network_params['L2_rec'] = L2_rec
network_params['L2_out'] = L2_out
network_params['L2_firing_rate'] = L2_FR
train_params = {}
train_params['training_iters'] = train_iters
train_params['learning_rate'] = 0.003

ac_m2 = Basic(network_params)
losses, initialTime, trainTime = ac_m2.train(afterCorrect_m2, train_params) # train model to perform pd task

print('initialTime:', initialTime, 'trainTime:',trainTime)
plt.figure() # plot loss
plt.plot(losses)
plt.title('Loss during training')
plt.ylabel('Minibatch loss')
plt.xlabel('Batch number')
plt.legend()
plt.savefig('./trial_specific_models/'+network_params['name']+'_loss')

ac_m2.save('./trial_specific_models/'+network_params['name'])
#%% test after correct m2
test = StimHist_constant3(dt, tau, T, N_test, K, dat, dat_inds=ac_test_m2, in_noise=in_noise)
network_params = test.get_task_params()
network_params['name'] = 'test_'+model_names[3]
network_params['N_rec'] = 200
network_params['rec_noise'] = rec_noise
network_params['load_weights_path'] = './trial_specific_models/'+model_names[3]+'.npz'

ac_m2_testmodel = Basic(network_params)
ac_m2_test = StimHist_constant3(dt, tau, T, N_test, K, dat, dat_inds=ac_test_m2, in_noise=in_noise, testall=True)
test_inputs, target_output, mask, trial_params = ac_m2_test.get_trial_batch()
model_output, state_var = ac_m2_testmodel.test(test_inputs)

accuracy = mf.get_modelAccuracy(model_output,target_output)
print(network_params['name']+' accuracy: ', accuracy)


