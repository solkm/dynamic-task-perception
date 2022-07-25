#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 00:18:42 2021

@author: Sol
"""

from FourAFC_taskModels import StimHist_constant2
from psychrnn.backend.models.basic import Basic
import numpy as np
import visfunctions as vf
import matplotlib.pyplot as plt
import model_functions as mf
import pandas as pd
from scipy.io import loadmat

N_rec = 100
N_batch = 200
dt = 10
tau = 100
T = 1000
K = 10
rec_noise = 0.5 # 0.1; 0.2; 0.5
in_noise = 1.0 # 0.2; 0.5; 1.0
vis_noise = 0.9 # 0.8; 0.8; 0.9
mem_noise = 0.3

model_name = 'correctHighNoise2'

# Train data

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

t_ind = np.delete(t_ind, del_ind)
t_ind = t_ind[t_ind<N_tot-K+1]

np.random.seed(413)
N_test = 2000
np.random.shuffle(t_ind)
test_inds = t_ind[:N_test] # test set
train_inds = t_ind[N_test:]

# network params

task = StimHist_constant2(dt, tau, T, N_batch, K, dat1=tParams_df, dat1_inds=train_inds, \
                          in_noise=in_noise, vis_noise=vis_noise, mem_noise=mem_noise)
network_params = task.get_task_params()
network_params['name'] = model_name
network_params['N_rec'] = N_rec
network_params['rec_noise'] = rec_noise
network_params['autapses'] = False
network_params['dale_ratio'] = None
modIn = True

N_in = task.N_in
N_out = task.N_out
input_connectivity = np.ones((N_rec, N_in))
rec_connectivity = np.ones((N_rec, N_rec))
output_connectivity = np.ones((N_out, N_rec))

if modIn==True:
    input_connectivity[:N_rec//2, :-2] = 0
    input_connectivity[N_rec//2:, -2:] = 0
    
network_params['input_connectivity'] = input_connectivity
network_params['rec_connectivity'] = rec_connectivity
network_params['output_connectivity'] = output_connectivity
    
model = Basic(network_params)
# Train
train_params = {}
train_params['training_iters'] = 300000
train_params['learning_rate'] = 0.008 # 0.003; 0.005 highnoise2: 0.008
train_params['training_weights_path'] = './saved_weights/train_weights/' + model_name
train_params['save_training_weights_epoch'] = 300

losses, initialTime, trainTime = model.train(task, train_params) # train model to perform pd task

print('initialTime:', initialTime, 'trainTime:',trainTime)

# plot loss
plt.figure()
plt.plot(losses)
plt.title('Loss during training')
plt.ylabel('Minibatch loss')
plt.xlabel('Batch number')
plt.legend()
plt.savefig('./Figures/trainCorrect/' + model_name + '_trainingLoss')

model.save('./saved_weights/' + model_name)

# Accuracy
iters=10
accuracy = np.zeros(iters)
taskaccuracy = np.zeros(iters)
percaccuracy = np.zeros(iters)
switchp = np.zeros(iters)
for i in range(iters):
    task_test = StimHist_constant2(dt, tau, T, N_batch, K, dat1=tParams_df, dat1_inds=test_inds, \
                                   in_noise=in_noise, vis_noise=vis_noise, mem_noise=mem_noise)
        
    test_inputs, target_output, mask, trial_params = task_test.get_trial_batch()
    model_output, state_var = model.test(test_inputs)
    
    accuracy[i] = mf.get_modelAccuracy(model_output,target_output)
    taskaccuracy[i] = mf.get_modelTaskAccuracy(model_output, trial_params)
    percaccuracy[i] = mf.get_modelPercAccuracy(model_output, trial_params)
    switchp[i] = mf.modelSwitchPercentage(model_output, trial_params)
    
print('accuracy: ',np.mean(accuracy), '+/-', np.std(accuracy))
print('task accuracy: ',np.mean(taskaccuracy), '+/-', np.std(taskaccuracy))
print('perc accuracy: ',np.mean(percaccuracy), '+/-', np.std(percaccuracy))
print('switch prob: ',np.mean(switchp), '+/-', np.std(switchp))

# Plots
weights = model.get_weights()
vf.plot_weights(weights['W_rec'])