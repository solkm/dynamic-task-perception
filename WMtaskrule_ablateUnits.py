#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 14:18:08 2021

@author: Sol
"""

import os
import numpy as np
from FourAFC_taskModels import WM_TaskRule2, WM_TaskRule_varDelay
from psychrnn.backend.models.basic import Basic
import pickle
import pandas as pd
from model_functions import get_Accuracies
from numpy.random import default_rng
rng=default_rng()
import copy
import matplotlib.pyplot as plt

#%%

name = 'vardelay100'
weights = dict(np.load('./WM_taskrule/vardelay100/Weights/vardelay100.npz', allow_pickle=True))

sv_10 = pickle.load(open('./WM_taskrule/vardelay100/vardelay100_1on0off_statevar.pickle','rb'))
fr_10 = np.maximum(sv_10,0)

avgfr_all = np.mean(fr_10.reshape((fr_10.shape[0]*fr_10.shape[1], fr_10.shape[2])), axis=0)
sortinds = np.argsort(-(avgfr_all)) # sort by avg FR
#%%
N_u = weights['W_in'].shape[0]
taskInWeights = np.zeros(N_u)
stimInWeights = np.zeros(N_u)
allInWeights = np.zeros(N_u)

for i in range(N_u):
    taskInWeights[i] = np.sum(abs(weights['W_in'][i,:2]))
    stimInWeights[i] = np.sum(abs(weights['W_in'][i,2:4]))
    allInWeights[i] = np.sum(abs(weights['W_in'][i,:]))

sortinds_bytaskweight = np.argsort(-taskInWeights)
sortinds_bystimweight = np.argsort(-stimInWeights)
sortinds_byallweight = np.argsort(-allInWeights)

"""plt.hist(allInWeights)
plt.figure()
plt.plot(np.arange(30), allInWeights[sortinds_byallweight[:30]])"""

plt.scatter(taskInWeights, stimInWeights)
plt.xlabel('task-rule input weights')
plt.ylabel('stimulus input weights')
#%%
#sorted_inds_toablate = [8,10,12,14]
#units_to_ablate = sortinds[sorted_inds_toablate]
np.random.seed(100)
units_to_ablate = np.random.choice(sortinds_bystimweight[:30],10)
#np.random.seed(100)
#units_to_ablate = np.random.choice(sortinds[:20],4)

mod_weights = copy.deepcopy(weights)
mod_weights['W_rec'][units_to_ablate,:]=0
mod_weights['W_rec'][:,units_to_ablate]=0
mod_weights['W_out'][:,units_to_ablate]=0
mod_weights['W_in'][units_to_ablate,:]=0

np.savez('./WM_taskrule/Ablation/vardelay100_ablate10RandomFrom30HighAllInseed100.npz', **mod_weights)

#%% ablate units one at a time
ablate_units = np.arange(1,101)
percAcc10 = np.zeros(len(ablate_units))
taskAcc10 = np.zeros(len(ablate_units))
percAcc46 = np.zeros(len(ablate_units))
taskAcc46 = np.zeros(len(ablate_units))

percAcc10A = np.zeros(len(ablate_units))
taskAcc10A = np.zeros(len(ablate_units))
percAcc46A = np.zeros(len(ablate_units))
taskAcc46A = np.zeros(len(ablate_units))

percAcc10B = np.zeros(len(ablate_units))
taskAcc10B = np.zeros(len(ablate_units))
percAcc46B = np.zeros(len(ablate_units))
taskAcc46B = np.zeros(len(ablate_units))

N_testbatch = 500
N_rec=100
rec_noise=0.1
in_noise=0.4
name = 'model3b_1000'
weights = dict(np.load('./WM_taskrule/'+name+'.npz', allow_pickle=True))

for j in range(len(ablate_units)):
    unit = ablate_units[j]
    print('unit: ',unit)
    i = unit-1
    mod_weights = copy.deepcopy(weights)
    mod_weights['W_rec'][i,:]=0
    mod_weights['W_rec'][:,i]=0
    mod_weights['W_out'][:,i]=0
    mod_weights['W_in'][i,:]=0
    
    np.savez('./mod_weights.npz', **weights)
    
    modeltask = WM_TaskRule2(N_batch=N_testbatch, in_noise=in_noise, on_rule=1.0, off_rule=0.0, const_rule=True)
    network_params = modeltask.get_task_params()
    network_params['name'] = name
    network_params['N_rec'] = N_rec
    network_params['rec_noise'] = rec_noise
    network_params['load_weights_path'] = './mod_weights.npz'
    model = Basic(network_params)
    
    test_inputs, target_output, mask, trial_params = modeltask.get_trial_batch()
    model_output, state_var = model.test(test_inputs)
    
    accuracies = get_Accuracies(model_output, trial_params)
    print('perceptual accuracy: ', accuracies[1])
    percAcc10[j] = accuracies[1]
    print('task accuracy: ', accuracies[0])
    taskAcc10[j] = accuracies[0]
    
    taskA_inds = temp = [i for i in range(N_testbatch) if trial_params[i]['task']==0]
    taskA_acc = get_Accuracies(model_output[taskA_inds,:,:], trial_params[taskA_inds])
    percAcc10A[j] = taskA_acc[1]
    print('perceptual accuracy, task A: ', taskA_acc[1])
    taskAcc10A[j] = taskA_acc[0]
    print('task accuracy, task A: ', taskA_acc[0])

    taskB_inds = temp = [i for i in range(N_testbatch) if trial_params[i]['task']==1]
    taskB_acc = get_Accuracies(model_output[taskB_inds,:,:], trial_params[taskB_inds])
    percAcc10B[j] = taskB_acc[1]
    print('perceptual accuracy, task B: ', taskB_acc[1])
    taskAcc10B[j] = taskB_acc[0]
    print('task accuracy, task B: ', taskB_acc[0])
    
    model.destruct()
    
    modeltask = WM_TaskRule2(N_batch=N_testbatch, in_noise=in_noise, on_rule=0.6, off_rule=0.4, const_rule=True)
    network_params = modeltask.get_task_params()
    network_params['name'] = name
    network_params['N_rec'] = N_rec
    network_params['rec_noise'] = rec_noise
    network_params['load_weights_path'] = './mod_weights.npz'
    model = Basic(network_params)
    
    test_inputs, target_output, mask, trial_params = modeltask.get_trial_batch()
    model_output, state_var = model.test(test_inputs)
    
    accuracies = get_Accuracies(model_output, trial_params)
    print('perceptual accuracy, weak rule: ', accuracies[1])
    percAcc46[j] = accuracies[1]
    print('task accuracy, weak rule: ', accuracies[0])
    taskAcc46[j] = accuracies[0]
    
    taskA_inds = temp = [i for i in range(N_testbatch) if trial_params[i]['task']==0]
    taskA_acc = get_Accuracies(model_output[taskA_inds,:,:], trial_params[taskA_inds])
    percAcc46A[j] = taskA_acc[1]
    taskAcc46A[j] = taskA_acc[0]

    taskB_inds = temp = [i for i in range(N_testbatch) if trial_params[i]['task']==1]
    taskB_acc = get_Accuracies(model_output[taskB_inds,:,:], trial_params[taskB_inds])
    percAcc46B[j] = taskB_acc[1]
    taskAcc46B[j] = taskB_acc[0]
    
    model.destruct()

#%% get baseline accuracies
N_testbatch=1000

modeltask = WM_TaskRule2(N_batch=N_testbatch, in_noise=in_noise, on_rule=1.0, off_rule=0.0, const_rule=True)
network_params = modeltask.get_task_params()
network_params['name'] = name
network_params['N_rec'] = N_rec
network_params['rec_noise'] = rec_noise
network_params['load_weights_path'] = './WM_taskrule/'+name+'.npz'
model = Basic(network_params)

test_inputs, target_output, mask, trial_params = modeltask.get_trial_batch()
model_output, state_var = model.test(test_inputs)

accuracies = get_Accuracies(model_output, trial_params)
base_percAcc10 = accuracies[1]
base_taskAcc10 = accuracies[0]

taskA_inds = temp = [i for i in range(N_testbatch) if trial_params[i]['task']==0]
taskA_acc = get_Accuracies(model_output[taskA_inds,:,:], trial_params[taskA_inds])
base_percAcc10A = taskA_acc[1]
base_taskAcc10A = taskA_acc[0]

taskB_inds = temp = [i for i in range(N_testbatch) if trial_params[i]['task']==1]
taskB_acc = get_Accuracies(model_output[taskB_inds,:,:], trial_params[taskB_inds])
base_percAcc10B = taskB_acc[1]
base_taskAcc10B = taskB_acc[0]

model.destruct()

modeltask = WM_TaskRule2(N_batch=N_testbatch, in_noise=in_noise, on_rule=0.6, off_rule=0.4, const_rule=True)
network_params = modeltask.get_task_params()
network_params['name'] = name
network_params['N_rec'] = N_rec
network_params['rec_noise'] = rec_noise
network_params['load_weights_path'] = './WM_taskrule/'+name+'.npz'
model = Basic(network_params)

test_inputs, target_output, mask, trial_params = modeltask.get_trial_batch()
model_output, state_var = model.test(test_inputs)

accuracies = get_Accuracies(model_output, trial_params)
base_percAcc46 = accuracies[1]
base_taskAcc46 = accuracies[0]

taskA_inds = temp = [i for i in range(N_testbatch) if trial_params[i]['task']==0]
taskA_acc = get_Accuracies(model_output[taskA_inds,:,:], trial_params[taskA_inds])
base_percAcc46A = taskA_acc[1]
base_taskAcc46A = taskA_acc[0]

taskB_inds = temp = [i for i in range(N_testbatch) if trial_params[i]['task']==1]
taskB_acc = get_Accuracies(model_output[taskB_inds,:,:], trial_params[taskB_inds])
base_percAcc46B = taskB_acc[1]
base_taskAcc46B = taskB_acc[0]

model.destruct()

# append to unit measures df

df = pd.read_csv('./WM_taskrule/unitIndices_model3b_1000.csv')

df['percAcc10']=percAcc10 - base_percAcc10
df['percAcc46']=percAcc46 - base_percAcc46
df['taskAcc10']=taskAcc10 - base_taskAcc10
df['taskAcc46']=taskAcc46 - base_taskAcc46

df['percAcc10A']=percAcc10A - base_percAcc10A
df['percAcc46A']=percAcc46A - base_percAcc46A
df['percAcc46B']=percAcc46B - base_percAcc46B
df['percAcc10B']=percAcc10B - base_percAcc10B

df['taskAcc10A']=taskAcc10A - base_taskAcc10A
df['taskAcc46A']=taskAcc46A - base_taskAcc46A
df['taskAcc10B']=taskAcc10B - base_taskAcc10B
df['taskAcc46B']=taskAcc46B - base_taskAcc46B

df.to_csv('./WM_taskrule/unitIndices_model3b_1000.csv',index=False)

#%%
import matplotlib.pyplot as plt

plt.hist(df['percAcc46'])
plt.hist(df['percAcc10'])