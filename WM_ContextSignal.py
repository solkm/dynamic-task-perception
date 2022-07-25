#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 21:34:13 2021

@author: Sol
"""

from FourAFC_taskModels import WM_ContextSignal_varDelay
from psychrnn.backend.models.basic import Basic
import numpy as np
import matplotlib.pyplot as plt
import visfunctions as vf
import model_functions as mf
import pandas as pd
from scipy.io import loadmat
import pickle
#%% basic parameters, split data

N_rec = 200
in_noise = 0.7
mem_noise=0.2
rec_noise = 0.1
K=7
trainChoice=False
taskMask=False
name = 'ctxtsig_vardelay1000'

# Train data
tParams_df = pd.read_csv('./tParams_df.csv')
N_tot = tParams_df.shape[0]
tParams_mat = loadmat('./Cheng_recTrainTest/tParams.mat')
t_ind = np.arange(N_tot)
# remove indices that overlap sessions
sess_ind = np.array(tParams_mat['r_indinit'][0], dtype=int)-1
N_s = len(sess_ind)
del_ind = np.empty((N_s)*(K-1), dtype=int)
for j in range(0, N_s):
    temp = np.arange(sess_ind[j], sess_ind[j]+K-1, dtype=int)
    del_ind[j*(K-1):(j+1)*(K-1)] = temp

t_ind = np.delete(t_ind, del_ind)

np.random.seed(413)
N_test = 10000 #2000
np.random.shuffle(t_ind)
test_inds = t_ind[:N_test] # test set
train_inds = t_ind[N_test:]
#%% TRAIN

saveTrainWeights=False
preTrain=False
freezePerc=False

L2_in = 0.01
L2_out = 0.01
L2_rec = 0.01
L2_FR = 0.01

train_iters = 350000
N_trainbatch = 100
learn_rate = 0.002

task = WM_ContextSignal_varDelay(in_noise=in_noise, mem_noise=mem_noise, N_batch=N_trainbatch, \
                            dat=tParams_df, dat_inds=train_inds, K=K, trainChoice=trainChoice, taskMask=taskMask)

network_params = task.get_task_params()
network_params['name'] = name
network_params['N_rec'] = N_rec
network_params['rec_noise'] = rec_noise
network_params['L2_in'] = L2_in
network_params['L2_rec'] = L2_rec
network_params['L2_out'] = L2_out
network_params['L2_firing_rate'] = L2_FR

N_in = task.N_in
N_out = task.N_out
input_connectivity = np.ones((N_rec, N_in))
rec_connectivity = np.ones((N_rec, N_rec))
output_connectivity = np.ones((N_out, N_rec))

input_connectivity[:100, -3:] = 0
input_connectivity[100:, :-3] = 0
output_connectivity[:, :100] = 0
rec_connectivity[100:, :98] = 0
rec_connectivity[:98, 98:] = 0
rec_connectivity[98:100, 100:] = 0
    
network_params['input_connectivity'] = input_connectivity
network_params['rec_connectivity'] = rec_connectivity
network_params['output_connectivity'] = output_connectivity

model = Basic(network_params)
temp_weights = model.get_weights()
for k,v in temp_weights.items():
    network_params[k] = v
model.destruct()

if preTrain==True:
    percweights = dict(np.load('./WM_taskrule/vardelay100.npz', allow_pickle=True))
    network_params['W_in'][100:,-3:] = percweights['W_in'][:,-3:]
    network_params['W_rec'][100:, 98:100] = percweights['W_in'][:,:2]
    network_params['W_out'][:,100:] = percweights['W_out']
    network_params['W_rec'][100:,100:] = percweights['W_rec']

model = Basic(network_params)

# Set fixed weight matrices to the default -- fully trainable
W_in_fixed = np.zeros((N_rec,N_in))
W_rec_fixed = np.zeros((N_rec,N_rec))
W_out_fixed = np.zeros((N_out, N_rec))

# Specify certain weights to fix.
if freezePerc==True:
    W_rec_fixed[100:, 98:] = 1
    W_in_fixed[100:,-3:] = 1
    W_out_fixed[:,:] = 1

# Specify the fixed weights parameters in train_params
train_params = {}
train_params['fixed_weights'] = {
    'W_in': W_in_fixed,
    'W_rec': W_rec_fixed,
    'W_out': W_out_fixed
}
train_params['training_iters'] = train_iters
train_params['learning_rate'] = learn_rate
if saveTrainWeights==True:
    train_params['training_weights_path'] = './WM_taskrule/' + name
    train_params['save_training_weights_epoch'] = 1000

losses, initialTime, trainTime = model.train(task, train_params) # train model to perform pd task

print('initialTime:', initialTime, 'trainTime:',trainTime)

# plot loss
plt.figure()
plt.plot(losses)
plt.title('Loss during training')
plt.ylabel('Minibatch loss')
plt.xlabel('Batch number')
plt.legend()
plt.savefig('./WM_taskrule/'+name+'_loss')

model.save('./WM_taskrule/'+name)

model.destruct()
#%% TEST

in_noise = 1.5
mem_noise = 2.0
task = WM_ContextSignal_varDelay(in_noise=in_noise, mem_noise=mem_noise, N_batch=N_test, \
                                 dat=tParams_df, dat_inds=test_inds, K=K, testall=True, trainChoice=False)
#task = WM_ContextSignal_varDelay(in_noise=in_noise, mem_noise=mem_noise, N_batch=20000, \
#                                 dat=tParams_df, dat_inds=train_inds, K=K, testall=False, trainChoice=False)

network_params = task.get_task_params()
network_params['name'] = name
network_params['N_rec'] = N_rec
network_params['rec_noise'] = rec_noise
network_params['load_weights_path'] = './WM_taskrule/'+name+'.npz'
model = Basic(network_params)
#'''
test_inputs, target_output, mask, trial_params = task.get_trial_batch()
model_output, state_var = model.test(test_inputs)

print(name)
print('noise: ', in_noise, mem_noise)
print('overall accuracy: ', mf.get_modelAccuracy(model_output, target_output))
print('task accuracy: ', mf.get_modelTaskAccuracy(model_output, trial_params))
print('perceptual accuracy: ', mf.get_modelPercAccuracy(model_output, trial_params))

model.destruct()
#%% save testing data

savefile = open('./' +name+ '_modeloutput.pickle','wb')
pickle.dump(model_output, savefile, protocol=4)
savefile.close()
#%% construct dataframe for analyzing model choices
choice_0to3 = np.argmax(model_output[:,-1,:-1], axis=1)
N = trial_params.shape[0]

prev_mtask = mf.get_tasks([trial_params[i]['choice'][-2] for i in range(N)])
                    
modeldf = pd.DataFrame()
modeldf['trial'] = [trial_params[i]['trial_ind'] for i in range(N)]
modeldf['model_choice'] = choice_0to3 + 1
modeldf['model_task'] = np.where(((modeldf['model_choice']==1)|(modeldf['model_choice']==2)), 2, 1).astype('uint8')
modeldf['correct_choice'] = np.array(tParams_df.loc[modeldf['trial'], 'correct'])
modeldf['correct_task'] = np.array(tParams_df.loc[modeldf['trial'], 'task'])
modeldf['dsf'] = np.array(tParams_df.loc[modeldf['trial'], 'dsf'])
modeldf['dsl'] = np.array(tParams_df.loc[modeldf['trial'], 'dsl'])
modeldf['model_switch'] = np.where([modeldf.loc[i,'model_task']!=prev_mtask[i] for i in range(N)], 1, 0).astype('uint8')
modeldf['model_error'] = np.where(modeldf['model_choice']!=modeldf['correct_choice'], 1, 0).astype('uint8')
modeldf['model_task_err'] = np.where(modeldf['correct_task']!=modeldf['model_task'], 1, 0).astype('uint8')
modeldf['model_perc_err'] = np.where( (((modeldf['model_choice']==1)&(modeldf['dsf']<=0))| \
                                    ((modeldf['model_choice']==2)&(modeldf['dsf']>=0))| \
                                    ((modeldf['model_choice']==3)&(modeldf['dsl']>=0))| \
                                    ((modeldf['model_choice']==4)&(modeldf['dsl']<=0))), 1, 0).astype('uint8')

#modeldf.to_csv('./'+name+'_df.csv', index=False)
#%% open testing data and dataframe
#name = 'ctxtsig_vardelay'

model_output = pickle.load(open('./' +name+ '_modeloutput.pickle','rb'))
modeldf = pd.read_csv('./'+name+'_df.csv')

#%% analyze modeldf

N = modeldf.shape[0]

print('model switch percentage ', np.count_nonzero(modeldf['model_switch'])/N * 100)
print('model accuracy ', (1 - np.count_nonzero(modeldf['model_error'])/N) * 100)
print('task accuracy ', (1 - np.count_nonzero(modeldf['model_task_err'])/N) * 100)
print('perceptual accuracy ', (1 - np.count_nonzero(modeldf['model_perc_err'])/N) * 100)

# trials where model made error: modeldf[modeldf['model_error']==1]['trial']

print('% of model errors that are also monkey errors ', \
      np.count_nonzero(tParams_df.loc[modeldf[modeldf['model_error']==1]['trial'], 'err'])\
      /modeldf[modeldf['model_error']==1].shape[0] * 100)

monkey_err_inds = np.where(tParams_df.loc[modeldf['trial'], 'err'] == 1)[0]

print('% of monkey errors that are also model errors ', \
      np.count_nonzero(modeldf.loc[monkey_err_inds, 'model_error']) / monkey_err_inds.shape[0] * 100)

after_err_inds = [tParams_df.loc[modeldf.loc[i,'trial']-1,'err'] == 1 for i in range(N)]
after_rew_inds = np.invert(after_err_inds)

print('model perceptual performance after error ', \
      (1 - np.count_nonzero(modeldf.loc[after_err_inds, 'model_perc_err'])\
       /modeldf.loc[after_err_inds, 'model_perc_err'].shape[0]) * 100)

print('model perceptual performance after reward ', \
      (1 - np.count_nonzero(modeldf.loc[after_rew_inds, 'model_perc_err'])\
       /modeldf.loc[after_rew_inds, 'model_perc_err'].shape[0]) * 100)
'''
print('monkey perceptual performance after error ', \
      (1 - np.count_nonzero(tParams_df.loc[modeldf.loc[after_err_inds,'trial'], 'perc_err'])\
       /tParams_df.loc[modeldf.loc[after_err_inds,'trial'], 'perc_err'].shape[0]) * 100)
print('monkey perceptual performance after reward ', \
      (1 - np.count_nonzero(tParams_df.loc[modeldf.loc[after_rew_inds,'trial'], 'perc_err'])\
       /tParams_df.loc[modeldf.loc[after_rew_inds,'trial'], 'perc_err'].shape[0]) * 100) 
'''
# added 1/28:

print('% of model PERC. errors that are also monkey PERC. errors ', \
      np.count_nonzero(tParams_df.loc[modeldf[modeldf['model_perc_err']==1]['trial'], 'perc_err'])\
      /modeldf[modeldf['model_perc_err']==1].shape[0] * 100)

monkey_perc_err_inds = np.where(tParams_df.loc[modeldf['trial'], 'perc_err'] == 1)[0]

print('% of monkey PERC. errors that are also model PERC. errors ', \
      np.count_nonzero(modeldf.loc[monkey_perc_err_inds, 'model_perc_err'])\
      /monkey_perc_err_inds.shape[0] * 100)

print('% of model TASK errors that are also monkey TASK errors ', \
      np.count_nonzero(tParams_df.loc[modeldf[modeldf['model_task_err']==1]['trial'], 'task_err'])\
      /modeldf[modeldf['model_task_err']==1].shape[0] * 100)

monkey_task_err_inds = np.where(tParams_df.loc[modeldf['trial'], 'task_err'] == 1)[0]

print('% of monkey TASK errors that are also model TASK errors ', \
      np.count_nonzero(modeldf.loc[monkey_task_err_inds, 'model_task_err'])\
      /monkey_task_err_inds.shape[0] * 100)





#%%

vf.plot_activities(test_inputs[:,:,-3:-1], 0, 10)
    
#%% test after switch
model.destruct()

afterSwitchInds = tParams_df[tParams_df['m_switch']==1].index+1
afterSwitchMonkeyAcc = np.count_nonzero(tParams_df.loc[afterSwitchInds,'err']==0)/afterSwitchInds.shape[0]
print('monkey accuracy after switch: ', afterSwitchMonkeyAcc)

task = WM_ContextSignal_varDelay(N_batch=afterSwitchInds[10:].shape[0], \
                                 dat=tParams_df, dat_inds=afterSwitchInds[10:], K=K, testall=True)
network_params = task.get_task_params()
network_params['name'] = name
network_params['N_rec'] = N_rec
network_params['rec_noise'] = rec_noise
network_params['load_weights_path'] = './WM_taskrule/'+name+'.npz'
model = Basic(network_params)

test_inputs, target_output, mask, trial_params = task.get_trial_batch()
model_output, state_var = model.test(test_inputs)

print(name)
print('overall accuracy after switch: ', mf.get_modelAccuracy(model_output, target_output))
print('task accuracy after switch: ', mf.get_modelTaskAccuracy(model_output, trial_params))
print('perceptual accuracy after switch: ', mf.get_modelPercAccuracy(model_output, trial_params))


#%% test more

in_noise = 1.5
mem_noise = 2.0

task = WM_ContextSignal_varDelay(in_noise=in_noise, mem_noise=mem_noise, N_batch=20000, \
                                 dat=tParams_df, dat_inds=train_inds, K=K, testall=False, trainChoice=False)

network_params = task.get_task_params()
network_params['name'] = name
network_params['N_rec'] = N_rec
network_params['rec_noise'] = rec_noise
network_params['load_weights_path'] = './WM_taskrule/'+name+'.npz'
model = Basic(network_params)
#'''
test_inputs, target_output, mask, trial_params2 = task.get_trial_batch()
model_output2, state_var = model.test(test_inputs)

print(name)
print('noise: ', in_noise, mem_noise)
print('overall accuracy: ', mf.get_modelAccuracy(model_output2, target_output))
print('task accuracy: ', mf.get_modelTaskAccuracy(model_output2, trial_params2))
print('perceptual accuracy: ', mf.get_modelPercAccuracy(model_output2, trial_params2))

model.destruct()
#%%
import behavior_analysis as ba

trial_params_all = np.concatenate((trial_params, trial_params2))
model_output_all = np.concatenate((model_output, model_output2))

dsl_perf_aR, dsl_perf_aNR, dsf_perf_aR, dsf_perf_aNR, p_dsl, p_dsf = ba.percPerfSameStim(trial_params_all, model_output_all)

plt.figure()
plt.scatter(dsl_perf_aR, dsl_perf_aNR, edgecolors='m', facecolors='none', label='SL choice, p=%2.1e'%p_dsl)
plt.scatter(dsf_perf_aR, dsf_perf_aNR, edgecolors='c', facecolors='none', label='SF choice, p=%2.1e'%p_dsf)
plt.scatter(np.mean(dsl_perf_aR), np.mean(dsl_perf_aNR), s=60, edgecolors='k', facecolors='m')
plt.scatter(np.mean(dsf_perf_aR), np.mean(dsf_perf_aNR), s=60, edgecolors='k', facecolors='c')
X = np.linspace(0,1,100)
plt.plot(X,X, color='slategrey')
minperf = np.min(np.concatenate((dsl_perf_aR, dsl_perf_aNR, dsf_perf_aR, dsf_perf_aNR)))
plt.xlim(left = min(minperf-.02, 0.5))
plt.ylim(bottom = min(minperf-.02, 0.5))
plt.xlabel('After a rewarded trial', fontsize=12)
plt.ylabel('After an unrewarded trial', fontsize=12)
plt.legend(fontsize=10, loc='upper left')
plt.title('Perceptual performance for the same stimulus', fontsize=12)
plt.tick_params(labelsize=10)
plt.show()