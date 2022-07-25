#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 01:32:13 2021

@author: Sol
"""

from FourAFC_taskModels import WM_TaskRule_varDelay
from psychrnn.backend.models.basic import Basic
import numpy as np
import matplotlib.pyplot as plt
import visfunctions as vf
from model_functions import get_Accuracies
#%% set parameters
N_trainbatch = 100
train_iters = 150000
N_rec = 100
in_noise = 0.7
rec_noise = 0.1
L2_in = 0.01
L2_out = 0.01
L2_rec = 0.01
L2_FR = 0.01
learn_rate = 0.002
name = 'vardelay100'
varDelay = True

#%% TRAIN
modeltask = WM_TaskRule_varDelay(N_batch=N_trainbatch, in_noise=in_noise, varDelay=varDelay)
network_params = modeltask.get_task_params()
network_params['name'] = name
network_params['N_rec'] = N_rec
network_params['rec_noise'] = rec_noise
network_params['L2_in'] = L2_in
network_params['L2_rec'] = L2_rec
network_params['L2_out'] = L2_out
network_params['L2_firing_rate'] = L2_FR

train_params = dict()
train_params['training_iters'] = train_iters
train_params['learning_rate'] = learn_rate
train_params['training_weights_path'] = './WM_taskrule/' + name + '_'
train_params['save_training_weights_epoch'] = 250

model = Basic(network_params)
losses, initialTime, trainTime = model.train(modeltask, train_params)
print('initialTime:', initialTime, 'trainTime:',trainTime)
plt.figure() # plot loss
plt.plot(losses)
plt.title('Loss during training')
plt.ylabel('Minibatch loss')
plt.xlabel('Batch number')
plt.legend()
plt.savefig('./WM_taskrule/'+name+'_loss')

model.save('./WM_taskrule/'+name)
#%% TEST
#model.destruct()
in_noise_test = 0.5
N_testbatch = 10
load_path = './WM_taskrule/'+name+'.npz'

modeltask = WM_TaskRule_varDelay(N_batch=N_testbatch, in_noise=in_noise_test, on_rule=1.0, off_rule=0.0, varDelay=varDelay)
network_params = modeltask.get_task_params()
network_params['name'] = name
network_params['N_rec'] = N_rec
network_params['rec_noise'] = rec_noise
network_params['load_weights_path'] = load_path
model = Basic(network_params)

test_inputs, target_output, mask, trial_params = modeltask.get_trial_batch()
model_output, state_var = model.test(test_inputs)

"""pred=model_output[:,-1,:]
target=target_output[:,-1,:]
dww=np.concatenate(([np.argmax(pred, axis=1)],[np.argmax(target,axis=1)]),axis=0)
loss=np.diff(np.transpose(dww))
model_accuracy = 1.0 - np.count_nonzero(loss)/len(loss)
error_trials = np.flatnonzero(loss)"""

print(name)
accuracies1 = get_Accuracies(model_output, trial_params)
print('overall accuracy: ', accuracies1[2])
print('task accuracy: ', accuracies1[0])
print('non-response:', accuracies1[3])
print('perceptual accuracy: ', accuracies1[1])
#%% Test no rule
'''
N_testbatch2 = 10000

modeltask2 = WM_TaskRule(N_batch=N_testbatch2, in_noise=in_noise, P_norule=1.0, fixation=fixation, const_rule=const_rule)
network_params2 = modeltask2.get_task_params()
network_params2['name'] = name + '_norule'
network_params2['N_rec'] = N_rec
network_params2['rec_noise'] = rec_noise
network_params2['load_weights_path'] = load_path
model2 = Basic(network_params2)

test_inputs2, target_output2, mask2, trial_params2 = modeltask2.get_trial_batch()
model_output2, state_var2 = model2.test(test_inputs2)

print(network_params2['name'])
accuracies2 = get_Accuracies(model_output2, trial_params2)
print('overall accuracy: ', accuracies2[2])
print('task accuracy: ', accuracies2[0])
print('perceptual accuracy: ', accuracies2[1])
print('non-response: ', accuracies2[3])
'''
#%% Test competing rules
'''
N_testbatch3 = 10000

modeltask3 = WM_TaskRule(N_batch=N_testbatch3, in_noise=in_noise, P_bothrule=1.0, fixation=fixation, const_rule=const_rule)
network_params3 = modeltask3.get_task_params()
network_params3['name'] = name+'_bothrule'
network_params3['N_rec'] = N_rec
network_params3['rec_noise'] = rec_noise
network_params3['load_weights_path'] = load_path
model3 = Basic(network_params3)

test_inputs3, target_output3, mask3, trial_params3 = modeltask3.get_trial_batch()
model_output3, state_var3 = model3.test(test_inputs3)

print(network_params3['name'])
accuracies3 = get_Accuracies(model_output3, trial_params3)
print('overall accuracy: ', accuracies3[2])
print('task accuracy: ', accuracies3[0])
print('perceptual accuracy: ', accuracies3[1])
print('non-response: ', accuracies3[3])
'''
#%% Test weak rule
'''
WS = [0.25]
for ws in WS:
    N_testbatch4 = 10000
    modeltask4 = WM_TaskRule(N_batch=N_testbatch4, in_noise=in_noise, P_weakrule=1.0, weakstrength=ws, fixation=fixation, const_rule=const_rule)
    network_params4 = modeltask4.get_task_params()
    network_params4['name'] = name+'_weakrule_' + str(ws)
    network_params4['N_rec'] = N_rec
    network_params4['rec_noise'] = rec_noise
    network_params4['load_weights_path'] = load_path
    model4 = Basic(network_params4)
    
    test_inputs4, target_output4, mask4, trial_params4 = modeltask4.get_trial_batch()
    model_output4, state_var4 = model4.test(test_inputs4)
    
    print(network_params4['name'])
    accuracies4 = get_Accuracies(model_output4, trial_params4)
    print('overall accuracy: ', accuracies4[2])
    print('task accuracy: ', accuracies4[0])
    print('perceptual accuracy: ', accuracies4[1])
    print('non-response: ', accuracies4[3])
'''
#%% PLOT
plotWeights=False

fr = np.maximum(state_var, 0)
avg_unit_fr = np.mean(fr.reshape((fr.shape[0]*fr.shape[1], N_rec)), axis=0)
top20_avgfr = np.argsort(avg_unit_fr)[-20:]

t1=1 # trial to plot

outlabels = ['a1>a2', 'a1<a2', 'b1>b2', 'b1<b2', 'fixation']
inlabels = ['task a', 'task b', 'feature a', 'feature b', 'fixation']

vf.plot_activities(test_inputs, t1, dt=10, ylabel='Activity of input units', label=inlabels, colormap=None)
plt.savefig('./WM_taskrule/'+name+'_inputs.png', dpi=200, bbox_inches='tight')
vf.plot_activities(model_output, t1, dt=10, ylabel='Activity of output units', label=outlabels, colormap=None)
plt.savefig('./WM_taskrule/'+name+'_outputs.png', dpi=200, bbox_inches='tight')
vf.plot_activities(fr[:,:,top20_avgfr], t1, dt=10, ylabel='Firing rates (20 units)')
plt.savefig('./WM_taskrule/'+name+'_top20fr.png', dpi=200, bbox_inches='tight')
vf.plot_activities(target_output, t1, dt=10, ylabel='Target outputs', label=outlabels, colormap=None)
plt.savefig('./WM_taskrule/'+name+'_targs.png', dpi=200, bbox_inches='tight')

if plotWeights==True:
    weights = model.get_weights()
    vf.plot_weights(weights['W_rec'])
    vf.plot_weights(weights['W_in'])
    vf.plot_weights(weights['W_out'])
