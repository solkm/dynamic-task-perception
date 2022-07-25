#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 00:53:39 2021

@author: Sol
"""

from FourAFC_taskModels import WM_TaskRule_varDelay
from psychrnn.backend.models.basic import Basic
import numpy as np
import matplotlib.pyplot as plt
from model_functions import get_Accuracies
#%%
N_rec = 100
in_noise = 0.5
rec_noise = 0.1
N_testbatch = 1000
name = 'vardelay100_ablate10RandomFrom30HighAllInseed100'
load_path = './WM_taskrule/Ablation/'+name+'.npz'
varDelay = True

onRules = np.round(np.linspace(0,1,11), decimals=1)
offRules = np.round(np.linspace(0,1,11), decimals=1)

#%%
accuracyMatrix = np.zeros((4,len(onRules),len(offRules)))

for i in range(len(onRules)):
    for j in range(len(offRules)):
        
        on_rule = onRules[i]
        off_rule = offRules[j]
        print('rule strengths: ', on_rule, off_rule)
        
        modeltask = WM_TaskRule_varDelay(N_batch=N_testbatch, in_noise=in_noise, on_rule=on_rule, off_rule=off_rule, varDelay=varDelay)
        network_params = modeltask.get_task_params()
        network_params['name'] = name
        network_params['N_rec'] = N_rec
        network_params['rec_noise'] = rec_noise
        network_params['load_weights_path'] = load_path
        model = Basic(network_params)
        
        test_inputs, target_output, mask, trial_params = modeltask.get_trial_batch()
        model_output, state_var = model.test(test_inputs)
        
        accuracies = np.array(get_Accuracies(model_output, trial_params)[:4])
        
        print('overall accuracy: ', accuracies[2])
        accuracyMatrix[0,i,j] = accuracies[2]
        print('task accuracy: ', accuracies[0])
        accuracyMatrix[1,i,j] = accuracies[0]
        print('perceptual accuracy: ', accuracies[1])
        accuracyMatrix[2,i,j] = accuracies[1]
        print('non-response:', accuracies[3])
        accuracyMatrix[3,i,j] = accuracies[3]
        
        model.destruct()


with open('./WM_taskrule/'+name+'_accMat.npy', 'wb') as f:
    np.save(f, accuracyMatrix)
#%%
with open('./WM_taskrule/'+name+'_accMat.npy', 'rb') as f:
    am = np.load(f)
print(np.min(am[2]), np.max(am[2]))
#%%
plotPercAcc = True

fig,ax = plt.subplots()
if plotPercAcc:
    im = ax.imshow(am[2]*100, cmap='viridis', vmin=46, vmax=61)
    plt.title('Perceptual Accuracy (%)')
else:
    im = ax.imshow(am[1]*100, cmap='viridis', vmin=0, vmax=100)
    plt.title('Task Accuracy (%)')
plt.colorbar(im, orientation='vertical')
ax.set_xticks(np.arange(11))
ax.set_yticks(np.arange(11))
ax.set_xticklabels(offRules)
ax.set_yticklabels(onRules)
plt.xlabel('Irrelevant Task Rule Strength')
plt.ylabel('Relevant Task Rule Strength')
plt.show()
if plotPercAcc:
    plt.savefig('./WM_taskrule/'+name+'_perceptualAccuracyPlot', dpi=200, bbox_inches='tight')
else:
    plt.savefig('./WM_taskrule/'+name+'_taskAccuracyPlot_normed', dpi=200, bbox_inches='tight')