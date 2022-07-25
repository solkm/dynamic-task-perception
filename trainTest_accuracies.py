#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 12:53:37 2021

@author: Sol
"""

from FourAFC_taskModels import StimHist_constant2
from psychrnn.backend.models.basic import Basic
import numpy as np
import scipy.io
import csv
import matplotlib.pyplot as plt
from model_functions import get_modelAccuracy
from statsmodels.stats.weightstats import ztest
import pandas as pd

N_rec = 200
N_batch = 100
dt = 10
tau = 100
T = 1000
K = 20

rec_noise=0.1
in_noise=0.1
vis_noise=0
mem_noise=0

tParams_df = pd.read_csv('./tParams_df.csv')
N_tot = tParams_df.shape[0]
tParams_mat = scipy.io.loadmat('./Cheng_recTrainTest/tParams.mat')
t_ind = np.arange(N_tot)

# remove indices that overlap sessions, v4(2)+
sess_ind = np.array(tParams_mat['r_indinit'][0], dtype=int)-1
N_s = len(sess_ind)
del_ind = np.empty((N_s-1)*(K-1), dtype=int)
for j in range(0, N_s-1):
    temp = np.arange(sess_ind[j+1]-K+1, sess_ind[j+1], dtype=int)
    del_ind[j*(K-1):(j+1)*(K-1)] = temp
t_ind2 = np.delete(t_ind, del_ind)

# overrepresent perceptual and task errors
corr_ind = np.array(tParams_df[tParams_df['err']==0].index, dtype='int')
corrTerrP_ind = np.array(tParams_df[(tParams_df['task_err']==0) & (tParams_df['perc_err']==1)].index, dtype='int')
errT_ind = np.array(tParams_df[tParams_df['task_err']==1].index, dtype='int')

N_25 = len(errT_ind)//2
np.random.seed(10)
np.random.shuffle(corrTerrP_ind)
trial_inds = np.concatenate((errT_ind, corrTerrP_ind[0:N_25]))
dat1_inds = trial_inds[trial_inds-K+1 >= 0] -K+1
dat2_inds = corr_ind[corr_ind-K+1 >= 0] -K+1

dat1_inds = dat1_inds[np.isin(dat1_inds,t_ind2)]
dat2_inds = dat2_inds[np.isin(dat2_inds,t_ind2)]
pdat2=0.25

# ---------------------- Load models ---------------------------
task = StimHist_constant2(dt, tau, T, N_batch, K, dat1=tParams_df)

network_params1 = task.get_task_params()
network_params1['name'] = 'choice'
network_params1['N_rec'] = N_rec
network_params1['rec_noise'] = rec_noise
network_params1['load_weights_path'] = './saved_weights/trainChoice_v4_2.npz'

choice_model = Basic(network_params1)

network_params2 = task.get_task_params()
network_params2['name'] = 'correct'
network_params2['N_rec'] = N_rec
network_params2['rec_noise'] = rec_noise
network_params2['load_weights_path'] = './saved_weights/trainChoice_v4_2_correctctrl.npz'

correct_model = Basic(network_params2)

# ---------------------- Test models ---------------------------
iters=500

choice_choice = np.zeros(iters)
correct_choice = np.zeros(iters)
choice_correct = np.zeros(iters)
correct_correct = np.zeros(iters)

test_choice = StimHist_constant2(dt, tau, T, N_batch, K, dat1=tParams_df, dat1_inds=dat1_inds, dat2=tParams_df, dat2_inds=dat2_inds, pdat2=pdat2, targChoice=True, \
                                 in_noise=in_noise, vis_noise=vis_noise, mem_noise=mem_noise)
test_correct = StimHist_constant2(dt, tau, T, N_batch, K, dat1=tParams_df, dat1_inds=dat1_inds, dat2=tParams_df, dat2_inds=dat2_inds, pdat2=pdat2, targChoice=False, \
                                 in_noise=in_noise, vis_noise=vis_noise, mem_noise=mem_noise)

for i in range(iters):
    print('batch ',i+1)
    x, target_output, _, _ = test_choice.get_trial_batch()
    model_output, _ = choice_model.test(x)
    model_accuracy = get_modelAccuracy(model_output, target_output)
    choice_choice[i] = model_accuracy
    
    x, target_output, _, _ = test_choice.get_trial_batch()
    model_output, _ = correct_model.test(x)
    model_accuracy = get_modelAccuracy(model_output, target_output)
    correct_choice[i] = model_accuracy
    
    x, target_output, _, _ = test_correct.get_trial_batch()
    model_output, _ = choice_model.test(x)
    model_accuracy = get_modelAccuracy(model_output, target_output)
    choice_correct[i] = model_accuracy
    
    x, target_output, _, _ = test_correct.get_trial_batch()
    model_output, _ = correct_model.test(x)
    model_accuracy = get_modelAccuracy(model_output, target_output)
    correct_correct[i] = model_accuracy
'''
with open('model_trainTestAccuracies.csv', 'a') as file:
    writer = csv.writer(file)
    writer.writerow(['trainChoice_v4(2)','choice_choice', [str(x) for x in choice_choice]])
    writer.writerow(['trainChoice_v4(2)','correct_choice', [str(x) for x in correct_choice]])
    writer.writerow(['trainChoice_v4(2)','choice_correct', [str(x) for x in choice_correct]])
    writer.writerow(['trainChoice_v4(2)','choice_choice', [str(x) for x in choice_choice]])
'''
# ---------------------- Plot the results ---------------------------

accuracies = np.array([np.mean(choice_choice), np.mean(correct_choice), np.mean(choice_correct), np.mean(correct_correct)])
error = np.array([np.std(choice_choice), np.std(correct_choice), np.std(choice_correct), np.std(correct_correct)])
fig, ax = plt.subplots()
ax.bar(np.arange(len(accuracies)), accuracies, yerr=error, align='center', alpha=0.5, capsize=5)
ax.set_xticks(np.arange(len(accuracies)))
ax.set_xticklabels(['M-M', 'C-M', 'M-C', 'C-C'], fontsize=18)
ax.set_title('Train-Test Accuracies', fontsize=18)
plt.xlabel('M = monkey choice, C = correct choice', fontsize=18)
plt.tick_params(axis='y', labelsize=14)
plt.hlines(.25,-0.6,3.6,colors='k',linestyles='--')
ax.yaxis.grid(True)
plt.savefig('./Figures/uPNC_poster/correctChoiceTrainTest', dpi=200)