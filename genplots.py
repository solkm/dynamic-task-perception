#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 14:03:37 2021

@author: Sol
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt
import behavior_analysis as ba
import perf_measures as pm
import visfunctions as vf
import pandas as pd

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'
#%%
# load test data

mname = 'correctv2_6'
model_output_m1 = pickle.load(open('./pickled_testdata/' + mname + '_testm1_modeloutput.pickle', 'rb'))
model_output_m2 = pickle.load(open('./pickled_testdata/' + mname + '_testm2_modeloutput.pickle', 'rb'))
trial_params_m1 = pickle.load(open('./pickled_testdata/' + mname + '_testm1_trialparams.pickle', 'rb'))
trial_params_m2 = pickle.load(open('./pickled_testdata/' + mname + '_testm2_trialparams.pickle', 'rb'))
    
model_output_all = np.concatenate((model_output_m1, model_output_m2), axis=0)
trial_params_all = np.concatenate((trial_params_m1, trial_params_m2), axis=0)
#%% current-trial stim dependence of switching
LtoF, FtoL = ba.switchOnStim(trial_params_all, model_output_all)
print(LtoF)
print(FtoL)

tParams_df = pd.read_csv('./tParams_df.csv')
LtoF_b, FtoL_b = ba.switchOnStim(tParams_df, None)

D=FtoL_b
plt.bar(range(len(D)), list(D.values()), align='center')
plt.xticks(range(len(D)), list(D.keys()))
plt.title('FtoL, monkey')

#%%
# Behavioral analysis: Perceptual performance after reward vs. non-reward
'''
from scipy.io import loadmat
tParams_mat = loadmat('./mat_files/tParams.mat')
dsl_perf_aR0, dsl_perf_aNR0, dsf_perf_aR0, dsf_perf_aNR0, p_dsl0, p_dsf0 = ba.percPerfSameStim(tParams_mat, model_output=None)
plt.figure()
plt.scatter(dsl_perf_aR0, dsl_perf_aNR0, edgecolors='m', facecolors='none', label='SL choice, p=%2.1e'%p_dsl0)
plt.scatter(dsf_perf_aR0, dsf_perf_aNR0, edgecolors='c', facecolors='none', label='SF choice, p=%2.1e'%p_dsf0)
plt.scatter(np.mean(dsl_perf_aR0), np.mean(dsl_perf_aNR0), s=60, edgecolors='k', facecolors='m')
plt.scatter(np.mean(dsf_perf_aR0), np.mean(dsf_perf_aNR0), s=60, edgecolors='k', facecolors='c')
X = np.linspace(0,1,100)
plt.plot(X,X, color='slategrey')
minperf = np.min(np.concatenate((dsl_perf_aR0, dsl_perf_aNR0, dsf_perf_aR0, dsf_perf_aNR0)))
plt.xlim(left = min(minperf-.02, 0.5))
plt.ylim(bottom = min(minperf-.02, 0.5))
plt.xlabel('After a rewarded trial', fontsize=16)
plt.ylabel('After an unrewarded trial', fontsize=16)
plt.legend(fontsize=12, loc='upper left')
plt.title('Perceptual performance for the same stimulus', fontsize=16)
plt.tick_params(labelsize=12)
plt.show()
#plt.savefig('./Figures/uPNC_poster/tParams_behana0', dpi=200)
'''
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
plt.xlabel('After a rewarded trial', fontsize=16)
plt.ylabel('After an unrewarded trial', fontsize=16)
plt.legend(fontsize=12, loc='upper left')
plt.title('Perceptual performance for the same stimulus', fontsize=16)
plt.tick_params(labelsize=12)
plt.show()
#plt.savefig('./Figures/trainCorrect/fullConn2_behana0', dpi=150)

#%%
# Behavioral analysis: Switch proportions after non-reward (and optionally reward) trials with different stimulus conditions


NRnorm_LF, NRnorm_FL = ba.switchAfterStim(trial_params_all, model_output_all, NRnorm=True, plot=False)


fig, ax = plt.subplots()
width = 0.9
labels = [r'$\Delta SL_{big},\,\Delta SF_{big}$', r'$\Delta SL_{big},\,\Delta SF_{sm}$', r'$\Delta SL_{sm},\,\Delta SF_{big}$', r'$\Delta SL_{sm},\,\Delta SF_{sm}$']
colors = ['tab:blue', 'm', 'c', 'slategrey']
for x in range(4):
    ax.bar(x, NRnorm_LF[x], width=width, label=labels[x], color=colors[x])
    ax.bar(x+5, NRnorm_FL[x], width=width, color=colors[x])
ax.bar([11], [0], width=width)
ax.set_ylabel('Proportion of switch-preceding NRs'+'\n $\div$ '+ 'proportion of all NRs', fontsize=18)
ax.set_xticklabels(['Location to frequency','Frequency to location'], fontsize=18)
ax.set_xticks([1.5,6.5])
plt.title('Stimulus-dependent switching after non-reward trials (NRs)', fontsize=18)
ax.tick_params(axis='y',labelsize=12)
plt.ylim(0,1.5)
plt.legend(fontsize=16, loc='upper right')

#plt.savefig('./Figures/uPNC_poster/correctNoBio_behana1', dpi=200)

#%%
# Accuracies

y2 = pm.accuracyMeasures(trial_params_all, model_output_all)
    
x=np.arange(4)
width=0.4

fig, ax = plt.subplots()
ax.bar(x-width/2-.01, 100*y2[0:4], width=width, label='Model', color='tab:blue')
ax.bar(x+width/2+.01, 100*y2[4:], width=width, label='Monkey', color='firebrick')
ax.set_xticks(x)
ax.set_xticklabels(['Accuracy', 'Task\naccuracy', 'Perceptual\naccuracy', 'Switch\npercentage'], fontsize=18)
ax.set_ylabel('Percentages', fontsize=18)
plt.legend(fontsize=14)
plt.ylim(0,100)
ax.tick_params(axis='y', labelsize=12)
vf.add_value_labels(ax, spacing=2, dec=1, fontsize=14)

plt.savefig('./Figures/uPNC_poster/correctNoBio_accuracies', dpi=200)

#%%
# Percentage of monkey behavior predicted

y = pm.choicePrediction(trial_params_all, model_output_all).values()

fig, ax = plt.subplots()
ax.bar(np.arange(len(y)), y, align='center', alpha=1)
ax.set_xticks(np.arange(len(y)))
ax.set_xticklabels(['All', 'Correct', 'Error', 'Task \nError', 'Perceptual \nError', 'Switch', 'Correct \nSwitch', 'Error \nSwitch'])
ax.set_ylabel('Percent predicted')
ax.set_xlabel('Monkey choices')
vf.add_value_labels(ax, spacing=2, dec=0)
#plt.savefig('./Figures/trainChoice/v4_2_2_choicePreds', bbox_inches="tight")

#%%
# Behavioral analysis (no plot): Exploratory switching/switch efficiency

sp, spErr, spErrAr = ba.exploreSwitch(trial_params_all, model_output_all)
print(sp, spErr, spErrAr, spErr/sp)

#%%
# plot recurrent weights

weights_path = './saved_weights/trainCorrect_v2_4.npz'
weights = np.load(weights_path)

W_in = weights['W_in']
W_rec = weights['W_rec']

sort_by = np.mean(W_rec, axis=0)
perc_sort2 = np.flip(np.argsort(sort_by[:40].copy())) 
task_sort2 = np.flip(np.argsort(sort_by[40:80].copy()))+40
perc_sort_2i = np.argsort(sort_by[80:90].copy())+80
task_sort_2i = np.argsort(sort_by[90:].copy())+90

sort_ind2 = np.concatenate((perc_sort2, task_sort2, perc_sort_2i, task_sort_2i))
W_rec_sort2 = W_rec[:,sort_ind2].copy()
W_rec_sort2 = W_rec_sort2[sort_ind2,:].copy()

vf.plot_weights(W_rec_sort2, colorbar=False)
plt.ylabel('To', fontsize=14)
plt.title('From', fontsize=14)
plt.savefig('./Figures/uPNC_poster/correctAllBio_recweights',dpi=200)