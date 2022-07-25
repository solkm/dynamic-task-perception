#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 12:11:41 2021

@author: Sol
"""

import numpy as np
from FourAFC_taskModels import StimHist_constant2
from psychrnn.backend.models.basic import Basic
import matplotlib.pyplot as plt
import pandas as pd
import visfunctions as vf
#import model_functions as mf
#from statsmodels.stats.weightstats import ztest

#%%
weights_path = './saved_weights/trainChoice_v4_2_correctctrl.npz'
model_name = 'v4_2_correctctrl'
weights = np.load(weights_path)

W_in = weights['W_in']
W_rec = weights['W_rec']
#%%
#--------------task inputs weights trend?----------------------------
N_rec = 200

W_in_task = W_in[N_rec//2:,0:-2] # no dale
#W_in_task = np.vstack((W_in[40:80,0:-2], W_in[90:100,0:-2])) # dale

N_taskin = len(W_in_task[0,:])
W_in_sumtaskproj = np.zeros(N_taskin)
thresh=0.0
for i in range(N_taskin):
    W_in_sumtaskproj[i] = np.sum(np.abs(W_in_task[:,i][W_in_task[:,i] > thresh]))

plt.figure()
X = np.arange(start=1,stop=N_taskin+1)
Y = W_in_sumtaskproj

import itertools

colors = itertools.cycle(['orange', 'c', 'm', 'r'])
labels = itertools.cycle(['dsl', 'dsf', 'choice', 'reward'])
for i in range(len(Y)):
    if i<4:
        plt.scatter(X[i], Y[i], color=next(colors), label=next(labels))
    else:
        plt.scatter(X[i], Y[i], color=next(colors))

#plt.scatter(X,Y)
plt.xlabel('Input, trial n-K+1 --> n')
plt.ylabel('Summed projections')
plt.legend()
#plt.savefig('./Figures/trainChoice/'+model_name+'Win_trend')
#%%
#------------Compute each unit's belief index------------------------
tParams_df = pd.read_csv('tParams_df.csv')
N_rec = 100
dt = 10
tau = 100
T = 1000
K = 10
N_batch = 1000

in_noise = 0.1
rec_noise = 0.1
vis_noise = 0.8
mem_noise = 0.3

#Load model
task1 = StimHist_constant2(dt, tau, T, N_batch, K, dat1=tParams_df, in_noise=in_noise, vis_noise=vis_noise, mem_noise=mem_noise)
network_params1 = task1.get_task_params()
network_params1['name'] = model_name
network_params1['N_rec'] = N_rec
network_params1['load_weights_path'] = weights_path
network_params1['rec_noise'] = rec_noise
model1 = Basic(network_params1)

# Get test batch
x1, target_output1, mask1, trial_params1 = task1.get_trial_batch()
model_output1, state_var1 = model1.test(x1)

state_var1 = np.maximum(state_var1,0) #pass state vars through RELU
model_choice = np.argmax(model_output1[:,-1,:], axis=1) + 1

sf_inc_tr = np.nonzero(model_choice==1)[0]
sf_dec_tr = np.nonzero(model_choice==2)[0]
sl_dec_tr = np.nonzero(model_choice==3)[0]
sl_inc_tr = np.nonzero(model_choice==4)[0]
N_fi = len(sf_inc_tr)
N_fd = len(sf_dec_tr)
N_li = len(sl_inc_tr)
N_ld = len(sl_dec_tr)

print(N_fi,N_fd,N_li,N_ld)

avg_act = dict(sf=np.zeros((N_fi+N_fd, N_rec)), sl=np.zeros((N_li+N_ld, N_rec)), sf_inc=np.zeros((N_fi, N_rec)), \
               sf_dec=np.zeros((N_fd, N_rec)), sl_inc=np.zeros((N_li, N_rec)), sl_dec=np.zeros((N_ld, N_rec)))
    
t1 = int(dt)
t2 = int(T/(4*dt))
t3 = int(T/(4*dt) + dt)
t4 = int(3*T/(4*dt))

for fi in range(N_fi):
    for unit in range(N_rec):
        avg_act['sf_inc'][fi,unit] = np.mean(state_var1[sf_inc_tr[fi], t3:t4, unit])
        avg_act['sf'][fi,unit] = np.mean(state_var1[sf_inc_tr[fi], t1:t2, unit])
for fd in range(N_fd):
    for unit in range(N_rec):
        avg_act['sf_dec'][fd,unit] = np.mean(state_var1[sf_dec_tr[fd], t3:t4, unit])
        avg_act['sf'][N_fi+fd,unit] = np.mean(state_var1[sf_dec_tr[fd], t1:t2, unit])
for li in range(N_li):
    for unit in range(N_rec):
        avg_act['sl_inc'][li,unit] = np.mean(state_var1[sl_inc_tr[li], t3:t4, unit])
        avg_act['sl'][li,unit] = np.mean(state_var1[sl_inc_tr[li], t1:t2, unit])
for ld in range(N_ld):
    for unit in range(N_rec):
        avg_act['sl_dec'][ld,unit] = np.mean(state_var1[sl_dec_tr[ld], t3:t4, unit])
        avg_act['sl'][N_li+ld,unit] = np.mean(state_var1[sl_dec_tr[ld], t1:t2, unit])

belief_ind = np.zeros(N_rec)
sf_stim_ind = np.zeros(N_rec)
sl_stim_ind = np.zeros(N_rec)

for u in range(N_rec):
    SF_u = np.mean(avg_act['sf'][:,u])
    SL_u = np.mean(avg_act['sl'][:,u])
    sigSF_u = np.std(avg_act['sf'][:,u])
    sigSL_u = np.std(avg_act['sl'][:,u])
    
    belief_ind[u] = np.abs(SF_u - SL_u)
    
    SFinc_u = np.mean(avg_act['sf_inc'][:,u])
    SFdec_u = np.mean(avg_act['sf_dec'][:,u])
    sigSFinc_u = np.std(avg_act['sf_inc'][:,u])
    sigSFdec_u = np.std(avg_act['sf_dec'][:,u])

    sf_stim_ind[u] = np.abs(SFinc_u - SFdec_u)

    SLinc_u = np.mean(avg_act['sl_inc'][:,u])
    SLdec_u = np.mean(avg_act['sl_dec'][:,u])
    sigSLinc_u = np.std(avg_act['sl_inc'][:,u])
    sigSLdec_u = np.std(avg_act['sl_dec'][:,u])
    
    sl_stim_ind[u] = np.abs(SLinc_u - SLdec_u)

plt.figure()
plt.plot(np.arange(1,N_rec+1), belief_ind, 'o', label='belief index', color='k')
plt.plot(np.arange(1,N_rec+1), sf_stim_ind, 'o', label='SF index', color='c')
plt.plot(np.arange(1,N_rec+1), sl_stim_ind,'o', label='SL index', color='m')
plt.xlabel('unit')
plt.legend()
plt.show()
#plt.savefig('./Figures/folder/indices_units')

plt.figure()
plt.scatter(belief_ind, sf_stim_ind)
plt.xlabel('belief index')
plt.ylabel('SF index')
#plt.savefig('./Figures/folder/BI_SFI')
#%%
#----sort recurrent weight matrix (maintaing task and perception modules)------------------
'''
# by index:
stim_ind = np.sum((sf_stim_ind, sl_stim_ind), axis=0)
perc_sort1 = np.flip(np.argsort(stim_ind[:40].copy()))
task_sort1 = np.flip(np.argsort(belief_ind[40:80].copy()))

sort_ind1 = np.concatenate((perc_sort1, task_sort1))
W_rec_sort1 = W_rec[:,sort_ind1].copy()
W_rec_sort1 = W_rec_sort1[sort_ind1,:].copy()
vf.plot_weights(W_rec_sort1, title='Recurrent weights (sorted by response)')
'''
# by weights:
sort_by = np.mean(W_rec, axis=0)
perc_sort2 = np.flip(np.argsort(sort_by[:40].copy())) 
task_sort2 = np.flip(np.argsort(sort_by[40:80].copy()))+40
perc_sort_2i = np.argsort(sort_by[80:90].copy())+80
task_sort_2i = np.argsort(sort_by[90:].copy())+90

sort_ind2 = np.concatenate((perc_sort2, task_sort2, perc_sort_2i, task_sort_2i))
W_rec_sort2 = W_rec[:,sort_ind2].copy()
W_rec_sort2 = W_rec_sort2[sort_ind2,:].copy()
vf.plot_weights(W_rec_sort2, title='Recurrent weights')