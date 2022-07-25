#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 12:03:33 2021

@author: Sol
"""

import scipy.io
import pandas as pd
import numpy as np

dat = scipy.io.loadmat('./mat_files/reduced_main_FO20190401.mat')
#1271 trials
#%% timestamps

ts = dat['ts_events']

delay = np.array(ts[5]-ts[4],dtype='float64')
delay = delay[~np.isnan(delay)]
print('avg delay: ', np.mean(delay))

godelay = np.array(ts[7]-ts[6],dtype='float64')
godelay = godelay[~np.isnan(godelay)]
print('avg go delay: ', np.mean(godelay))

choicetime = np.array(ts[8]-ts[7],dtype='float64')
choicetime = choicetime[~np.isnan(choicetime)]
print('avg choice time:', np.mean(choicetime))

beforestim = np.array(ts[3]-ts[0],dtype='float64')
beforestim = beforestim[~np.isnan(beforestim)]
print('avg before stim: ', np.mean(beforestim))

intertrial = ts[0][~np.isnan(ts[0])]
print('avg intertrial: ', np.mean(intertrial))

#%%
print(dat.keys())

N_t = dat['tag_block'][0].shape[0]//2
task = np.zeros(N_t, dtype=int)
sf1 = np.zeros(N_t)
sf2 = np.zeros(N_t)
dsf = np.zeros(N_t)
sl1 = np.zeros(N_t)
sl2 = np.zeros(N_t)
dsl = np.zeros(N_t)
choice = np.zeros(N_t)
correct = np.zeros(N_t)
err = np.zeros(N_t)
chosen_task = np.zeros(N_t)
task_err = np.zeros(N_t)
perc_err = np.zeros(N_t)

delay = np.array(ts[5]-ts[4],dtype='float64')

for i in range(0,N_t):
    task[i] = dat['tag_block'][0][2*i].copy()
    
    sf1[i] = dat['tag_sf'][0][2*i].copy()
    sf2[i] = dat['tag_sf'][0][2*i+1].copy()
    dsf[i] = sf2[i]-sf1[i]

    sl1[i] = dat['tag_sl'][0][2*i].copy()
    sl2[i] = dat['tag_sl'][0][2*i+1].copy()
    dsl[i] = sl2[i]-sl1[i]
    
    choice[i] = dat['tag_choice'][0][i]
    
    if (choice[i]==1 and dsf[i]<0)|(choice[i]==2 and dsf[i]>0)|(choice[i]==3 and dsl[i]>0)|(choice[i]==4 and dsl[i]<0):
        perc_err[i]=1
    
    if task[i]==1:
        if dsl[i]<0:
            correct[i]=3
        elif dsl[i]>0:
            correct[i]=4
    else:
        if dsf[i]<0:
            correct[i]=2
        elif dsf[i]>0:
            correct[i]=1

    if choice[i] != correct[i]:
        err[i]=1
    
    if choice[i]==1 or choice[i]==2:
        chosen_task[i]=2
    else:
        chosen_task[i]=1
    
    if task[i] != chosen_task[i]:
        task_err[i]=1
        
dat_df = pd.DataFrame()
dat_df['task']=task
dat_df['sf1']=sf1
dat_df['sf2']=sf2
dat_df['sl1']=sl1
dat_df['sl2']=sl2
dat_df['dsf']=dsf
dat_df['dsl']=dsl
dat_df['choice']=choice
dat_df['correct']=correct
dat_df['err']=err
dat_df['chosen_task']=chosen_task
dat_df['task_err']=task_err
dat_df['perc_err']=perc_err
dat_df['delay']=delay

print(1 - np.count_nonzero(dat_df.loc[dat_df['delay']<0.35, 'err'])/np.count_nonzero(dat_df['delay']<0.35))
print(1 - np.count_nonzero(dat_df.loc[dat_df['delay']>0.45, 'err'])/np.count_nonzero(dat_df['delay']>0.45))
print(1 - np.count_nonzero(dat_df.loc[dat_df['delay']<0.35, 'perc_err'])/np.count_nonzero(dat_df['delay']<0.35))
print(1 - np.count_nonzero(dat_df.loc[dat_df['delay']>0.45, 'perc_err'])/np.count_nonzero(dat_df['delay']>0.45))
print(1 - np.count_nonzero(dat_df.loc[dat_df['delay']<0.35, 'task_err'])/np.count_nonzero(dat_df['delay']<0.35))
print(1 - np.count_nonzero(dat_df.loc[dat_df['delay']>0.45, 'task_err'])/np.count_nonzero(dat_df['delay']>0.45))
#%%

#stimuli distributions

import matplotlib.pyplot as plt
plt.figure()
plt.hist(sf1, label='sf1')
plt.hist(sf2, label='sf2')
plt.legend()

plt.figure()
plt.hist(sl1, label='sl1')
plt.hist(sl2, label='sl2')
plt.legend()

plt.figure()
plt.hist(dsf, label='dsf')
plt.legend()

plt.figure()
plt.hist(dsl, label='dsl')
plt.legend()
