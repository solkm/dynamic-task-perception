#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 14:58:54 2021

@author: Sol
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns
from model_functions import consecutive_errors, accuracy_nAfterError, task_switches, task_streaks, get_tasks
import pandas as pd
#%%
#length of tParams is 155958

tParams = scipy.io.loadmat('./mat_files/tParams.mat')
#print('dataset: normal')

tParams_Long = scipy.io.loadmat('./mat_files/tParams_Long.mat')
#print('dataset: low volatility')

tParams_Short = scipy.io.loadmat('./mat_files/tParams_Short.mat')
#print('dataset: high volatility')

tParams_Easy = scipy.io.loadmat('./mat_files/tParams_Easy.mat')
#print('dataset: easy perception')

tParams_Hard = scipy.io.loadmat('./mat_files/tParams_Hard.mat')
#print('dataset: difficult perception')

tParams_LongHard = scipy.io.loadmat('./mat_files/tParams_LongHard.mat')
#print('dataset: low volatility, difficult perception')

tParams_ShortEasy = scipy.io.loadmat('./mat_files/tParams_ShortEasy.mat')
#print('dataset: high volatility, easy perception')
#%%
n=20
acc_type='perceptual'
acc_n = accuracy_nAfterError(tParams, n, acc_type)
print('normal ',acc_n)
acc_l = accuracy_nAfterError(tParams_Long, n, acc_type)
print('long ',acc_l)
acc_s = accuracy_nAfterError(tParams_Short, n, acc_type)
print('short ',acc_s)
acc_e = accuracy_nAfterError(tParams_Easy, n, acc_type)
print('easy ',acc_e)
acc_h = accuracy_nAfterError(tParams_Hard, n, acc_type)
print('hard ',acc_h)
acc_lh = accuracy_nAfterError(tParams_LongHard, n, acc_type)
print('long hard ',acc_lh)
acc_se = accuracy_nAfterError(tParams_ShortEasy, n, acc_type)
print('short easy ',acc_se)

x = np.arange(n)
plt.plot(x+1,acc_n,label='normal')
plt.plot(x+1,acc_l,label='long')
plt.plot(x+1,acc_s,label='short')
plt.plot(x+1,acc_e,label='easy')
plt.plot(x+1,acc_h,label='hard')
plt.plot(x+1,acc_lh,label='long hard')
plt.plot(x+1,acc_se,label='short easy')
plt.legend()
plt.xlabel('Trials after an error')
plt.ylabel('Perceptual Accuracy')
#plt.savefig('./Figures/tParams_Plots/overallAccuracy10AfterError.png',dpi=200)
#%%
dat = tParams_ShortEasy

choice = dat['r_CHOICE'][0].copy()
correct = dat['r_CORRECT'][0].copy()
task = dat['r_TASK'][0].copy()

print('total number of trials: ',len(task))

consErrs, accuracy = consecutive_errors(choice, correct)
print('Monkey error percentage: ', 1-accuracy)
print('Max number of consecutive errors: ', np.max(consErrs))
print('Avg error streak: ', np.mean(consErrs))

switch, numSwitches = task_switches(task)
print('There are ',numSwitches,' task switches in the dataset')

taskStreaks = task_streaks(task)
print('Avg task streak:', np.mean(taskStreaks))

monkey_task = get_tasks(choice)

monkey_switch, numMonkeySwitches = task_switches(monkey_task)
print('The monkey switches tasks ',numMonkeySwitches,' times in the dataset')

monkeyTaskStreaks = task_streaks(monkey_task)
print('Avg monkey task streak:', np.mean(monkeyTaskStreaks))

sns.displot(monkeyTaskStreaks, binwidth=5)
plt.xlabel('Monkey task streak length')
plt.ylabel('Count')
plt.title('Monkey task streaks, high volatility and easy perception')
#plt.savefig('./Figures/monkeyTaskStreaks_ShortEasy')
plt.show()

sns.displot(taskStreaks, binwidth=5)
plt.xlabel('Actual task streak length')
plt.ylabel('Count')
plt.title('Actual task streaks, high volatility and rasy perception')
#plt.savefig('./Figures/taskStreaks_ShortEasy')
plt.show()

#%%

tParams_df = pd.read_csv('./tParams_df.csv')

choice = np.array(tParams_df['choice'])
correct = np.array(tParams_df['correct'])
consErrs, accuracy, long = consecutive_errors(choice, correct, 20)

#%%
df = pd.read_csv('./tParams_full.csv')
N_tot = df.shape[0]
err_inds = df[df['err']==1].index.to_numpy()
ae_inds = err_inds[err_inds < N_tot-1] + 1
ndB_ae = df.loc[ae_inds,'ndBelief']
ac_inds = np.flatnonzero(np.isin(np.arange(N_tot),ae_inds,invert=True))
ndB_ac = df.loc[ac_inds,'ndBelief']
print(np.mean(np.abs(ndB_ae)))
print(np.mean(np.abs(ndB_ac)))

print(np.min(df['ndBelief']), np.max(df['ndBelief']))



plt.hist(np.abs(df['ndBelief']), bins=20, fill=False, hatch='///', edgecolor='k')
plt.hist(np.abs(ndB_ac), label='after correct', bins=20, fill=False, hatch='///', edgecolor='b')
plt.hist(np.abs(ndB_ae), label='after error', bins=20, fill=False, hatch='///', edgecolor='r')
plt.legend()
plt.xlabel('normalized belief strength (absolute value)')
#plt.savefig('./Figures/tParams_Plots/ndBeliefAV_hist.png', dpi=200)
#%%


#%%
'''
dataset: normal
total number of trials: 155958
Monkey error percentage:  0.2980161325485067
Max number of consecutive errors: 44
Avg error streak: 1.9555265704548324
There are  4155  task switches in the dataset
Max number of trials without a task switch: 321
Min number of trials without a task switch: 1
Avg task streak: 37.620570012391575
The monkey switches tasks  12408  times in the dataset
Max number of trials without a monkey task switch: 359
Min number of trials without a monkey task switch: 1
Avg monkey task streak: 17.96833145575166
Monkey's accuracy, 1 after correct:  0.7829101205699671

dataset: low volatility
total number of trials: 5542
Monkey error percentage:  0.2971851317214002
Max number of consecutive errors: 25
Avg error streak: 1.8985005767012688
There are  103  task switches in the dataset
Max number of trials without a task switch: 294
Min number of trials without a task switch: 2
Avg task streak: 52.271844660194176
The monkey switches tasks  466  times in the dataset
Max number of trials without a monkey task switch: 145
Min number of trials without a monkey task switch: 1
Avg monkey task streak: 17.033557046979865
Monkey's accuracy, 1 after correct:  0.7774069319640565

dataset: high volatility
total number of trials: 4367
Monkey error percentage:  0.3260819784749256
Max number of consecutive errors: 40
Avg error streak: 2.218068535825545
There are  193  task switches in the dataset
Max number of trials without a task switch: 120
Min number of trials without a task switch: 1
Avg task streak: 22.721311475409838
The monkey switches tasks  420  times in the dataset
Max number of trials without a monkey task switch: 89
Min number of trials without a monkey task switch: 1
Avg monkey task streak: 13.416382252559726
Monkey's accuracy, 1 after correct:  0.7821210061182868

dataset: easy perception
total number of trials: 4486
Monkey error percentage:  0.21466785555060186
Max number of consecutive errors: 14
Avg error streak: 1.924
There are  105  task switches in the dataset
Max number of trials without a task switch: 214
Min number of trials without a task switch: 1
Avg task streak: 41.65346534653465
The monkey switches tasks  288  times in the dataset
Max number of trials without a monkey task switch: 118
Min number of trials without a monkey task switch: 1
Avg monkey task streak: 20.368932038834952
Monkey's accuracy, 1 after correct:  0.8577916548396253

dataset: difficult perception
total number of trials:  3070
Monkey error percentage:  0.4006514657980456
Max number of consecutive errors: 30
Avg error streak: 2.232727272727273
There are  85  task switches in the dataset
Max number of trials without a task switch: 431
Min number of trials without a task switch: 2
Avg task streak: 36.207317073170735
The monkey switches tasks  294  times in the dataset
Max number of trials without a monkey task switch: 116
Min number of trials without a monkey task switch: 1
Avg monkey task streak: 13.815
Monkey's accuracy, 1 after correct:  0.7010869565217391
'''