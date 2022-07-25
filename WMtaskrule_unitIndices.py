#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 12:14:01 2021

@author: Sol
"""

import os
os.chdir('/Users/Sol/Desktop/CohenLab/DynamicTaskPerceptionProject')
import numpy as np
import pickle
import pandas as pd
from numpy.random import default_rng
rng=default_rng()
import matplotlib.pyplot as plt
#%%
name = 'vardelay100/vardelay100'  #Ablation/vardelay100_ablateConstantIndex'

load_path = './WM_taskrule/vardelay100/Weights/vardelay100.npz'   #./WM_taskrule/Ablation/vardelay100_ablateConstantIndex.npz'
weights = np.load(load_path)

# load testing files generated from PCAvis
modelout_10 = pickle.load(open('./WM_taskrule/'+name+'_1on0off_modeloutput.pickle','rb'))
sv_10 = pickle.load(open('./WM_taskrule/'+name+'_1on0off_statevar.pickle','rb'))
fr_10 = np.maximum(sv_10,0)
tlabs_10 = pd.read_csv('./WM_taskrule/'+name+'_1on0off_triallabels.csv')

"""modelout_46 = pickle.load(open('./WM_taskrule/'+name+'_.6on.4off_modeloutput.pickle','rb'))
sv_46 = pickle.load(open('./WM_taskrule/'+name+'_.6on.4off_statevar.pickle','rb'))
fr_46 = np.maximum(sv_46,0)
tlabs_46 = pd.read_csv('./WM_taskrule/'+name+'_.6on.4off_triallabels.csv')

modelout_55 = pickle.load(open('./WM_taskrule/'+name+'_.5on.5off_modeloutput.pickle','rb'))
sv_55 = pickle.load(open('./WM_taskrule/'+name+'_.5on.5off_statevar.pickle','rb'))
fr_55 = np.maximum(sv_55,0)
tlabs_55 = pd.read_csv('./WM_taskrule/'+name+'_.5on.5off_triallabels.csv')"""

N_u = fr_10.shape[2]

#%% task index vs. time, 100 ms bins, task rule [1,0]

tstep = 10
T_tr = fr_10.shape[1]
taskA_10 = tlabs_10[(tlabs_10['model_choice']==0)|(tlabs_10['model_choice']==1)].index
taskB_10 = tlabs_10[(tlabs_10['model_choice']==2)|(tlabs_10['model_choice']==3)].index
taskindmat = np.zeros((T_tr//tstep, N_u))

for t in range(T_tr//tstep):

    fr_10A = fr_10[taskA_10,t*tstep:(t+1)*tstep,:]
    avgfr_10A = np.mean(fr_10A.reshape((fr_10A.shape[0]*fr_10A.shape[1], fr_10A.shape[2])), axis=0)
    fr_10B = fr_10[taskB_10,t*tstep:(t+1)*tstep,:] # had an error in the time index before 2/24
    avgfr_10B = np.mean(fr_10B.reshape((fr_10B.shape[0]*fr_10B.shape[1], fr_10B.shape[2])), axis=0)
    
    for i in range(N_u):
        if (avgfr_10A[i] + avgfr_10B[i])>0.001: # avoid division by zero
            taskindmat[t,i] = (avgfr_10A[i] - avgfr_10B[i])/(avgfr_10A[i] + avgfr_10B[i])
        else:
            taskindmat[t,i] = np.nan

#summed = np.sum(taskindmat, axis=0)  # sort by overall task index
#sortinds = np.argsort(-summed)
avgfr_all = np.mean(fr_10.reshape((fr_10.shape[0]*fr_10.shape[1], fr_10.shape[2])), axis=0)
sortinds = np.argsort(-(avgfr_all)) # sort by avg FR

plt.matshow(taskindmat[:,sortinds],cmap='coolwarm')
plt.xlabel('Units', fontsize=16)
plt.yticks(np.arange(T_tr//tstep,step=2),np.arange(1,T_tr//tstep+1,step=2),fontsize=10)
plt.ylabel('Time, 100 ms bins',fontsize=16)
plt.colorbar()
#plt.savefig('./WM_taskrule/Figs_unitIndex/'+name+'TaskIndvsTime_sortFR.png',dpi=200,bbox_inches='tight')
#%% add avg FR vs unit subplot


# Create two subplots and unpack the output array immediately
ax1 = plt.subplot2grid((3,1), (0,0), rowspan=2)
ax2 = plt.subplot2grid((3,1), (2,0))
mat = ax1.matshow(taskindmat[:,sortinds],cmap='coolwarm')
cbar = plt.colorbar(mat, ax=ax1, orientation='horizontal', shrink=0.5)
cbar.ax.set_title('Task index', fontsize=10)
ax1.set_xlabel('Units')
ax1.xaxis.set_label_position('top') 
ax1.set_ylabel('Time (100 ms)')
ax2.bar(x=np.arange(0,N_u), height=avgfr_all[sortinds])
ax2.set_ylabel('Avg. Firing Rate')
ax2.set_xlabel('Units')
#plt.savefig('./WM_taskrule/vardelay100/Figures/vardelay100_TaskIndvsTime_withavgFR.png',dpi=200,bbox_inches='tight')
#%% choice indices: task rule [1,0]
save = False # whether to save the figures or not
name = 'vardelay100'

#task A
choice0 = tlabs_10[(tlabs_10['model_choice']==0)].index
choice1 = tlabs_10[(tlabs_10['model_choice']==1)].index

fr_10_0 = fr_10[choice0,-50:,:]
avgfr_10_0 = np.mean(fr_10_0.reshape((fr_10_0.shape[0]*fr_10_0.shape[1],fr_10_0.shape[2])),axis=0)
fr_10_1 = fr_10[choice1,-50:,:]
avgfr_10_1 = np.mean(fr_10_1.reshape((fr_10_1.shape[0]*fr_10_1.shape[1],fr_10_1.shape[2])),axis=0)

stim_ind10A = np.zeros(N_u)
for i in range(N_u):
    if (avgfr_10_0[i] + avgfr_10_1[i])>0.001: # avoid division by zero
        stim_ind10A[i] = (avgfr_10_0[i] - avgfr_10_1[i])/(avgfr_10_0[i] + avgfr_10_1[i])
    else:
        stim_ind10A[i] = np.nan

plt.hist(stim_ind10A)
plt.xlabel('Choice index: '+r'$\frac{<FR>_{a_1>a_2}-<FR>_{a_1<a_2}}{<FR>_{a_1>a_2}+<FR>_{a_1<a_2}}$',fontsize=12)
plt.title('Unit choice indices for task rule [1,0], task A')
if save==True:
    plt.savefig('./WM_taskrule/Figs_unitIndex/'+name+'choiceInd_10A.png',dpi=200,bbox_inches='tight')

#taskB
choice2 = tlabs_10[(tlabs_10['model_choice']==2)].index
choice3 = tlabs_10[(tlabs_10['model_choice']==3)].index

fr_10_2 = fr_10[choice2,-50:,:]
avgfr_10_2 = np.mean(fr_10_2.reshape((fr_10_2.shape[0]*fr_10_2.shape[1],fr_10_2.shape[2])),axis=0)
fr_10_3 = fr_10[choice3,-50:,:]
avgfr_10_3 = np.mean(fr_10_3.reshape((fr_10_3.shape[0]*fr_10_3.shape[1],fr_10_3.shape[2])),axis=0)

stim_ind10B = np.zeros(N_u)
for i in range(N_u):
    if (avgfr_10_2[i] + avgfr_10_3[i])>0.001: # avoid division by zero
        stim_ind10B[i] = (avgfr_10_2[i] - avgfr_10_3[i])/(avgfr_10_2[i] + avgfr_10_3[i])
    else:
        stim_ind10B[i] = np.nan

plt.figure()
plt.hist(stim_ind10B)
plt.xlabel('Choice index: '+r'$\frac{<FR>_{b_1>b_2}-<FR>_{b_1<b_2}}{<FR>_{b_1>b_2}+<FR>_{b_1<b_2}}$',fontsize=12)
plt.title('Unit choice indices for task rule [1,0], task B')
if save==True:
    plt.savefig('./WM_taskrule/Figs_unitIndex/'+name+'choiceInd_10B.png',dpi=200,bbox_inches='tight')
#%% choice index for each unit (sorted)

sorted_choiceA = stim_ind10A[sortinds]
sorted_choiceB = stim_ind10B[sortinds]

plt.figure()
ax1 = plt.subplot2grid((2,1), (0,0))
ax2 = plt.subplot2grid((2,1), (1,0))

ax1.bar(x=np.arange(100), height=sorted_choiceA[:100])
ax1.set_ylabel('Choice index, task A')
ax2.bar(x=np.arange(100), height=sorted_choiceB[:100])
ax2.set_xlabel('Units')
ax2.set_ylabel('Choice index, task B')

#plt.savefig('./WM_taskrule/vardelay100/Figures/vardelay100_ChoiceInds_sortedunits.png',dpi=200,bbox_inches='tight')

#%% calculate task variance and fractional task variance (Yang 2019)

taskA_10 = tlabs_10[(tlabs_10['model_choice']==0)|(tlabs_10['model_choice']==1)].index
taskB_10 = tlabs_10[(tlabs_10['model_choice']==2)|(tlabs_10['model_choice']==3)].index

fr_10A = fr_10[taskA_10,:,:]
var_10A = np.var(fr_10A,axis=0)
TV_10A = np.mean(var_10A,axis=0)

fr_10B = fr_10[taskB_10,:,:]
var_10B = np.var(fr_10B,axis=0)
TV_10B = np.mean(var_10B,axis=0)

FTV10 = np.zeros(N_u)
for i in range(N_u):
    if (TV_10A[i] + TV_10B[i])>0.001: # avoid division by zero
        FTV10[i] = (TV_10A[i] - TV_10B[i])/(TV_10A[i] + TV_10B[i])
    else:
        FTV10[i] = np.nan

plt.hist(FTV10)
plt.xlabel('Fractional Task Variance: '+r'$\frac{TV_A-TV_B}{TV_A+TV_B}$',fontsize=12)
plt.title('Unit fractional task variance distribution for task rule [1,0]')

#plt.savefig('./WM_taskrule/FTV_10.png',dpi=200,bbox_inches='tight')
#%% calculate task index: task rule [1,0]

taskA_10 = tlabs_10[(tlabs_10['model_choice']==0)|(tlabs_10['model_choice']==1)].index
taskB_10 = tlabs_10[(tlabs_10['model_choice']==2)|(tlabs_10['model_choice']==3)].index

fr_10A = fr_10[taskA_10,:,:]
avgfr_10A = np.mean(fr_10A.reshape((fr_10A.shape[0]*fr_10A.shape[1], fr_10A.shape[2])), axis=0)
fr_10B = fr_10[taskB_10,:,:]
avgfr_10B = np.mean(fr_10B.reshape((fr_10B.shape[0]*fr_10B.shape[1], fr_10B.shape[2])), axis=0)

task_index10 = np.zeros(N_u)
for i in range(N_u):
    if (avgfr_10A[i] + avgfr_10B[i])>0.001: # avoid division by zero
        task_index10[i] = (avgfr_10A[i] - avgfr_10B[i])/(avgfr_10A[i] + avgfr_10B[i])
    else:
        task_index10[i] = np.nan

plt.hist(task_index10)
plt.yticks(np.arange(0,20,1))
plt.xlabel('Task index: '+r'$\frac{<FR>_A-<FR>_B}{<FR>_A+<FR>_B}$',fontsize=12)
plt.title('Unit task indices for task rule [1,0]')
plt.savefig('./WM_taskrule/'+name+'taskInd_10.png',dpi=200,bbox_inches='tight')
#%% calculate PRE-STIM task index: task rule [1,0]

#t_delay1=100
t_delay1 = 50

fr_10Ae = fr_10[taskA_10,:t_delay1,:]
avgfr_10Ae = np.mean(fr_10Ae.reshape((fr_10Ae.shape[0]*fr_10Ae.shape[1], fr_10Ae.shape[2])), axis=0)
fr_10Be = fr_10[taskB_10,:t_delay1,:]
avgfr_10Be = np.mean(fr_10Be.reshape((fr_10Be.shape[0]*fr_10Be.shape[1], fr_10Be.shape[2])), axis=0)

task_index10e = np.zeros(N_u)
for i in range(N_u):
    if (avgfr_10Ae[i] + avgfr_10Be[i])>0.001: # avoid division by zero
        task_index10e[i] = (avgfr_10Ae[i] - avgfr_10Be[i])/(avgfr_10Ae[i] + avgfr_10Be[i])
    else:
        task_index10e[i] = np.nan

plt.hist(task_index10e)
plt.yticks(np.arange(0,10,1))
plt.xlabel('Task index, pre-stim: '+r'$\frac{<FR>_A-<FR>_B}{<FR>_A+<FR>_B}$',fontsize=12)
plt.title('Unit task indices for task rule [1,0]')
#plt.savefig('./WM_taskrule/'+name+'taskInd_10prestim.png',dpi=200,bbox_inches='tight')

#%% calculate INTER-STIM DELAY task index: task rule [1,0]

fr_10Ad = fr_10[taskA_10,120:160,:]
avgfr_10Ad = np.mean(fr_10Ad.reshape((fr_10Ad.shape[0]*fr_10Ad.shape[1], fr_10Ad.shape[2])), axis=0)
fr_10Bd = fr_10[taskB_10,120:160,:]
avgfr_10Bd = np.mean(fr_10Bd.reshape((fr_10Bd.shape[0]*fr_10Bd.shape[1], fr_10Bd.shape[2])), axis=0)

task_index10d = np.zeros(N_u)
for i in range(N_u):
    if (avgfr_10Ad[i] + avgfr_10Bd[i])>0.001: # avoid division by zero
        task_index10d[i] = (avgfr_10Ad[i] - avgfr_10Bd[i])/(avgfr_10Ad[i] + avgfr_10Bd[i])
    else:
        task_index10d[i] = np.nan

plt.hist(task_index10d)
plt.yticks(np.arange(0,21,2))
plt.xlabel('Task index, delay period: '+r'$\frac{<FR>_A-<FR>_B}{<FR>_A+<FR>_B}$',fontsize=12)
plt.title('Unit task indices for task rule [1,0]')
#plt.savefig('./WM_taskrule/taskInd_10delay.png',dpi=200,bbox_inches='tight')
#%% calculate task index: task rule [0.6,0.4]

taskA_46 = tlabs_46[(tlabs_46['model_choice']==0)|(tlabs_46['model_choice']==1)].index
taskB_46 = tlabs_46[(tlabs_46['model_choice']==2)|(tlabs_46['model_choice']==3)].index

fr_46A = fr_46[taskA_46,:,:]
avgfr_46A = np.mean(fr_46A.reshape((fr_46A.shape[0]*fr_46A.shape[1], fr_46A.shape[2])), axis=0)
fr_46B = fr_46[taskB_46,:,:]
avgfr_46B = np.mean(fr_46B.reshape((fr_46B.shape[0]*fr_46B.shape[1], fr_46B.shape[2])), axis=0)

task_index46 = np.zeros(N_u)
for i in range(N_u):
    if (avgfr_46A[i] + avgfr_46B[i])>0.001: # avoid division by zero
        task_index46[i] = (avgfr_46A[i] - avgfr_46B[i])/(avgfr_46A[i] + avgfr_46B[i])
    else:
        task_index46[i] = np.nan
        
plt.hist(task_index46)
plt.yticks(np.arange(0,50,5))
plt.xlabel('Task index: '+r'$\frac{<FR>_A-<FR>_B}{<FR>_A+<FR>_B}$',fontsize=12)
plt.title('Unit task indices for task rule [0.6,0.4]')
#plt.savefig('./WM_taskrule/taskInd_46.png',dpi=200,bbox_inches='tight')
#%% calculate task index: task rule [0.5,0.5]

taskA_55 = tlabs_55[(tlabs_55['model_choice']==0)|(tlabs_55['model_choice']==1)].index
taskB_55 = tlabs_55[(tlabs_55['model_choice']==2)|(tlabs_55['model_choice']==3)].index

fr_55A = fr_55[taskA_55,:,:]
avgfr_55A = np.mean(fr_55A.reshape((fr_55A.shape[0]*fr_55A.shape[1], fr_55A.shape[2])), axis=0)
fr_55B = fr_55[taskB_55,:,:]
avgfr_55B = np.mean(fr_55B.reshape((fr_55B.shape[0]*fr_55B.shape[1], fr_55B.shape[2])), axis=0)

task_index55 = np.zeros(N_u)
for i in range(N_u):
    if (avgfr_55A[i] + avgfr_55B[i])>0.001: # avoid division by zero
        task_index55[i] = (avgfr_55A[i] - avgfr_55B[i])/(avgfr_55A[i] + avgfr_55B[i])
    else:
        task_index55[i] = np.nan

plt.hist(task_index55)
plt.yticks(np.arange(0,55,5))
plt.xlabel('Task index: '+r'$\frac{<FR>_A-<FR>_B}{<FR>_A+<FR>_B}$',fontsize=12)
plt.title('Unit task indices for task rule [0.5,0.5]')
#plt.savefig('./WM_taskrule/taskInd_55.png',dpi=200,bbox_inches='tight')


#%% save to csv
df = pd.DataFrame()
df['task10']=task_index10
df['task10_prestim']=task_index10e
df['task64']=task_index46
df['task55']=task_index55
df['choice10A']=stim_ind10A
df['choice10B']=stim_ind10B

#df.to_csv('./WM_taskrule/unitIndices_model3b_1000.csv',index=False)
#%% append a new column(s)
df = pd.read_csv('./WM_taskrule/unitIndices_model3b_1000.csv')

df['TV10A']=TV_10A
df['TV10B']=TV_10B
df['avgFR_10A']=avgfr_10A
df['avgFR_10B']=avgfr_10B

df.to_csv('./WM_taskrule/unitIndices_model3b_1000.csv',index=False)
#%%
df = pd.read_csv('./WM_taskrule/unitIndices_model3b_1000.csv')

import visfunctions as vf

unit_labels = np.arange(1,101)
#unit_colors = plt.cm.jet(np.linspace(0,1,100))

lg_task10 = df[df['task10']>0.75].index
vf.plot_activities(sv_10[:,:,lg_task10],taskA_10[0],10,colormap=plt.cm.tab20,label=unit_labels[lg_task10])

lg_task10e = df[df['task10_prestim']>0.75].index
vf.plot_activities(sv_10[:,:,lg_task10e],taskA_10[0],10,colormap=plt.cm.tab20,label=unit_labels[lg_task10e])

lg_task10d = df[df['task10_delay']>0.9].index
vf.plot_activities(sv_10[:,:,lg_task10d],taskA_10[0],10,colormap=plt.cm.tab20,label=unit_labels[lg_task10d])

outlabels = ['a1>a2', 'a1<a2', 'b1>b2', 'b1<b2', 'fixation']
vf.plot_activities(modelout_10, taskA_10[0], dt=10, colormap=plt.cm.tab10,label=outlabels)
#%%
plt.scatter(df['task10'],df['task10_delay'])

vf.plot_weights(weights['W_out'])
#%%
taskind_FTV_diff = df['task10']-df['FTV10']

unit_labels = np.arange(1,101)

plt.scatter(df['TV10A'],df['TV10B'])
for i in range(100):
    plt.annotate(unit_labels[i], (df.loc[i,'TV10A'], df.loc[i,'TV10B']))

plt.figure()
plt.scatter(df['avgFR_10A'],df['avgFR_10B'])
for i in range(100):
    plt.annotate(unit_labels[i], (df.loc[i,'avgFR_10A'], df.loc[i,'avgFR_10B']))

plt.figure()
plt.scatter(df['task10'],df['FTV10'])
for i in range(100):
    plt.annotate(unit_labels[i], (df.loc[i,'task10'], df.loc[i,'FTV10']))