#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 10:34:43 2021

@author: Sol
"""
import os
os.chdir('/Users/Sol/Desktop/CohenLab/DynamicTaskPerceptionProject')
import numpy as np
from sklearn.decomposition import PCA
from FourAFC_taskModels import WM_TaskRule_varDelay
from psychrnn.backend.models.basic import Basic
import pickle
import pandas as pd
import model_functions as mf
from numpy.random import default_rng
rng=default_rng()
import matplotlib
matplotlib.rcdefaults()
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
import visfunctions as vf
def proj_ortho_basis(basis_mat, vec):
    v = vec
    dim = basis_mat.shape[0]
    x = np.zeros(dim)
    for i in range(dim):
        b = basis_mat[i]
        p = np.dot(v, b) / np.sqrt(np.dot(b, b))
        x[i] = p
    
    return x

N_rec = 100
in_noise = 0.5
rec_noise = 0.1
name = 'Ablation/vardelay100_ablate4HighestTaskInput'
load_path = './WM_taskrule/'+name+'.npz'

on_rule = 1.0
off_rule = 0.0
#%% test and save test data
N_testbatch = 10000
#ON_OFF = [[1,0],[0.8,0.2],[0.7,0.3],[0.6,0.4],[0.5,0.5]]
#rules = ['1on0off','.8on.2off','.7on.3off','.6on.4off','.5on.5off']
ON_OFF = [[1,0],[0.8,0.2],[0.6,0.4]]
rules = ['1on0off','.8on.2off','.6on.4off']

for i in range(len(rules)):
    modeltask = WM_TaskRule_varDelay(N_batch=N_testbatch, in_noise=in_noise, on_rule = ON_OFF[i][0], off_rule = ON_OFF[i][1])
    network_params = modeltask.get_task_params()
    network_params['name'] = name
    network_params['N_rec'] = N_rec
    network_params['rec_noise'] = rec_noise
    network_params['load_weights_path'] = load_path
    model = Basic(network_params)
    
    test_inputs, target_output, mask, trial_params = modeltask.get_trial_batch()
    model_output, state_var = model.test(test_inputs)
    
    savename = './WM_taskrule/' + name + '_' + rules[i]
    
    if savename is not None:
        savefile = open(savename+'_modeloutput.pickle','wb')
        pickle.dump(model_output, savefile, protocol=4)
        savefile.close()
    
        savefile = open(savename+'_trialparams.pickle','wb')
        pickle.dump(trial_params, savefile, protocol=4)
        savefile.close()
        
        savefile = open(savename+'_statevar.pickle','wb')
        pickle.dump(state_var, savefile, protocol=4)
        savefile.close()
        
    model.destruct()
#%% label tested trials

ON_OFF = [[1,0],[0.8,0.2],[0.6,0.4]]
rules = ['1on0off','.8on.2off','.6on.4off']


for r in range(len(rules)):
    tparams = pickle.load(open('./WM_taskrule/'+name+'_'+rules[r]+'_trialparams.pickle','rb'))
    modelout = pickle.load(open('./WM_taskrule/'+name+'_'+rules[r]+'_modeloutput.pickle','rb'))
    choice = np.argmax(modelout[:,-1,:], axis=1)
    
    trial_labels = pd.DataFrame(columns=['da','db','task','rulein_0','rulein_1','targ','model_choice','err','perc_err','task_err'])
    
    for i in range(tparams.shape[0]):
        trial_labels.loc[i,'da'] = tparams[i]['a2'] - tparams[i]['a1']
        trial_labels.loc[i,'db'] = tparams[i]['b2'] - tparams[i]['b1']
        trial_labels.loc[i,'task'] = tparams[i]['task']
        trial_labels.loc[i,'rulein_0'] = tparams[i]['taskrule'][0]
        trial_labels.loc[i,'rulein_1'] = tparams[i]['taskrule'][1]
        trial_labels.loc[i,'targ'] = tparams[i]['targ']
        trial_labels.loc[i,'model_choice'] = choice[i]
        
        if (trial_labels.loc[i,'task']==0 and np.abs(trial_labels.loc[i,'da'])<0.2) or (trial_labels.loc[i,'task']==1 and np.abs(trial_labels.loc[i,'db'])<0.2):
            trial_labels.loc[i,'perc_diff'] = 5
        elif (trial_labels.loc[i,'task']==0 and np.abs(trial_labels.loc[i,'da'])<0.4) or (trial_labels.loc[i,'task']==1 and np.abs(trial_labels.loc[i,'db'])<0.4):
            trial_labels.loc[i,'perc_diff'] = 4
        elif (trial_labels.loc[i,'task']==0 and np.abs(trial_labels.loc[i,'da'])<0.6) or (trial_labels.loc[i,'task']==1 and np.abs(trial_labels.loc[i,'db'])<0.6):
            trial_labels.loc[i,'perc_diff'] = 3
        elif (trial_labels.loc[i,'task']==0 and np.abs(trial_labels.loc[i,'da'])<0.8) or (trial_labels.loc[i,'task']==1 and np.abs(trial_labels.loc[i,'db'])<0.8):
            trial_labels.loc[i,'perc_diff'] = 2
        else:
            trial_labels.loc[i,'perc_diff'] = 1
    
    task_acc, perc_acc, overall_acc, noresponse, taskerr_trials, percerr_trials, nr_trials = mf.get_Accuracies(modelout, tparams)
    trial_labels.loc[:,'task_err'] = 0
    trial_labels.loc[:,'perc_err'] = 0
    trial_labels.loc[:,'err'] = 0
    trial_labels.loc[taskerr_trials,'task_err']=1
    trial_labels.loc[percerr_trials,'perc_err']=1
    trial_labels.loc[np.concatenate((taskerr_trials,percerr_trials)),'err']=1
    
    trial_labels.to_csv('./WM_taskrule/'+name+'_'+rules[r]+'_triallabels.csv',index=False)

#%% load test data

tparams10 = pickle.load(open('./WM_taskrule/'+name+'_1on0off_trialparams.pickle','rb'))
modelout10 = pickle.load(open('./WM_taskrule/'+name+'_1on0off_modeloutput.pickle','rb'))
choice10 = np.argmax(modelout10[:,-1,:], axis=1)
sv10 = pickle.load(open('./WM_taskrule/'+name+'_1on0off_statevar.pickle','rb'))
fr10 = np.maximum(sv10,0)
tlabs10 = pd.read_csv('./WM_taskrule/'+name+'_1on0off_triallabels.csv')

tparams82 = pickle.load(open('./WM_taskrule/'+name+'_.8on.2off_trialparams.pickle','rb'))
modelout82 = pickle.load(open('./WM_taskrule/'+name+'_.8on.2off_modeloutput.pickle','rb'))
choice82 = np.argmax(modelout82[:,-1,:], axis=1)
sv82 = pickle.load(open('./WM_taskrule/'+name+'_.8on.2off_statevar.pickle','rb'))
fr82 = np.maximum(sv82,0)
tlabs82 = pd.read_csv('./WM_taskrule/'+name+'_.8on.2off_triallabels.csv')
'''
tparams73 = pickle.load(open('./WM_taskrule/'+name+'_.7on.3off_trialparams.pickle','rb'))
modelout73 = pickle.load(open('./WM_taskrule/'+name+'_.7on.3off_modeloutput.pickle','rb'))
choice73 = np.argmax(modelout73[:,-1,:], axis=1)
sv73 = pickle.load(open('./WM_taskrule/'+name+'_.7on.3off_statevar.pickle','rb'))
fr73 = np.maximum(sv73,0)
tlabs73 = pd.read_csv('./WM_taskrule/'+name+'_.7on.3off_triallabels.csv')"""
'''
tparams64 = pickle.load(open('./WM_taskrule/'+name+'_.6on.4off_trialparams.pickle','rb'))
modelout64 = pickle.load(open('./WM_taskrule/'+name+'_.6on.4off_modeloutput.pickle','rb'))
choice64 = np.argmax(modelout64[:,-1,:], axis=1)
sv64 = pickle.load(open('./WM_taskrule/'+name+'_.6on.4off_statevar.pickle','rb'))
fr64 = np.maximum(sv64,0)
tlabs64 = pd.read_csv('./WM_taskrule/'+name+'_.6on.4off_triallabels.csv')
'''
tparams55 = pickle.load(open('./WM_taskrule/'+name+'_.5on.5off_trialparams.pickle','rb'))
modelout55 = pickle.load(open('./WM_taskrule/'+name+'_.5on.5off_modeloutput.pickle','rb'))
choice55 = np.argmax(modelout55[:,-1,:], axis=1)
sv55 = pickle.load(open('./WM_taskrule/'+name+'_.5on.5off_statevar.pickle','rb'))
fr55 = np.maximum(sv55,0)
tlabs55 = pd.read_csv('./WM_taskrule/'+name+'_.5on.5off_triallabels.csv')
'''
#%% PCA
# used the 1 on 0 off task rule for 11/4 and 11/16 PCA plots, concatenated 1,0 and .6,.4 for 11/15 plots
X = fr10.reshape((fr10.shape[0]*fr10.shape[1], fr10.shape[2]))
#X = np.concatenate((fr10.reshape((fr10.shape[0]*fr10.shape[1], fr10.shape[2])), fr64.reshape((fr64.shape[0]*fr64.shape[1], fr64.shape[2]))))
pca = PCA(n_components=3)
pca.fit_transform(X)
print(pca.explained_variance_ratio_)

#%% plot1: selection of trials to plot, compare choices and perceptual difficulties
tsteps = 180 #240

AVGFR = []
#for (tlabs,fr) in [(tlabs10,fr10),(tlabs82,fr82),(tlabs73,fr73),(tlabs64,fr64),(tlabs55,fr55)]:
#for (tlabs,fr) in [(tlabs10,fr10),(tlabs64,fr64)]:
for (tlabs,fr) in [(tlabs10,fr10),(tlabs82,fr82),(tlabs64,fr64)]:
    np.random.seed(100)
    # 5 perceptual difficulty levels per choice
    n2avg=20
    percdifftrials = np.zeros((4,5,n2avg), dtype='int') # rows: trial choice (0,1,2,3), columns: perceptual difficulty (1,2,3,4,5)
    for i in range(4):
        for j in range(5):
    #        percdifftrials[i,j,:] = rng.choice(tlabs[(tlabs['model_choice']==i) & (tlabs['err']==0) & (tlabs['perc_diff']==j+1)].index,n2avg,replace=False) #must be correct
            percdifftrials[i,j,:] = rng.choice(tlabs[(tlabs['model_choice']==i) & (tlabs['perc_diff']==j+1)].index,n2avg,replace=False) #need not be correct
    
    #avg the rates for each condition
    avg_fr = np.zeros((4,5,tsteps,N_rec))
    for i in range(4):
        for j in range(5):
            temp = np.mean(fr[percdifftrials[i,j,:],:,:], axis=0)
            avg_fr[i,j,:,:] = temp
    AVGFR.append(avg_fr)

#%% plot1: 3D visualization,compare choices and perceptual difficulties

#tspec = [100,120,160,180,210] # special timepoints: stim1 on 1000-1200, stim2 on 1600-1800, fixation off 2100
tspec = [50,70,110,130,145] # for average delay period vardelay
fig = plt.figure()
ax = fig.gca(projection='3d')
ts = fr10.shape[1]
n_cat = 5
dim=3
dlevs = [3] # difficulty levels to plot

cmins = np.linspace(0.3,1,len(AVGFR))

#for (avg_fr,rule,cmin) in [(AVGFR[0],'rule [1,0]',cmins[-1]),(AVGFR[1],'rule [0.8,0.2]',cmins[-2]),(AVGFR[2],'rule [0.7,0.3]',cmins[-3]),(AVGFR[3],'rule [0.6,0.4]',cmins[-4]),(AVGFR[4],'rule [0.5,0.5]',cmins[-5])]:
for (avg_fr,rule,cmin) in [(AVGFR[0],'rule [1,0]',cmins[-1]),(AVGFR[1],'rule [0.8,0.2]',cmins[-2]),(AVGFR[2],'rule [0.6,0.4]',cmins[-3])]:
#for (avg_fr,rule,cmin) in [(AVGFR[0],'rule [1,0]',cmins[-1]),(AVGFR[2],'rule [0.6,0.4]',cmins[-4])]:
#    '''#choice 0
    traj0 = np.zeros((n_cat, ts, dim))
    for i in range(n_cat):
        for t in range(ts):
            rates = avg_fr[0,i,t,:]
            p = proj_ortho_basis(pca.components_[:dim,:], rates)
            traj0[i, t,:] = p
    
    colors0 = itertools.cycle(plt.cm.Blues(np.linspace(cmin,1,len(dlevs))))
    for trial in dlevs: # select difficulties to plot
        color = next(colors0)
        ax.scatter(traj0[trial,0,0], traj0[trial,0,1], traj0[trial,0,2], zdir='z', color=color, label='pd ' + str(trial+1) +' choice 0 '+rule)
        ax.scatter(traj0[trial,:,0], traj0[trial,:,1], traj0[trial,:,2], zdir='z', color=color)
        ax.scatter(traj0[trial,tspec,0], traj0[trial,tspec,1], traj0[trial,tspec,2], zdir='z', color='k', marker='X', s=70)
        ax.scatter(traj0[trial,-1,0], traj0[trial,-1,1], traj0[trial,-1,2], zdir='z', color='k', s=80)
    
    #choice 1
    traj1 = np.zeros((n_cat, ts, dim))
    for i in range(n_cat):
        for t in range(ts):
            rates = avg_fr[1,i,t,:]
            p = proj_ortho_basis(pca.components_[:dim,:], rates)
            traj1[i, t,:] = p
    
    colors1 = itertools.cycle(plt.cm.Reds(np.linspace(cmin,1,len(dlevs)))) 
    for trial in dlevs: #select difficulties to plot
        color = next(colors1)
        ax.scatter(traj1[trial,0,0], traj1[trial,0,1], traj1[trial,0,2], zdir='z', color=color, label='pd ' + str(trial+1) +' choice 1 '+rule)
        ax.scatter(traj1[trial,:,0], traj1[trial,:,1], traj1[trial,:,2], zdir='z', color=color)
        ax.scatter(traj1[trial,tspec,0], traj1[trial,tspec,1], traj1[trial,tspec,2], zdir='z', color='k', marker='X', s=70)
        ax.scatter(traj1[trial,-1,0], traj1[trial,-1,1], traj1[trial,-1,2], zdir='z', color='k', s=80)


    # add task B
#    '''
    #choice 2
    traj2 = np.zeros((n_cat, ts, dim))
    for i in range(n_cat):
        for t in range(ts):
            rates = avg_fr[2,i,t,:]
            p = proj_ortho_basis(pca.components_[:dim,:], rates)
            traj2[i, t,:] = p
    
    colors2 = itertools.cycle(plt.cm.Greens(np.linspace(cmin,1,len(dlevs))))
    for trial in dlevs: # select difficulties to plot
        color = next(colors2)
        ax.scatter(traj2[trial,0,0], traj2[trial,0,1], traj2[trial,0,2], zdir='z', color=color, label='pd ' + str(trial+1) +' choice 2 '+rule)
        ax.scatter(traj2[trial,1:,0], traj2[trial,1:,1], traj2[trial,1:,2], zdir='z', color=color)
        ax.scatter(traj2[trial,tspec,0], traj2[trial,tspec,1], traj2[trial,tspec,2], zdir='z', color='k', marker='X', s=70)
        ax.scatter(traj2[trial,-1,0], traj2[trial,-1,1], traj2[trial,-1,2], zdir='z', color='k', s=80)
        
    #choice 3
    traj3 = np.zeros((n_cat, ts, dim))
    for i in range(n_cat):
        for t in range(ts):
            rates = avg_fr[3,i,t,:]
            p = proj_ortho_basis(pca.components_[:dim,:], rates)
            traj3[i, t,:] = p
    
    colors3 = itertools.cycle(plt.cm.Purples(np.linspace(cmin,1,len(dlevs)))) 
    for trial in dlevs: #select difficulties to plot
        color = next(colors3)
        ax.scatter(traj3[trial,0,0], traj3[trial,0,1], traj3[trial,0,2], zdir='z', color=color, label='pd ' + str(trial+1) +' choice 3 '+rule)
        ax.scatter(traj3[trial,:,0], traj3[trial,:,1], traj3[trial,:,2], zdir='z', color=color)
        ax.scatter(traj3[trial,tspec,0], traj3[trial,tspec,1], traj3[trial,tspec,2], zdir='z', color='k', marker='X', s=70)
        ax.scatter(traj3[trial,-1,0], traj3[trial,-1,1], traj3[trial,-1,2], zdir='z', color='k', s=80)
        
#    '''
    plt.legend(loc = 'upper left', fontsize=6)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
#plt.savefig('./WM_taskrule/'+name+'_108264_PCA.png',dpi=200)
#%% plot2: selection of trials to plot, task A, compare task rule inputs, errors allowed no avging
tlabs_d0 = pd.read_csv('./WM_taskrule/'+name+'_.6on.4off_triallabels.csv') #change
tlabs_d1 = pd.read_csv('./WM_taskrule/'+name+'_1on0off_triallabels.csv')
print('perc diff 5, perc accuracy, 0.6,0.4 rule: ', 1 - np.count_nonzero(tlabs_d0.loc[tlabs_d0['perc_diff']==5,'perc_err'])/tlabs_d0.loc[tlabs_d0['perc_diff']==5,'perc_err'].shape[0])
print('perc diff 5, perc accuracy, 1,0 rule: ', 1 - np.count_nonzero(tlabs_d1.loc[tlabs_d1['perc_diff']==5,'perc_err'])/tlabs_d1.loc[tlabs_d1['perc_diff']==5,'perc_err'].shape[0])

# choose n2plt trials of perceptual difficulty 5, errors allowed
n2plt=5
trials = np.zeros((2,n2plt),dtype='int') #row 0 is d0, row 1 is d1
trials[0,:] = tlabs_d0[(tlabs_d0['task']==0) & (tlabs_d0['perc_diff']==5)].index[:n2plt]
trials[1,:] = tlabs_d1[(tlabs_d1['task']==0) & (tlabs_d1['perc_diff']==5)].index[:n2plt]

# get firing rates
sv_d0 = pickle.load(open('./WM_taskrule/'+name+'_.6on.4off_statevar.pickle','rb')) #change
fr_d0 = np.maximum(sv_d0,0)
fr_d0_2plt = fr_d0[trials[0,:],:,:]

sv_d1 = pickle.load(open('./WM_taskrule/'+name+'_1on0off_statevar.pickle','rb'))
fr_d1 = np.maximum(sv_d1,0)
fr_d1_2plt = fr_d1[trials[1,:],:,:]
#%% plot2: 3D visualization, task A, compare task rule inputs, errors allowed no avging

# different choices and perceptual difficulties, task rule 0 (A)
fig = plt.figure()
ax = fig.gca(projection='3d')
ts = fr_d0.shape[1]
dim=3

#d0
traj0 = np.zeros((n2plt, ts, dim))
for i in range(n2plt):
    for t in range(ts):
        rates = fr_d0_2plt[i,t,:]
        p = proj_ortho_basis(pca.components_, rates)
        traj0[i, t,:] = p

for trial in range(n2plt):
    ax.scatter(traj0[trial,:,0], traj0[trial,:,1], traj0[trial,:,2], zdir='z', color='tab:red',s=8)
    ax.scatter(traj0[trial,-1,0], traj0[trial,-1,1], traj0[trial,-1,2], zdir='z', color='orange',s=50)
ax.scatter(traj0[trial,1,0], traj0[trial,1,1], traj0[trial,1,2], zdir='z', color='tab:red',s=10, label='task in. 0.6,0.4')

#d1
traj1 = np.zeros((n2plt, ts, dim))
for i in range(n2plt):
    for t in range(ts):
        rates = fr_d1_2plt[i,t,:]
        p = proj_ortho_basis(pca.components_, rates)
        traj1[i, t,:] = p

for trial in range(n2plt):
    ax.scatter(traj1[trial,:,0], traj1[trial,:,1], traj1[trial,:,2], zdir='z', color='tab:green',s=8)
    ax.scatter(traj1[trial,-1,0], traj1[trial,-1,1], traj1[trial,-1,2], zdir='z', color='lime',s=50)
ax.scatter(traj1[trial,1,0], traj1[trial,1,1], traj1[trial,1,2], zdir='z', color='tab:green',s=10, label='task in. 1,0')

plt.legend(loc='upper left')
#plt.savefig('./WM_taskrule/PCA_visualizations/'+name+'_PCAtraj_rulediff1vs.2_2.png',dpi=200)

#%%
'''
colors = itertools.cycle(plt.cm.rainbow(np.linspace(0,1,len(trials))))

for trial in range(len(trials)):
    label=None
    if trial==0:
        label = 'initial'
    ax.scatter(traj[trial,0,0], traj[trial,0,1], traj[trial,0,2], zdir='z', c='royalblue', label=label, s=50)
    for i in range(1,ts-1):
        ax.scatter(traj[trial,i,0], traj[trial,i,1], traj[trial,i,2], zdir='z', c='k')
    ax.scatter(traj[trial,-1,0], traj[trial,-1,1], traj[trial,-1,2], zdir='z', color=next(colors), label='final '+str(trial), s=50)

plt.legend()
'''
vf.plot_activities(modelout64, 43, 10, label=[0,1,2,3])

vf.plot_activities(modelout10, 11, 10, label=[0,1,2,3])
#%%
print('perc diff 5, perc accuracy, 1,0 rule: ', 1 - np.count_nonzero(tlabs10.loc[tlabs10['perc_diff']==5,'perc_err'])/tlabs10.loc[tlabs10['perc_diff']==5,'perc_err'].shape[0])
print('perc diff 5, perc accuracy, 0.6,0.4 rule: ', 1 - np.count_nonzero(tlabs64.loc[tlabs64['perc_diff']==5,'perc_err'])/tlabs64.loc[tlabs64['perc_diff']==5,'perc_err'].shape[0])
print('perc diff 5, perc accuracy, 0.5,0.5 rule: ', 1 - np.count_nonzero(tlabs55.loc[tlabs55['perc_diff']==5,'perc_err'])/tlabs55.loc[tlabs55['perc_diff']==5,'perc_err'].shape[0])

print('perc diff 4, perc accuracy, 1,0 rule: ', 1 - np.count_nonzero(tlabs10.loc[tlabs10['perc_diff']==4,'perc_err'])/tlabs10.loc[tlabs10['perc_diff']==4,'perc_err'].shape[0])
print('perc diff 4, perc accuracy, 0.6,0.4 rule: ', 1 - np.count_nonzero(tlabs64.loc[tlabs64['perc_diff']==4,'perc_err'])/tlabs64.loc[tlabs64['perc_diff']==4,'perc_err'].shape[0])
print('perc diff 4, perc accuracy, 0.5,0.5 rule: ', 1 - np.count_nonzero(tlabs55.loc[tlabs55['perc_diff']==4,'perc_err'])/tlabs55.loc[tlabs55['perc_diff']==4,'perc_err'].shape[0])

print('perc diff 4, perc accuracy, 1,0 rule: ', 1 - np.count_nonzero(tlabs10.loc[tlabs10['perc_diff']==4,'perc_err'])/tlabs10.loc[tlabs10['perc_diff']==4,'perc_err'].shape[0])
print('perc diff 4, perc accuracy, 0.6,0.4 rule: ', 1 - np.count_nonzero(tlabs64.loc[tlabs64['perc_diff']==4,'perc_err'])/tlabs64.loc[tlabs64['perc_diff']==4,'perc_err'].shape[0])
print('perc diff 4, perc accuracy, 0.5,0.5 rule: ', 1 - np.count_nonzero(tlabs55.loc[tlabs55['perc_diff']==4,'perc_err'])/tlabs55.loc[tlabs55['perc_diff']==4,'perc_err'].shape[0])



print('perc diff 4, perc accuracy task 0, 0.5,0.5 rule: ', 1 - np.count_nonzero(tlabs55.loc[(tlabs55['perc_diff']==4)&(tlabs55['task']==0),'perc_err'])/tlabs55.loc[(tlabs55['perc_diff']==4)&(tlabs55['task']==0),'perc_err'].shape[0])
print('perc diff 4, perc accuracy task 1, 0.5,0.5 rule: ', 1 - np.count_nonzero(tlabs55.loc[(tlabs55['perc_diff']==4)&(tlabs55['task']==1),'perc_err'])/tlabs55.loc[(tlabs55['perc_diff']==4)&(tlabs55['task']==1),'perc_err'].shape[0])

print('perc diff 4, perc accuracy task 0, 0.6,0.4 rule: ', 1 - np.count_nonzero(tlabs64.loc[(tlabs64['perc_diff']==4)&(tlabs64['task']==0),'perc_err'])/tlabs64.loc[(tlabs64['perc_diff']==4)&(tlabs64['task']==0),'perc_err'].shape[0])
print('perc diff 4, perc accuracy task 1, 0.6,0.4 rule: ', 1 - np.count_nonzero(tlabs64.loc[(tlabs64['perc_diff']==4)&(tlabs64['task']==1),'perc_err'])/tlabs64.loc[(tlabs64['perc_diff']==4)&(tlabs64['task']==1),'perc_err'].shape[0])

print('perc diff 4, perc accuracy task 0, 1,0 rule: ', 1 - np.count_nonzero(tlabs10.loc[(tlabs10['perc_diff']==4)&(tlabs10['task']==0),'perc_err'])/tlabs10.loc[(tlabs10['perc_diff']==4)&(tlabs10['task']==0),'perc_err'].shape[0])
print('perc diff 4, perc accuracy task 1, 1,0 rule: ', 1 - np.count_nonzero(tlabs10.loc[(tlabs10['perc_diff']==4)&(tlabs10['task']==1),'perc_err'])/tlabs10.loc[(tlabs10['perc_diff']==4)&(tlabs10['task']==1),'perc_err'].shape[0])
