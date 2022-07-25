#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 12:11:48 2021

@author: Sol
"""

import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from FourAFC_taskModels import StimHist_constant2
from psychrnn.backend.models.basic import Basic
import scipy.io
import pickle

#%%
# dimensionality: large test set, all trials and after reward vs. after non-reward

N_rec = 200
dt = 10
tau = 100
T = 1000
K = 20
in_noise = 0.1
rec_noise = 0.1
vis_noise = 0.0
mem_noise = 0.0

tParams_df = pd.read_csv('./tParams_df.csv')
N_tot = tParams_df.shape[0]
tParams_mat = scipy.io.loadmat('./Cheng_recTrainTest/tParams.mat')
t_ind = np.arange(N_tot)
# remove indices that overlap sessions
sess_ind = np.array(tParams_mat['r_indinit'][0], dtype=int)-1
N_s = len(sess_ind)
del_ind = np.empty((N_s-1)*(K-1), dtype=int)
for j in range(0, N_s-1):
    temp = np.arange(sess_ind[j+1]-K+1, sess_ind[j+1], dtype=int)
    del_ind[j*(K-1):(j+1)*(K-1)] = temp
    
t_ind2 = np.delete(t_ind, del_ind)
t_ind2 = t_ind2[t_ind2<N_tot-K+1]
np.random.seed(1234)
N_batch = 20000
test_inds = np.random.choice(t_ind2, N_batch, replace=False)

df = pd.read_csv('./dimensionality.csv')
weights_paths = ['./saved_weights/trainChoice_v4_2.npz', './saved_weights/trainChoice_v4_2_correctctrl.npz']
model_names = ['choice_v4_2', 'choice_v4_2_correctctrl']
exp_vars = []
dim90 = []
exp_vars_aR = []
dim90_aR = []
exp_vars_aNR = []
dim90_aNR = []
Nb = []
for i in range(len(weights_paths)):
    
    weights_path = weights_paths[i]
    model_name = model_names[i]

    #Load model
    task1 = StimHist_constant2(dt, tau, T, N_batch, K, dat1=tParams_df, dat1_inds=test_inds, testall=True, \
                               in_noise=in_noise, vis_noise=vis_noise, mem_noise=mem_noise)
    network_params1 = task1.get_task_params()
    network_params1['name'] = model_name
    network_params1['N_rec'] = N_rec
    network_params1['load_weights_path'] = weights_path
    network_params1['rec_noise'] = rec_noise
    model1 = Basic(network_params1)
    
    # Get test batch
    x1, _, _, trial_params1 = task1.get_trial_batch()
    _, state_var1 = model1.test(x1)

    tp_aR = np.array([trial_params1[i]['err'][-2]==0 for i in range(N_batch)])
    state_var_aR = state_var1[tp_aR,:,:]
    state_var_aNR = state_var1[np.invert(tp_aR),:,:]
    
    rates = np.maximum(state_var1, 0)
    rates_aR = np.maximum(state_var_aR,0)
    rates_aNR = np.maximum(state_var_aNR,0)
    
    t_off = int(T//dt*3/4)
    rates_endstim = rates[:, t_off-5:t_off, :]
    rates_endstim_aR = rates_aR[:, t_off-5:t_off, :]
    rates_endstim_aNR = rates_aNR[:, t_off-5:t_off, :]
    
    avg_rates = np.mean(rates_endstim, axis=1)
    avg_rates_aR = np.mean(rates_endstim_aR, axis=1)
    avg_rates_aNR = np.mean(rates_endstim_aNR, axis=1)
    
    pca_90 = PCA(n_components=.9, svd_solver='full')
    pca_90.fit(avg_rates)
    exp_vars.append(pca_90.explained_variance_ratio_)
    dim90.append(pca_90.n_components_)
    print(model_name,' dim90: ',dim90[i])
    
    pca_90_aR = PCA(n_components=.9, svd_solver='full')
    pca_90_aR.fit(avg_rates_aR)
    exp_vars_aR.append(pca_90_aR.explained_variance_ratio_)
    dim90_aR.append(pca_90_aR.n_components_)
    print(model_name,' dim90_aR: ',dim90_aR[i])
    
    pca_90_aNR = PCA(n_components=.9, svd_solver='full')
    pca_90_aNR.fit(avg_rates_aNR)
    exp_vars_aNR.append(pca_90_aNR.explained_variance_ratio_)
    dim90_aNR.append(pca_90_aNR.n_components_)   
    print(model_name,' dim90_aNR: ',dim90_aNR[i])
    
    Nb.append(N_batch)
    
df_a = pd.DataFrame(columns=['model','dim90','dim90_aR','dim90_aNR','N_batch'])
df_a['model'] = model_names
df_a['dim_90'] = dim90
df_a['dim90_aR'] = dim90_aR
df_a['dim90_aNR'] = dim90_aNR
df_a['N_batch'] = Nb
df_a.to_csv('./dimensionality.csv', mode='a', index=False, header=False)

#%%
'''
savefile = open('./pickled_testdata/choice_v4_2_expvaraR.pickle','wb')
pickle.dump(exp_vars_aR[0], savefile, protocol=4)
savefile.close()
savefile = open('./pickled_testdata/choice_v4_2_correctctrl_expvaraR.pickle','wb')
pickle.dump(exp_vars_aR[1], savefile, protocol=4)
savefile.close()
savefile = open('./pickled_testdata/choice_v4_2_expvaraNR.pickle','wb')
pickle.dump(exp_vars_aNR[0], savefile, protocol=4)
savefile.close()
savefile = open('./pickled_testdata/choice_v4_2_correctctrl_expvaraNR.pickle','wb')
pickle.dump(exp_vars_aNR[1], savefile, protocol=4)
savefile.close()
'''
#%%

import matplotlib.pyplot as plt

exp_vars = [pickle.load(open('./pickled_testdata/choice_v4_2_expvar.pickle', 'rb')), pickle.load(open('./pickled_testdata/choice_v4_2_correctctrl_expvar.pickle', 'rb'))]
exp_vars_aR = [pickle.load(open('./pickled_testdata/choice_v4_2_expvaraR.pickle', 'rb')), pickle.load(open('./pickled_testdata/choice_v4_2_correctctrl_expvaraR.pickle', 'rb'))]
exp_vars_aNR = [pickle.load(open('./pickled_testdata/choice_v4_2_expvaraNR.pickle', 'rb')), pickle.load(open('./pickled_testdata/choice_v4_2_correctctrl_expvaraNR.pickle', 'rb'))]

tev_choice = np.cumsum(exp_vars[0])
tev_correct = np.cumsum(exp_vars[1])
tev_aR_choice = np.cumsum(exp_vars_aR[0])
tev_aR_correct = np.cumsum(exp_vars_aR[1])
tev_aNR_choice = np.cumsum(exp_vars_aNR[0])
tev_aNR_correct = np.cumsum(exp_vars_aNR[1])

# cumulative variance plot
plt.figure()

plt.plot(np.arange(1, len(tev_correct)+1), tev_correct*100, label='correct model', color='blue')
#plt.plot(np.arange(1, len(tev_aNR_correct)+1), tev_aNR_correct*100, label='correct model, after nonreward', color='dodgerblue', linestyle=(0, (6, 1)))
#plt.plot(np.arange(1, len(tev_aR_correct)+1), tev_aR_correct*100, label='correct model, after reward', color='dodgerblue', linestyle=(0, (4, 2, 1, 2)))
plt.plot(np.arange(1, len(tev_choice)+1), tev_choice*100, label='choice model', color='red')
#plt.plot(np.arange(1, len(tev_aNR_choice)+1), tev_aNR_choice*100, label='choice model, after nonreward', color='darkorange', linestyle=(0, (6, 1)))
#plt.plot(np.arange(1, len(tev_aR_choice)+1), tev_aR_choice*100, label='choice model, after reward', color='darkorange', linestyle=(0, (4, 2, 1, 2)))

plt.hlines(90,1,40,color='darkgrey')

plt.legend(loc='lower right', fontsize=16)
plt.xlabel('Number of principle components', fontsize=16)
plt.ylabel('Cumulative explained variance (%)', fontsize=16)
plt.xticks(ticks=np.arange(1, len(tev_aNR_choice)+1, 3), fontsize=12)
yticks = np.arange(11)*10
plt.yticks(yticks, labels=yticks, fontsize=12)
plt.ylim(15,95)
plt.xlim(1,35)
#plt.savefig('./Figures/uPNC_poster/correctvchoice_dim_allonly', dpi=200)
#%%
# barplot, difference in explained variance after reward vs after nonreward trials
n=5
ev_aR_choice = np.sum(exp_vars_aR[0][:n])
ev_aR_correct = np.sum(exp_vars_aR[1][:n])
ev_aNR_choice = np.sum(exp_vars_aNR[0][:n])
ev_aNR_correct = np.sum(exp_vars_aNR[1][:n])

diff_aR = ev_aR_correct - ev_aR_choice
diff_aNR = ev_aNR_correct - ev_aNR_choice

fig, ax = plt.subplots()
width = 0.8
x = np.arange(2)
ax.bar(x[0], diff_aR, width=width, color='tab:blue')
ax.bar(x[1], diff_aNR, width=width, color='tab:cyan')
ax.bar(x[0]-width/2,0)
ax.bar(x[1]+width/2,0)
ax.set_xticks(x)
ax.set_xticklabels(['after reward', 'after nonreward'])
ax.set_title('')
ax.set_ylabel('Difference in variance explained by first %d PCs\n(correct model - choice model)'%n)
ax.set_xlabel('')
plt.legend()

