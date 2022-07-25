#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 20:26:19 2021

@author: Sol
"""
import numpy as np
from NCT_Fnct_Tevin import RNNControllability
import timeit
import pandas as pd

ctrl_df = pd.read_csv('./controllability.csv')

#avgmodal_df = ctrl_df[ctrl_df['measure']=='avg_modal'].copy()
#avg_df = ctrl_df[ctrl_df['measure']=='avg'].copy()
#global_df = ctrl_df[ctrl_df['measure']=='global'].copy()
#%%
load_path = ['./saved_weights/trainChoice_v5_3.npz'] # saved weights files
save_name = ['choice_v5_3' ] # model name for saving the data

for i in range(len(load_path)):
    weights = np.load(load_path[i])
    W_rec = weights['W_rec']
    start = timeit.default_timer()
    modal_ctrl_vect, avg_ctrl, global_ctrl = RNNControllability(W_rec)
    stop = timeit.default_timer()
    print('runtime: ', stop-start)
    avg_modal = np.mean(modal_ctrl_vect)
    print('modal vect: ', modal_ctrl_vect)
    print('avg modal, avg, global: ', avg_modal, avg_ctrl, global_ctrl)
    
    df1 = pd.DataFrame()
    df1['model'] = [save_name[i], save_name[i], save_name[i], save_name[i]]
    df1['measure'] = ['modal', 'avg_modal', 'avg', 'global']
    df1['value'] = [modal_ctrl_vect, avg_modal, avg_ctrl, global_ctrl]
    df1.to_csv('./controllability.csv', mode='a', header=False, index=False)

#%%
#----------------Modular, no dale------------------
load_path2 = ['./saved_weights/trainChoice_v5_3.npz']
save_name2 = ['choice_v5_3']

for i in range(len(load_path2)):
    weights = np.load(load_path2[i])
    W_rec = weights['W_rec']
    ms = W_rec.shape[0]//2
    W_P = W_rec[:ms,:ms].copy()
    W_T = W_rec[ms:,ms:].copy()
    
    print('Perception in')
    start = timeit.default_timer()
    modal_ctrl_vect1, avg_ctrl1, global_ctrl1 = RNNControllability(W_P)
    stop = timeit.default_timer()
    print('runtime: ', stop-start)
    avg_modal1 = np.mean(modal_ctrl_vect1)
    print('modal vect: ', modal_ctrl_vect1)
    print('avg modal, avg, global: ', avg_modal1, avg_ctrl1, global_ctrl1)
    
    print('Task in')
    start = timeit.default_timer()
    modal_ctrl_vect2, avg_ctrl2, global_ctrl2 = RNNControllability(W_T)
    stop = timeit.default_timer()
    print('runtime: ', stop-start)
    avg_modal2 = np.mean(modal_ctrl_vect2)
    print('modal vect: ', modal_ctrl_vect2)
    print('avg modal, avg, global: ', avg_modal2, avg_ctrl2, global_ctrl2)
    
    df2 = pd.DataFrame()
    df2['model'] = [save_name2[i]+'_P', save_name2[i]+'_P', save_name2[i]+'_P', save_name2[i]+'_P', \
                    save_name2[i]+'_T', save_name2[i]+'_T', save_name2[i]+'_T', save_name2[i]+'_T']    
    df2['measure'] = ['modal','avg_modal','avg','global','modal','avg_modal','avg','global']
    df2['value'] = [modal_ctrl_vect1, avg_modal1, avg_ctrl1, global_ctrl1, modal_ctrl_vect2, avg_modal2, avg_ctrl2, global_ctrl2]
    
    df2.to_csv('./controllability.csv', mode='a', header=False, index=False)


#%%
#----------------Modular, dale------------------
load_path3 = ['./saved_weights/trainCorrect_v2(4).npz','./saved_weights/trainCorrect_v2(5).npz']
save_name3 = ['correct_v2(4)', 'correct_v2(5)']

for i in range(len(load_path3)):
    weights = np.load(load_path3[i])
    W_rec = weights['W_rec']
    W_Pe = W_rec[:40,:40].copy()
    W_Te = W_rec[40:80,40:80].copy()
    W_Pi = W_rec[80:90,80:90].copy()
    W_Ti = W_rec[90:,90:].copy()
    
    print('Excitatory, perception in')
    start = timeit.default_timer()
    modal_ctrl_vect1, avg_ctrl1, global_ctrl1 = RNNControllability(W_Pe)
    stop = timeit.default_timer()
    print('runtime: ', stop-start)
    avg_modal1 = np.mean(modal_ctrl_vect1)
    print('modal vect: ', modal_ctrl_vect1)
    print('avg modal, avg, global: ', avg_modal1, avg_ctrl1, global_ctrl1)
    
    print('Excitatory, task in')
    start = timeit.default_timer()
    modal_ctrl_vect2, avg_ctrl2, global_ctrl2 = RNNControllability(W_Te)
    stop = timeit.default_timer()
    print('runtime: ', stop-start)
    avg_modal2 = np.mean(modal_ctrl_vect2)
    print('modal vect: ', modal_ctrl_vect2)
    print('avg modal, avg, global: ', avg_modal2, avg_ctrl2, global_ctrl2)
    
    print('Inhibitory, perception in')
    start = timeit.default_timer()
    modal_ctrl_vect3, avg_ctrl3, global_ctrl3 = RNNControllability(W_Pi)
    stop = timeit.default_timer()
    print('runtime: ', stop-start)
    avg_modal3 = np.mean(modal_ctrl_vect3)
    print('modal vect: ', modal_ctrl_vect3)
    print('avg modal, avg, global: ', avg_modal3, avg_ctrl3, global_ctrl3)
    
    print('Inhibitory, task in')
    start = timeit.default_timer()
    modal_ctrl_vect4, avg_ctrl4, global_ctrl4 = RNNControllability(W_Ti)
    stop = timeit.default_timer()
    print('runtime: ', stop-start)
    avg_modal4 = np.mean(modal_ctrl_vect4)
    print('modal vect: ', modal_ctrl_vect4)
    print('avg modal, avg, global: ', avg_modal4, avg_ctrl4, global_ctrl4)

    df3 = pd.DataFrame()
    df3['model'] = [save_name3[i]+'_Pe', save_name3[i]+'_Pe', save_name3[i]+'_Pe', save_name3[i]+'_Pe', \
                    save_name3[i]+'_Te', save_name3[i]+'_Te', save_name3[i]+'_Te', save_name3[i]+'_Te', \
                    save_name3[i]+'_Pi', save_name3[i]+'_Pi', save_name3[i]+'_Pi', save_name3[i]+'_Pi', \
                    save_name3[i]+'_Ti', save_name3[i]+'_Ti', save_name3[i]+'_Ti', save_name3[i]+'_Ti']     
    df3['measure'] = ['modal','avg_modal','avg','global','modal','avg_modal','avg','global','modal','avg_modal','avg','global','modal','avg_modal','avg','global']
    df3['value'] = [modal_ctrl_vect1, avg_modal1, avg_ctrl1, global_ctrl1, modal_ctrl_vect2, avg_modal2, avg_ctrl2, global_ctrl2, \
                   modal_ctrl_vect3, avg_modal3, avg_ctrl3, global_ctrl3, modal_ctrl_vect4, avg_modal4, avg_ctrl4, global_ctrl4]
    
    df3.to_csv('./controllability.csv', mode='a', header=False, index=False)
    
#%%
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.weightstats import ztest

ctrl_modal = ctrl_df[ctrl_df['measure']=='avg_modal'].copy()

dropinds = np.concatenate([ctrl_modal[(33<= ctrl_modal.index)&(ctrl_modal.index <=85)].index, ctrl_modal[(101<=ctrl_modal.index)&(ctrl_modal.index <=161)].index])
#dropinds = np.concatenate([ctrl_modal[(37<= ctrl_modal.index)&(ctrl_modal.index <=81)].index, ctrl_modal[(101<=ctrl_modal.index)&(ctrl_modal.index <=129)].index, [137,141,145,149,157,161]])
modalfull = ctrl_modal.copy().drop(index=dropinds)
#modalfull['dale'] = [0,1,0,1,0,1,1,1,0,0,1,1,0,0,0]
modalfull['dale'] = [0,1,0,1,0,1,1,1,1,1,0]
modalfull = modalfull.sort_values('value')
modalfull.loc[modalfull['dale']==0,'dale'] = 'No Dale'
modalfull.loc[modalfull['dale']==1,'dale'] = 'Dale'
modalfull = modalfull.astype({'value':'float64'})

_,p = ztest(modalfull[modalfull['dale']=='Dale']['value'], modalfull[modalfull['dale']=='No Dale']['value'])

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'

fig,ax = plt.subplots()
sns.swarmplot(x='dale', y='value', data=modalfull, hue='dale', palette=sns.color_palette(['r','b']), size=8)
sns.pointplot(x='dale',  y='value', data=modalfull, estimator=np.mean, color='slategrey', capsize=0.1)
plt.ylabel('Average Modal Controllability', fontsize=18)
plt.xlabel('')
ax.tick_params(axis='y', labelsize=12)
ax.tick_params(axis='x', labelsize=18)
plt.legend([],[], frameon=False)
plt.text(-0.1,.3,s='p = %2.1e'%p, fontsize=16)
#plt.savefig('./Figures/uPNC_poster/modalCtrlDales_ctrld', dpi=200)

