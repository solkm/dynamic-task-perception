#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 09:30:28 2021

@author: Sol
"""

import numpy as np
from FourAFC_taskModels import StimHist_constant2
from psychrnn.backend.models.basic import Basic
import matplotlib.pyplot as plt
import pandas as pd
import visfunctions as vf
import scipy.optimize

model_name = 'ae_m1'
title = 'Model: trials after non-reward (monkey 1)'
weights_path = './trial_specific_models/'+model_name+'.npz'
weights = np.load(weights_path)
W_in = weights['W_in']
W_rec = weights['W_rec']
N_rec = W_rec.shape[0]
#%%
#--------------inputs weights trend visualization----------------------------

N_in = W_in.shape[1]
W_in_sumproj = np.zeros(N_in)
thresh=0.0
for i in range(N_in):
    W_in_sumproj[i] = np.sum(np.abs(W_in[:,i][W_in[:,i] > thresh]))

plt.figure()
X = np.arange(start=1,stop=N_in+1)
Y = W_in_sumproj

import itertools
colors = itertools.cycle(['orange', 'c', 'm','m','m','m', 'r'])
labels = itertools.cycle(['dsl', 'dsf', 'choice',None,None,None, 'reward'])

for i in range(len(Y)):
    if i<7:
        plt.scatter(X[i], Y[i], color=next(colors), label=next(labels))
    else:
        plt.scatter(X[i], Y[i], color=next(colors))

plt.xlabel('Input, trial n-K+1 --> n')
plt.ylabel('Summed projections')
plt.legend()
#plt.savefig('./trial_specific_models/'+model_name+'Win_trend', dpi=200)

#%% exponential fits

K=20

def f(t,a,b,c):
    return a*np.exp(-b*t)+c

#choices:

x = np.repeat(np.arange(K-1),4)
choice_in_sumproj = np.repeat(np.zeros(K-1),4)
for i in range(K-1):
    choice_in_sumproj[4*i] = W_in_sumproj[7*i+2]
    choice_in_sumproj[4*i+1] = W_in_sumproj[7*i+3]
    choice_in_sumproj[4*i+2] = W_in_sumproj[7*i+4]
    choice_in_sumproj[4*i+3] = W_in_sumproj[7*i+5]

y = np.flip(choice_in_sumproj)
plt.scatter(x,y)

popt,pcov = scipy.optimize.curve_fit(f, x, y)
xc = np.linspace(0,K-1,1000)
plt.plot(xc, f(xc,*popt),label='fit: a=%5.3f, b=%5.3f, c=%5.3f' %tuple(popt))
plt.ylabel('Summed projections: choice input')
plt.xticks(np.arange(K-1),labels=np.arange(1,K))
plt.xlabel('Trials back (n-1 --> n-K+1)')
plt.title(title)
plt.legend()
#plt.savefig('./trial_specific_models/'+model_name+'_choiceproj_expfit',dpi=200)

'''
#rewards:
x = np.arange(K-2) #exclude last
reward_in_sumproj = np.zeros(K-2)

for i in range(K-2):
    reward_in_sumproj[i] = W_in_sumproj[7*i+6]

y = np.flip(reward_in_sumproj)

#plt.figure()
plt.scatter(x,y)

popt,pcov = scipy.optimize.curve_fit(f, x, y)
xc = np.linspace(0,K-2,100)

plt.plot(x, f(x,*popt),label='fit: a=%5.3f, b=%5.3f, c=%5.3f' %tuple(popt))
#plt.plot(x, f(x,*popt),label='after R fit: a=%5.3f, b=%5.3f, c=%5.3f' %tuple(popt))
plt.ylim( np.min(reward_in_sumproj)-2, np.max(reward_in_sumproj)+1)
plt.xlabel('Trials back (n-2 --> n-K+1)')
plt.ylabel('Summed projections: reward input')
plt.xticks(x,labels=np.arange(2,K))
plt.title(title)
plt.legend()
'''
#plt.savefig('./trial_specific_models/'+model_name+'_rewardproj_expfit',dpi=200)

#plt.title('Overlayed models, monkey 1')
#plt.savefig('./trial_specific_models/m1_overlayed_rewardproj_expfit',dpi=200)