#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 16:59:40 2022

@author: Sol
"""

import numpy as np
import pickle
import pandas as pd
import model_functions as mf

rule = '1on0off'
name = 'vardelay50'
tparams = pickle.load(open('./WM_taskrule/'+name+'_'+rule+'_trialparams.pickle','rb'))
modelout = pickle.load(open('./WM_taskrule/'+name+'_'+rule+'_modeloutput.pickle','rb'))
choice = np.argmax(modelout[:,-1,:], axis=1)

model_df = pd.DataFrame(columns=['da','db','task','targ','model_choice','err','perc_err','task_err','delay2_dur'])

for i in range(tparams.shape[0]):
    model_df.loc[i,'da'] = tparams[i]['a2'] - tparams[i]['a1']
    model_df.loc[i,'db'] = tparams[i]['b2'] - tparams[i]['b1']
    model_df.loc[i,'task'] = tparams[i]['task']
    model_df.loc[i,'targ'] = tparams[i]['targ']
    model_df.loc[i,'model_choice'] = choice[i]
    model_df.loc[i, 'delay2_dur'] = tparams[i]['delay2_dur']

task_acc, perc_acc, overall_acc, noresponse, taskerr_trials, percerr_trials, nr_trials = mf.get_Accuracies(modelout, tparams)
model_df.loc[:,'task_err'] = 0
model_df.loc[:,'perc_err'] = 0
model_df.loc[:,'err'] = 0
model_df.loc[taskerr_trials,'task_err']=1
model_df.loc[percerr_trials,'perc_err']=1
model_df.loc[np.concatenate((taskerr_trials,percerr_trials)),'err']=1


print(1 - np.count_nonzero(model_df.loc[model_df['delay2_dur']<350, 'err'])/np.count_nonzero(model_df['delay2_dur']<350))
print(1 - np.count_nonzero(model_df.loc[model_df['delay2_dur']>450, 'err'])/np.count_nonzero(model_df['delay2_dur']>450))
print(1 - np.count_nonzero(model_df.loc[model_df['delay2_dur']<350, 'perc_err'])/np.count_nonzero(model_df['delay2_dur']<350))
print(1 - np.count_nonzero(model_df.loc[model_df['delay2_dur']>450, 'perc_err'])/np.count_nonzero(model_df['delay2_dur']>450))
print(1 - np.count_nonzero(model_df.loc[model_df['delay2_dur']<350, 'task_err'])/np.count_nonzero(model_df['delay2_dur']<350))
print(1 - np.count_nonzero(model_df.loc[model_df['delay2_dur']>450, 'task_err'])/np.count_nonzero(model_df['delay2_dur']>450))




