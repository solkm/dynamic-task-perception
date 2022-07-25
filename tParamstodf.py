#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 13:20:31 2021

@author: Sol
"""

import numpy as np
import scipy.io
import pandas as pd
import model_functions as mf

#%% tParams_df

tParams = scipy.io.loadmat('/Users/Sol/Desktop/CohenLab/DynamicTaskPerceptionProject/mat_files/tParams_LongHard.mat')

tParams_df = pd.DataFrame({'choice':tParams['r_CHOICE'][0], 'correct':tParams['r_CORRECT'][0], 'dsf':tParams['r_DSF'][0], 'dsl':tParams['r_DSL'][0], 'perc_err':tParams['r_DERR'][0], 'task':tParams['r_TASK'][0]})

tParams_df['err'] = np.where(tParams_df['choice']!=tParams_df['correct'], 1, 0).astype('uint8')
tParams_df['m_task'] = mf.get_tasks(tParams_df['choice']).astype('uint8')
tParams_df['task_err'] = np.where(tParams_df['task']!=tParams_df['m_task'], 1, 0).astype('uint8')
tParams_df['switch'] = mf.task_switches(tParams_df['task'])[0].astype('uint8')
tParams_df['m_switch'] = mf.task_switches(tParams_df['m_task'])[0].astype('uint8')

tParams_df.to_csv('/Users/Sol/Desktop/Psych4631/glmhmm/tParams_LongHard_df.csv', index=False)

#%% tParams_nDist_df

tParams = scipy.io.loadmat('./mat_files/tParams_nDist.mat')

tParams_df = pd.DataFrame({'choice':tParams['r_CHOICE'][0], 'correct':tParams['r_CORRECT'][0], 'dsf':tParams['r_DSF'][0], 'dsl':tParams['r_DSL'][0], 'perc_err':tParams['r_DERR'][0], 'task':tParams['r_TASK'][0], \
                           'ndBelief':tParams['r_NDB'][0], 'ndSL':tParams['r_NDSL'][0], 'ndSF':tParams['r_NDSF'][0]})

tParams_df['err'] = np.where(tParams_df['choice']!=tParams_df['correct'], 1, 0).astype('uint8')
tParams_df['m_task'] = mf.get_tasks(tParams_df['choice']).astype('uint8')
tParams_df['task_err'] = np.where(tParams_df['task']!=tParams_df['m_task'], 1, 0).astype('uint8')
tParams_df['switch'] = mf.task_switches(tParams_df['task'])[0].astype('uint8')
tParams_df['m_switch'] = mf.task_switches(tParams_df['m_task'])[0].astype('uint8')

tParams_df.to_csv('./tParams_nDist_df.csv', index=False)

#%%

tParams = scipy.io.loadmat(r'./mat_files/tParams.mat')

tParams_df = pd.read_csv('./tParams_df.csv')









