#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 12:53:05 2021

@author: Sol
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tParams_tags = pd.read_csv('./tParams_nDist_df.csv')

# bin by stimulus change sizes, define the bins based on the dsf and dsl distributions

ub = 1.1
lb = 0.9

tParams_tags.hist(column='dsf', bins=50)
plt.vlines([ub,-ub,lb,-lb],0,30000, colors='r')
tParams_tags.hist(column='dsl', bins=50)
plt.vlines([ub,-ub,lb,-lb],0,30000, colors='r')

tParams_tags['dsf_size'] = 'med'
tParams_tags.loc[np.abs(tParams_tags['dsf']) < lb, 'dsf_size'] = 'small'
tParams_tags.loc[np.abs(tParams_tags['dsf']) > ub, 'dsf_size'] = 'large'

N_dsfsmall = tParams_tags[tParams_tags['dsf_size']=='small'].shape[0]
N_dsflarge = tParams_tags[tParams_tags['dsf_size']=='large'].shape[0]
N_dsfmed = tParams_tags[tParams_tags['dsf_size']=='med'].shape[0]

dsf_small_acc = tParams_tags[(tParams_tags['dsf_size']=='small')&(tParams_tags['m_task']==2)&(tParams_tags['perc_err']==0)].shape[0]\
    / tParams_tags[(tParams_tags['dsf_size']=='small')&(tParams_tags['m_task']==2)].shape[0]
    
dsf_large_acc = tParams_tags[(tParams_tags['dsf_size']=='large')&(tParams_tags['m_task']==2)&(tParams_tags['perc_err']==0)].shape[0]\
    / tParams_tags[(tParams_tags['dsf_size']=='large')&(tParams_tags['m_task']==2)].shape[0]

dsf_med_acc = tParams_tags[(tParams_tags['dsf_size']=='med')&(tParams_tags['m_task']==2)&(tParams_tags['perc_err']==0)].shape[0]\
    / tParams_tags[(tParams_tags['dsf_size']=='med')&(tParams_tags['m_task']==2)].shape[0]

tParams_tags['dsl_size'] = 'med'
tParams_tags.loc[np.abs(tParams_tags['dsl']) < lb, 'dsl_size'] = 'small'
tParams_tags.loc[np.abs(tParams_tags['dsl']) > ub, 'dsl_size'] = 'large'

N_dslsmall = tParams_tags[tParams_tags['dsl_size']=='small'].shape[0]
N_dsllarge = tParams_tags[tParams_tags['dsl_size']=='large'].shape[0]
N_dslmed = tParams_tags[tParams_tags['dsl_size']=='med'].shape[0]

dsl_small_acc = tParams_tags[(tParams_tags['dsl_size']=='small')&(tParams_tags['m_task']==1)&(tParams_tags['perc_err']==0)].shape[0]\
    / tParams_tags[(tParams_tags['dsl_size']=='small')&(tParams_tags['m_task']==1)].shape[0]
    
dsl_large_acc = tParams_tags[(tParams_tags['dsl_size']=='large')&(tParams_tags['m_task']==1)&(tParams_tags['perc_err']==0)].shape[0]\
    / tParams_tags[(tParams_tags['dsl_size']=='large')&(tParams_tags['m_task']==1)].shape[0]

dsl_med_acc = tParams_tags[(tParams_tags['dsl_size']=='med')&(tParams_tags['m_task']==1)&(tParams_tags['perc_err']==0)].shape[0]\
    / tParams_tags[(tParams_tags['dsl_size']=='med')&(tParams_tags['m_task']==1)].shape[0]
    
# aNR_inds = tParams_tags.index[tParams_tags['err']==1] + 1 # get after reward and after non-reward indices
#tParams_tags.to_csv('./tParams_full.csv', index=False)

#%%

import behavior_analysis as ba

dsl_perf_aR1, dsl_perf_aNR1, dsf_perf_aR1, dsf_perf_aNR1, p_dsl1, p_dsf1 = \
    ba.percPerfSameStim(tParams_tags[(tParams_tags['dsf_size']=='small')&(tParams_tags['dsl_size']=='large')], None)
print('small dsf, large dsl.')
print('dsf perf aR: ',np.mean(dsf_perf_aR1), 'dsf perf aNR: ', np.mean(dsf_perf_aNR1), 'p: ', p_dsf1)
print('dsl perf aR: ',np.mean(dsl_perf_aR1), 'dsl perf aNR: ', np.mean(dsl_perf_aNR1), 'p: ', p_dsl1)

diff11 = np.subtract(dsl_perf_aR1, dsl_perf_aNR1)
plt.hist(diff11)
plt.vlines((np.mean(diff11),0), 0, 55, colors='k',linestyles='--')
plt.xlabel('SL performance after reward - after nonreward')
plt.title('Small dSF, large dSL trials')

plt.figure()
diff12 = np.subtract(dsf_perf_aR1, dsf_perf_aNR1)
plt.hist(diff12)
plt.vlines((np.mean(diff12),0), 0, 55, colors='k',linestyles='--')
plt.xlabel('SF performance after reward - after nonreward')
plt.title('Small dSF, large dSL trials')

dsl_perf_aR2, dsl_perf_aNR2, dsf_perf_aR2, dsf_perf_aNR2, p_dsl2, p_dsf2 = \
    ba.percPerfSameStim(tParams_tags[(tParams_tags['dsf_size']=='large')&(tParams_tags['dsl_size']=='small')], None)
print('large dsf, small dsl.')
print('dsf perf aR: ',np.mean(dsf_perf_aR2), 'dsf perf aNR: ', np.mean(dsf_perf_aNR2), 'p: ', p_dsf2)
print('dsl perf aR: ',np.mean(dsl_perf_aR2), 'dsl perf aNR: ', np.mean(dsl_perf_aNR2), 'p: ', p_dsl2)

dsl_perf_aR3, dsl_perf_aNR3, dsf_perf_aR3, dsf_perf_aNR3, p_dsl3, p_dsf3 = \
    ba.percPerfSameStim(tParams_tags[(tParams_tags['dsf_size']=='large')&(tParams_tags['dsl_size']=='large')], None)
print('large large.')
print('dsl perf aR: ',np.mean(dsl_perf_aR3), 'dsl perf aNR: ', np.mean(dsl_perf_aNR3), 'p: ', p_dsl3)
print('dsf perf aR: ',np.mean(dsf_perf_aR3), 'dsf perf aNR: ', np.mean(dsf_perf_aNR3), 'p: ', p_dsf3)

diff31 = np.subtract(dsl_perf_aR3, dsl_perf_aNR3)
plt.hist(diff31)
plt.vlines((np.mean(diff31),0), 0, 55, colors='k',linestyles='--')
plt.xlabel('SL performance after reward - after nonreward')
plt.title('Large dSF, large dSL trials')

plt.figure()
diff32 = np.subtract(dsf_perf_aR3, dsf_perf_aNR3)
plt.hist(diff32)
plt.vlines((np.mean(diff32),0), 0, 55, colors='k',linestyles='--')
plt.xlabel('SF performance after reward - after nonreward')
plt.title('Large dSF, large dSL trials')

dsl_perf_aR4, dsl_perf_aNR4, dsf_perf_aR4, dsf_perf_aNR4, p_dsl4, p_dsf4 = \
    ba.percPerfSameStim(tParams_tags[(tParams_tags['dsf_size']=='small')&(tParams_tags['dsl_size']=='small')], None)
print('small small.')
print('dsf perf aR: ',np.mean(dsf_perf_aR4), 'dsf perf aNR: ', np.mean(dsf_perf_aNR4), 'p: ', p_dsf4)
print('dsl perf aR: ',np.mean(dsl_perf_aR4), 'dsl perf aNR: ', np.mean(dsl_perf_aNR4), 'p: ', p_dsl4)

# when spliting into these partitions, the after reward vs after non-reward difference is only significant for large changes in the chosen stimulus.
# just a statistical effect (need larger n for significant difference)?
#%%
import scipy.io
tParams_getsess = scipy.io.loadmat('./Cheng_recTrainTest/tParams.mat')
sess_inds = tParams_getsess['r_indinit']

tParams_full = pd.read_csv('./tParams_full.csv')
percPerfs = pd.DataFrame(columns=['dataset','afterCorr','afterErr','afterErrHard','afterErrHard_S','afterErrHard_NS','afterErrEasy','afterErrEasy_S','afterErrEasy_NS'])


TPs = [tParams_full[:],tParams_full[0:20000],tParams_full[20000:40000],tParams_full[40000:60000],tParams_full[91493:111594],tParams_full[111594:131594],tParams_full[131594:151594]]

DSs = ['all trials, 1-155958','monkey1, 1-20000','monkey1, 200001-40000','monkey1, 40001-60000','monkey2, 91494-111594','monkey2, 111595-131594','monkey2, 131595-151594']

for i in range(len(TPs)):
    tp = TPs[i]
    ds = DSs[i]

    afterErr = tp[tp['err']==1].index + 1
    afterCorr = tp[tp['err']==0].index + 1
    
    afterErrHard = tp[(tp['err']==1) & (((tp['m_task']==1)&(tp['dsl_size']=='small')) | ((tp['m_task']==2)&(tp['dsf_size']=='small')))].index + 1
    
    afterErrEasy = tp[(tp['err']==1) & (((tp['m_task']==1)&(tp['dsl_size']=='large')) | ((tp['m_task']==2)&(tp['dsf_size']=='large')))].index + 1
    
    pp_afterErr = 1 - np.count_nonzero(tp.loc[afterErr[:-1], 'perc_err'])/afterErr[:-1].shape[0]
    pp_afterCorr = 1 - np.count_nonzero(tp.loc[afterCorr[:-1], 'perc_err'])/afterCorr[:-1].shape[0]
    
    pp_afterErrHard = 1 - np.count_nonzero(tp.loc[afterErrHard[:-1], 'perc_err'])/afterErrHard[:-1].shape[0]
    pp_afterErrEasy = 1 - np.count_nonzero(tp.loc[afterErrEasy[:-1], 'perc_err'])/afterErrEasy[:-1].shape[0]
    
    afterErrHard_S = tp[((tp['err']==1)&(tp['m_switch']==1)) & (((tp['m_task']==1)&(tp['dsl_size']=='small')) | ((tp['m_task']==2)&(tp['dsf_size']=='small')))].index + 1
        
    afterErrHard_NS = tp[((tp['err']==1)&(tp['m_switch']==0)) & (((tp['m_task']==1)&(tp['dsl_size']=='small')) | ((tp['m_task']==2)&(tp['dsf_size']=='small')))].index + 1
    
    afterErrEasy_S = tp[((tp['err']==1)&(tp['m_switch']==1)) & (((tp['m_task']==1)&(tp['dsl_size']=='large')) | ((tp['m_task']==2)&(tp['dsf_size']=='large')))].index + 1
        
    afterErrEasy_NS = tp[((tp['err']==1)&(tp['m_switch']==0)) & (((tp['m_task']==1)&(tp['dsl_size']=='large')) | ((tp['m_task']==2)&(tp['dsf_size']=='large')))].index + 1
    
    pp_afterErrHard_S = 1 - np.count_nonzero(tp.loc[afterErrHard_S[:-1], 'perc_err'])/afterErrHard_S[:-1].shape[0]
    pp_afterErrHard_NS = 1 - np.count_nonzero(tp.loc[afterErrHard_NS[:-1], 'perc_err'])/afterErrHard_NS[:-1].shape[0]
    pp_afterErrEasy_S = 1 - np.count_nonzero(tp.loc[afterErrEasy_S[:-1], 'perc_err'])/afterErrEasy_S[:-1].shape[0]
    pp_afterErrEasy_NS = 1 - np.count_nonzero(tp.loc[afterErrEasy_NS[:-1], 'perc_err'])/afterErrEasy_NS[:-1].shape[0]

    pp = pd.DataFrame(data={'dataset':[ds], 'afterCorr':[pp_afterCorr], 'afterErr':[pp_afterErr], 'afterErrHard':[pp_afterErrHard], 'afterErrHard_S':[pp_afterErrHard_S], \
                            'afterErrHard_NS':[pp_afterErrHard_NS], 'afterErrEasy':[pp_afterErrEasy], 'afterErrEasy_S':[pp_afterErrEasy_S],'afterErrEasy_NS':[pp_afterErrEasy_NS]})
    
    percPerfs = percPerfs.append(pp, ignore_index=True)
