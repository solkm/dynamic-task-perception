#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 17:19:27 2021

@author: Sol
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.weightstats import ztest
import model_functions as mf

def percPerfSameStim(trial_params, model_output):
    """
    Plots the perceptual performance for the same stimuli after a reward vs. after a non-reward.

    Parameters
    ----------
    trial_params : array or dict or df.
        Trial parameters, either a behavioral dataset or one that the model was tested on.
    model_output : array
        The model output from testing. None if plotting a behavioral dataset.
    title : str, optional
        Plot title. The default is ''.
    plot : bool, optional
        Generates a plot if True, doesn't if False. The default is True.
        
    Returns
    -------
    dsl_perf_aR : list
    dsl_perf_aNR : list
    dsf_perf_aR : list
    dsf_perf_aNR : list
    p_dsl : float
    p_dsf : float
    """
    
    if model_output is not None:
        N = len(trial_params[:])
        dsl = [trial_params[i]['dsl'][-1] for i in range(N)]
        dsf = [trial_params[i]['dsf'][-1] for i in range(N)]
        prevChoice = [trial_params[i]['choice'][-2] for i in range(N)]
        prevCorrect = [trial_params[i]['correct'][-2] for i in range(N)]
        model_choice = np.argmax(model_output[:,-1,:], axis=1) + 1

    elif model_output is None:
        dsl = np.array(trial_params['dsl'])
        dsf = np.array(trial_params['dsf'])
        prevChoice = np.concatenate(([10], np.array(trial_params['choice'][:-1])))
        prevCorrect = np.concatenate(([20], np.array(trial_params['correct'][:-1])))
        model_choice = np.array(trial_params['choice'])
    
    def stim_perf(stim, stim_type = ''):
        stim_unique, stim_unique_inv = np.unique(stim, return_inverse=True)
        print('num ',stim_type,' unique:', len(stim_unique))
        stim_perf_aR = []
        stim_perf_aNR = []
        
        for i in range(len(stim_unique)):        
            temp_aR = []
            temp_aNR = []
            if stim_unique[i] == 0: # do not consider zero stimulus change trials
                continue
            for j in range(len(stim_unique_inv)):
                if stim_unique_inv[j] == i:
                    
                    correct_percep = None
                    if stim_type == 'dsl':
                        if model_choice[j]==3 or model_choice[j]==4:
                            correct_percep = ((model_choice[j]==3) & (dsl[j]<0)) | ((model_choice[j]==4) & (dsl[j]>0))
                    elif stim_type == 'dsf':
                        if model_choice[j]==1 or model_choice[j]==2:
                            correct_percep = ((model_choice[j]==1) & (dsf[j]>0)) | ((model_choice[j]==2) & (dsf[j]<0))

                    if prevChoice[j] == prevCorrect[j] and correct_percep is not None:
                        temp_aR.append(int(correct_percep))
                    elif prevChoice[j] != prevCorrect[j] and correct_percep is not None:
                        temp_aNR.append(int(correct_percep))
                                
            if len(temp_aR) >= 10 and len(temp_aNR) >= 10:
                stim_perf_aR.append(np.count_nonzero(temp_aR)/len(temp_aR))
                stim_perf_aNR.append(np.count_nonzero(temp_aNR)/len(temp_aNR))
            
        return stim_perf_aR, stim_perf_aNR
    
    dsl_perf_aR, dsl_perf_aNR = stim_perf(dsl, stim_type='dsl')
    dsf_perf_aR, dsf_perf_aNR = stim_perf(dsf, stim_type='dsf')
    _, p_dsl = ztest(dsl_perf_aR, dsl_perf_aNR)
    _, p_dsf = ztest(dsf_perf_aR, dsf_perf_aNR)
    
    return dsl_perf_aR, dsl_perf_aNR, dsf_perf_aR, dsf_perf_aNR, p_dsl, p_dsf
#
#
#
def switchAfterStim(trial_params, model_output, account_noise=False, plot=True, saveplot_LtoF=None, saveplot_FtoL=None, NRnorm=False):
    """
    Plots the proportions of trials preceding a switch (SL to SF or SF to SL) that are small/big SL/SF, after reward/non-reward.
    
    Returns: LtoF (dict), FtoL (dict). Keys: RbLbF, RbLsF, RsLbF, RsLsF, NbLbF, NbLsF, NsLbF, NsLsF. 
            R (N) is reward (non-reward), b (s) is big change (small change), L (F) is spatial location (frequency).
    """
    N = len(trial_params[:])
 
    if model_output is not None:
        model_choice = np.argmax(model_output[:,-1,:], axis=1) + 1
        model_task = mf.get_tasks(model_choice)
        prev_dsl = np.empty(N)
        prev_dsf = np.empty(N)
        prev_err = np.empty(N)
        task = np.empty(N)
        b_switch = np.zeros(N, dtype=int)
        for i in range(N):
            if account_noise==False:
                prev_dsl[i] = trial_params[i]['dsl'][-2]
                prev_dsf[i] = trial_params[i]['dsf'][-2]
            else:
                prev_dsl[i] = trial_params[i]['dsl'][-2] + trial_params[i]['vis_noise'][-4]
                prev_dsf[i] = trial_params[i]['dsf'][-2] + trial_params[i]['vis_noise'][-3]
            prev_err[i] = trial_params[i]['err'][-2]
            task[i] = trial_params[i]['task'][-1]
            if model_task[i] != trial_params[i]['task'][-2]:
                b_switch[i] = 1
            
    elif model_output is None:
        b_switch = np.array(trial_params['m_switch'][1:])
        prev_dsl = np.array(trial_params['dsl'][:-1])
        prev_dsf = np.array(trial_params['dsf'][:-1])
        prev_err = np.array(trial_params['err'][:-1])
        task = np.array(trial_params['task'][1:])
        
    b_switch_inds = np.flatnonzero(b_switch)
    LtoF_inds = []
    FtoL_inds = []
    for s in b_switch_inds:
        if task[s]==2:
            LtoF_inds.append(s)
        elif task[s]==1:
            FtoL_inds.append(s)
    
    LtoF = dict(RbLbF=0, RbLsF=0, RsLbF=0, RsLsF=0, NbLbF=0, NbLsF=0, NsLbF=0, NsLsF=0)
    FtoL = dict(RbLbF=0, RbLsF=0, RsLbF=0, RsLsF=0, NbLbF=0, NbLsF=0, NsLbF=0, NsLsF=0)
    lc = np.median(np.abs(prev_dsl))
    fc = np.median(np.abs(prev_dsf))
    pm = 0.0
    norm_LF = 0
    norm_FL = 0
    
    for s in LtoF_inds:
        bL, sL = np.abs(prev_dsl[s])>lc+pm, np.abs(prev_dsl[s])<lc-pm
        bF, sF = np.abs(prev_dsf[s])>fc+pm, np.abs(prev_dsf[s])<fc-pm
        bLbF = int(bL and bF)
        bLsF = int(bL and sF)
        sLbF = int(sL and bF)
        sLsF = int(sL and sF)
        
        if (bLbF|bLsF|sLbF|sLsF)==True:
            norm_LF+=1
            if prev_err[s]==0:
                LtoF['RbLbF'] += bLbF
                LtoF['RbLsF'] += bLsF
                LtoF['RsLbF'] += sLbF
                LtoF['RsLsF'] += sLsF       
            elif prev_err[s]==1:
                LtoF['NbLbF'] += bLbF
                LtoF['NbLsF'] += bLsF
                LtoF['NsLbF'] += sLbF
                LtoF['NsLsF'] += sLsF
                
                
    for s in FtoL_inds:
        bL, sL = np.abs(prev_dsl[s])>lc+pm, np.abs(prev_dsl[s])<lc-pm
        bF, sF = np.abs(prev_dsf[s])>fc+pm, np.abs(prev_dsf[s])<fc-pm 
        bLbF = int(bL and bF)
        bLsF = int(bL and sF)
        sLbF = int(sL and bF)
        sLsF = int(sL and sF)

        if (bLbF|bLsF|sLbF|sLsF)==True:
            norm_FL+=1
            if prev_err[s]==0:
                FtoL['RbLbF'] += bLbF
                FtoL['RbLsF'] += bLsF
                FtoL['RsLbF'] += sLbF
                FtoL['RsLsF'] += sLsF       
            elif prev_err[s]==1:
                FtoL['NbLbF'] += bLbF
                FtoL['NbLsF'] += bLsF
                FtoL['NsLbF'] += sLbF
                FtoL['NsLsF'] += sLsF 
            
    # normalize
    LtoF = {k: v/norm_LF for k, v in LtoF.items()}
    FtoL = {k: v/norm_FL for k, v in FtoL.items()}
    
    if NRnorm==True:
        totN = 0
        totNbLbF = 0
        totNbLsF = 0
        totNsLbF = 0
        totNsLsF = 0   
        if model_output is not None:
            K = len(trial_params[0]['err'][:])
            for i in range(N):
                for j in range(K):
                    if trial_params[i]['err'][j]==1:
                        totN +=1
                        dsl = trial_params[i]['dsl'][j]
                        dsf = trial_params[i]['dsf'][j]
                        bL, sL = np.abs(dsl)>lc+pm, np.abs(dsl)<lc-pm
                        bF, sF = np.abs(dsf)>fc+pm, np.abs(dsf)<fc-pm
                        totNbLbF += int(bL and bF)
                        totNbLsF += int(bL and sF)
                        totNsLbF += int(sL and bF)
                        totNsLsF += int(sL and sF)
        elif model_output is None:
            tp = trial_params.copy()
            for i in range(tp.shape[0]):
                if tp.loc[i,'err']==1:
                    totN+=1
                    dsl = tp.loc[i,'dsl']
                    dsf = tp.loc[i,'dsf']
                    bL, sL = np.abs(dsl)>lc+pm, np.abs(dsl)<lc-pm
                    bF, sF = np.abs(dsf)>fc+pm, np.abs(dsf)<fc-pm
                    totNbLbF += int(bL and bF)
                    totNbLsF += int(bL and sF)
                    totNsLbF += int(sL and bF)
                    totNsLsF += int(sL and sF)                   
                
        NRnorm_LF = np.zeros(4)
        NRnorm_LF[0] = LtoF['NbLbF']/totNbLbF
        NRnorm_LF[1] = LtoF['NbLsF']/totNbLsF
        NRnorm_LF[2] = LtoF['NsLbF']/totNsLbF
        NRnorm_LF[3] = LtoF['NsLsF']/totNsLsF
        NRnorm_LF *= totN  
        NRnorm_FL = np.zeros(4)
        NRnorm_FL[0] = FtoL['NbLbF']/totNbLbF
        NRnorm_FL[1] = FtoL['NbLsF']/totNbLsF
        NRnorm_FL[2] = FtoL['NsLbF']/totNsLbF
        NRnorm_FL[3] = FtoL['NsLsF']/totNsLsF
        NRnorm_FL *= totN
        
        if plot==True:
            fig, ax = plt.subplots()
            width = 0.8
            labels = [r'$\Delta SL_{big},\,\Delta SF_{big}$', r'$\Delta SL_{big},\,\Delta SF_{small}$', r'$\Delta SL_{small},\,\Delta SF_{big}$', r'$\Delta SL_{small},\,\Delta SF_{small}$']
            colors = ['tab:blue', 'tab:cyan', 'tab:green', 'tab:olive']
            for x in range(4):
                ax.bar(x, NRnorm_LF[x], width=width, label=labels[x], color=colors[x])
            ax.bar([-0.4, 3.5, 4.25], [0,0,0], width=width)
            ax.set_title('Location to frequency switches', fontsize=16)
            ax.set_ylabel('Proportion of switch-preceding trials'+'\n $\div$ '+ 'proportion of all trials', fontsize=14)
            ax.set_xlabel('Non-rewarded trial', fontsize=16)
            ax.set_xticks([])
            plt.legend(fontsize=12, loc='upper right')    
            if saveplot_LtoF is not None:
                plt.savefig(saveplot_LtoF, bbox_inches='tight')
            
            fig, ax = plt.subplots()
            width = 0.8
            labels = [r'$\Delta SL_{big},\,\Delta SF_{big}$', r'$\Delta SL_{big},\,\Delta SF_{small}$', r'$\Delta SL_{small},\,\Delta SF_{big}$', r'$\Delta SL_{small},\,\Delta SF_{small}$']
            colors = ['tab:blue', 'tab:cyan', 'tab:green', 'tab:olive']
            for x in range(4):
                ax.bar(x, NRnorm_FL[x], width=width, label=labels[x], color=colors[x])
            ax.bar([-0.4, 3.5, 4.25], [0,0,0], width=width)
            ax.set_title('Frequency to location switches', fontsize=16)
            ax.set_ylabel('Proportion of switch-preceding trials'+'\n $\div$ '+ 'proportion of all trials', fontsize=14)
            ax.set_xlabel('Non-rewarded trial', fontsize=16)
            ax.set_xticks([])
            plt.legend(fontsize=12, loc='upper right')    
            if saveplot_FtoL is not None:
                plt.savefig(saveplot_FtoL, bbox_inches='tight')
        
        return NRnorm_LF, NRnorm_FL

    
    elif NRnorm==False:
        if plot==True:
            fig, ax = plt.subplots()
            x = np.array([0,5])
            width = 0.8
            ax.bar(x, [LtoF['RbLbF'], LtoF['NbLbF']], width=width, label=r'$\Delta SL_{big},\,\Delta SF_{big}$', color='tab:blue')
            ax.bar(x+1, [LtoF['RbLsF'], LtoF['NbLsF']], width=width, label=r'$\Delta SL_{big},\,\Delta SF_{small}$', color='tab:cyan')
            ax.bar(x+2, [LtoF['RsLbF'], LtoF['NsLbF']], width=width, label=r'$\Delta SL_{small},\,\Delta SF_{big}$', color='tab:green')
            ax.bar(x+3, [LtoF['RsLsF'], LtoF['NsLsF']], width=width, label=r'$\Delta SL_{small},\,\Delta SF_{small}$', color='tab:olive')  
            ax.set_xticks([1.5,6.5])
            ax.set_xticklabels(['reward', 'no reward'])
            ax.set_title('Location to frequency switches')
            ax.set_ylabel('Proportion of preceding trials')
            ax.set_xlabel('Trial preceding switch')
            plt.legend()
            if saveplot_LtoF is not None:
                plt.savefig(saveplot_LtoF, bbox_inches='tight')
        
            fig, ax = plt.subplots()
            x = np.array([0,5])
            width = 0.8
            ax.bar(x, [FtoL['RbLbF'], FtoL['NbLbF']], width=width, label=r'$\Delta SL_{big},\,\Delta SF_{big}$', color='tab:blue')
            ax.bar(x+1, [FtoL['RbLsF'], FtoL['NbLsF']], width=width, label=r'$\Delta SL_{big},\,\Delta SF_{small}$', color='tab:cyan')
            ax.bar(x+2, [FtoL['RsLbF'], FtoL['NsLbF']], width=width, label=r'$\Delta SL_{small},\,\Delta SF_{big}$', color='tab:green')
            ax.bar(x+3, [FtoL['RsLsF'], FtoL['NsLsF']], width=width, label=r'$\Delta SL_{small},\,\Delta SF_{small}$', color='tab:olive') 
            ax.set_xticks([1.5,6.5])
            ax.set_xticklabels(['reward', 'no reward'])
            ax.set_title('Frequency to location switches')
            ax.set_ylabel('Proportion of preceding trials')
            ax.set_xlabel('Trial preceding the switch')
            plt.legend()
            if saveplot_FtoL is not None:
                plt.savefig(saveplot_FtoL)
    
        return LtoF, FtoL

def exploreSwitch(trial_params, model_output):
    '''
    Parameters
    ----------
    trial_params : array
    model_output : array. If None, trial_params is monkey data.

    Returns
    -------
    SP (switch proportion): float
    SP_err (switch to wrong task): float
    SP_err_aR (switch to wrong task after reward): float
    '''
    N = len(trial_params[:])
 
    if model_output is not None:
        model_choice = np.argmax(model_output[:,-1,:], axis=1) + 1
        model_task = mf.get_tasks(model_choice)
        s = 0
        s_err = 0
        s_err_aR = 0
        for i in range(N):
            if model_task[i] != trial_params[i]['m_task'][-2]:
                s += 1
                if model_task[i] != trial_params[i]['task'][-1]:
                    s_err += 1
                    if trial_params[i]['err'][-2] == 0:
                        s_err_aR += 1   
            
    elif model_output is None:
        tp = trial_params
        s = tp[tp['m_switch']==1].shape[0]
        tp_es = tp[(tp['m_switch']==1) & (tp['task_err']==1)].copy()
        s_err = tp_es.shape[0]
        s_err_aR = 0
        for i in np.array(tp_es.index):
           s_err_aR += int(tp.loc[i-1,'err']==0)
    
    SP, SP_err, SP_err_aR = s/N, s_err/N, s_err_aR/N
      
    return SP, SP_err, SP_err_aR

def switchOnStim(trial_params, model_output, plot=False):
    """
    Gives the proportions of switch trials which have large vs. small stimulus changes, considering both features.
    Normalized by the proportion of all trials.
    
    Returns: LtoF (dict), FtoL (dict). Keys: sLsF, bLbF, sLbF, bLsF (s=small change, b=big change, L=location, F=frequency).
    """
    N = len(trial_params[:])
 
    if model_output is not None: # for an RNN model
        model_choice = np.argmax(model_output[:,-1,:], axis=1) + 1
        model_task = mf.get_tasks(model_choice)
        dsl = np.empty(N)
        dsf = np.empty(N)
        task = np.empty(N)
        b_switch = np.zeros(N, dtype=int)
        for i in range(N):
            dsl[i] = trial_params[i]['dsl'][-1]
            dsf[i] = trial_params[i]['dsf'][-1]
            task[i] = trial_params[i]['task'][-1]
            if model_task[i] != trial_params[i]['task'][-2]:
                b_switch[i] = 1
            
    elif model_output is None: # when trial_params is behavioral data
        b_switch = np.array(trial_params['m_switch'])
        dsl = np.array(trial_params['dsl'])
        dsf = np.array(trial_params['dsf'])
        task = np.array(trial_params['task'])
        
    b_switch_inds = np.flatnonzero(b_switch) # indices of switch trials, to be separated into L->F and F->L switches.
    LtoF_inds = []
    FtoL_inds = []
    for s in b_switch_inds:
        if task[s]==2:
            LtoF_inds.append(s)
        elif task[s]==1:
            FtoL_inds.append(s)
    
    LtoF = dict(bLbF=0, bLsF=0, sLbF=0, sLsF=0)
    FtoL = dict(bLbF=0, bLsF=0, sLbF=0, sLsF=0)
    lc = np.median(np.abs(dsl)) # dsl cutoff
    fc = np.median(np.abs(dsf)) # dsf cutoff
    norm_LF = 0
    norm_FL = 0
    
    for s in LtoF_inds:
        bL, sL = np.abs(dsl[s])>lc, np.abs(dsl[s])<lc
        bF, sF = np.abs(dsf[s])>fc, np.abs(dsf[s])<fc
        bLbF = int(bL and bF)
        bLsF = int(bL and sF)
        sLbF = int(sL and bF)
        sLsF = int(sL and sF)
        
        if (bLbF|bLsF|sLbF|sLsF)==True:
            norm_LF += 1
            LtoF['bLbF'] += bLbF
            LtoF['bLsF'] += bLsF
            LtoF['sLbF'] += sLbF
            LtoF['sLsF'] += sLsF       
                
                
    for s in FtoL_inds:
        bL, sL = np.abs(dsl[s])>lc, np.abs(dsl[s])<lc
        bF, sF = np.abs(dsf[s])>fc, np.abs(dsf[s])<fc
        bLbF = int(bL and bF)
        bLsF = int(bL and sF)
        sLbF = int(sL and bF)
        sLsF = int(sL and sF)

        if (bLbF|bLsF|sLbF|sLsF)==True:
            norm_FL += 1
            FtoL['bLbF'] += bLbF
            FtoL['bLsF'] += bLsF
            FtoL['sLbF'] += sLbF
            FtoL['sLsF'] += sLsF       
            
    # normalize
    LtoF = {k: v/norm_LF for k, v in LtoF.items()}
    FtoL = {k: v/norm_FL for k, v in FtoL.items()}
    
    # compute proportion of all trials for each trial type (bLsF, etc.)
    tot = 0
    totbLbF = 0
    totbLsF = 0
    totsLbF = 0
    totsLsF = 0   
    if model_output is not None: # for an RNN model
        for i in range(N):
            tot +=1
            dsl = trial_params[i]['dsl'][-1]
            dsf = trial_params[i]['dsf'][-1]
            bL, sL = np.abs(dsl)>lc, np.abs(dsl)<lc
            bF, sF = np.abs(dsf)>fc, np.abs(dsf)<fc
            totbLbF += int(bL and bF)
            totbLsF += int(bL and sF)
            totsLbF += int(sL and bF)
            totsLsF += int(sL and sF)
    elif model_output is None: # when trial_params is behavioral data
        tp = trial_params.copy()
        for i in range(tp.shape[0]):
            tot+=1
            dsl = tp.loc[i,'dsl']
            dsf = tp.loc[i,'dsf']
            bL, sL = np.abs(dsl)>lc, np.abs(dsl)<lc
            bF, sF = np.abs(dsf)>fc, np.abs(dsf)<fc
            totbLbF += int(bL and bF)
            totbLsF += int(bL and sF)
            totsLbF += int(sL and bF)
            totsLsF += int(sL and sF)                   
                
    LtoF['bLbF'] *= tot/totbLbF
    LtoF['bLsF'] *= tot/totbLsF
    LtoF['sLbF'] *= tot/totsLbF
    LtoF['sLsF'] *= tot/totsLsF 
    
    FtoL['bLbF'] *= tot/totbLbF
    FtoL['bLsF'] *= tot/totbLsF
    FtoL['sLbF'] *= tot/totsLbF
    FtoL['sLsF'] *= tot/totsLsF
    
    return LtoF, FtoL