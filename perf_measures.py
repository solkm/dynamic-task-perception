#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 00:38:58 2021

@author: Sol
"""

import numpy as np
import model_functions as mf

def choicePrediction(trial_params, model_output):
    
    N = len(trial_params[:])
    model_choice = np.argmax(model_output[:,-1,:], axis=1) + 1
    errs = np.empty(N, dtype='int')
    m_switches = np.empty(N, dtype='int')
    task_errs = np.empty(N, dtype='int')
    perc_errs = np.empty(N, dtype='int')
    choicePred = 0
    errPred = 0
    taskErrPred = 0
    percErrPred = 0
    switchPred = 0
    errSwitchPred = 0
    numErrSwitch = 0
    
    for i in range(N):
        errs[i] = trial_params[i]['err'][-1]
        task_errs[i] =  trial_params[i]['task_err'][-1]
        perc_errs[i] = trial_params[i]['perc_err'][-1]
        m_switches[i] = trial_params[i]['m_switch'][-1]     
        
        if model_choice[i] == trial_params[i]['choice'][-1]:
            choicePred += 1
            if errs[i] == 1:
                errPred += 1
                if task_errs[i] == 1:
                    taskErrPred += 1
                if perc_errs[i] == 1:
                    percErrPred += 1
            if m_switches[i] == 1:
                switchPred += 1
                if errs[i] == 1:
                    errSwitchPred += 1
        if errs[i] == 1 and m_switches[i] == 1:
            numErrSwitch += 1

    numErr = np.count_nonzero(errs)
    numSwitch = np.count_nonzero(m_switches) 
    pred = {
        'p_choice' : 100*choicePred/N,
        'p_corr' : 100*(choicePred-errPred)/(N-numErr),
        'p_err' : 100*errPred/numErr,
        'p_terr' : 100*taskErrPred/np.count_nonzero(task_errs),
        'p_perr' : 100*percErrPred/np.count_nonzero(perc_errs),
        'p_switch' : 100*switchPred/numSwitch,
        'p_errswitch' : 100*errSwitchPred/numErrSwitch,
        'p_corrswitch' : 100*(switchPred-errSwitchPred)/(numSwitch-numErrSwitch),
    } 
    return pred

def accuracyMeasures(trial_params, model_output):
    
    measures = np.zeros(8)
    measures[0] = mf.get_modelAccuracy(model_output, trial_params, fromTP=True)
    measures[1] = mf.get_modelTaskAccuracy(model_output, trial_params)
    measures[2] = mf.get_modelPercAccuracy(model_output, trial_params)
    measures[3] = mf.modelSwitchPercentage(model_output, trial_params)
    
    measures[4] = mf.get_monkeyAccuracy(trial_params)
    measures[5] = mf.get_monkeyTaskAccuracy(trial_params)
    measures[6] = mf.get_monkeyPercAccuracy(trial_params)
    measures[7] = mf.monkeySwitchPercentage(trial_params)

    return measures
