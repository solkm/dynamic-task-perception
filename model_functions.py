#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 15:18:12 2021

@author: Sol
"""

import numpy as np

def get_Accuracies(model_output, trial_params):
    N = trial_params.shape[0]
    #choice = np.argmax(model_output[:,-1,:], axis=1)
    choice = np.argmax(model_output[:,-1,:-1], axis=1) # don't include non-response as a possible choice
    correct = 0
    taskerrs = 0
    percerrs = 0
    noresponse = 0
    taskerr_trials = []
    percerr_trials = []
    nr_trials = []
    for i in range(N):
        if trial_params[i]['targ']==choice[i]:
            correct += 1
            continue
        if trial_params[i]['task']==0:
            if choice[i]==0 or choice[i]==1:
                percerrs += 1
                percerr_trials.append(i)
            else:
                taskerrs += 1
                taskerr_trials.append(i)
                if (choice[i]==2 and trial_params[i]['b1'] < trial_params[i]['b2']) or (choice[i]==3 and trial_params[i]['b1'] > trial_params[i]['b2']):
                    percerrs += 1
                    percerr_trials.append(i)
        elif trial_params[i]['task']==1:
            if choice[i]==2 or choice[i]==3:
                percerrs += 1
                percerr_trials.append(i)
            else:
                taskerrs += 1
                taskerr_trials.append(i)
                if (choice[i]==0 and trial_params[i]['a1'] < trial_params[i]['a2']) or (choice[i]==1 and trial_params[i]['a1'] > trial_params[i]['a2']):
                    percerrs += 1
                    percerr_trials.append(i)
        if choice[i]==4:
            noresponse+=1
            nr_trials.append(i)

    task_acc = 1 - taskerrs/N
    perc_acc = 1 - percerrs/N
    overall_acc = correct/N
    noresponse /= N
    
    return task_acc, perc_acc, overall_acc, noresponse, np.array(taskerr_trials), np.array(percerr_trials), np.array(nr_trials)
#
#
# Get the network model's accuracy from model output and target output
#
def get_modelAccuracy(model_output, target_output, fromTP=False):
    
    if fromTP==False:
        pred=model_output[:,-1,:]
        target=target_output[:,-1,:]
        dww=np.concatenate(([np.argmax(pred, axis=1)],[np.argmax(target,axis=1)]),axis=0)
        loss=np.diff(np.transpose(dww))
        model_accuracy = 1.0 - np.count_nonzero(loss)/len(loss)
    
    else:
        model_choice = np.argmax(model_output[:,-1,:], axis=1) + 1
        correct=0
        for i in range(len(model_choice)):
            if model_choice[i] == target_output[i]['correct'][-1]:
                correct += 1
        model_accuracy = correct/len(model_choice)
        
    return model_accuracy
#
#
#
def get_monkeyAccuracy(trial_params):
    N = len(trial_params[:])
    monkey_choice = [ trial_params[i]['choice'][-1] for i in range(N) ]
    correct_choice = [ trial_params[i]['correct'][-1] for i in range(N) ]  
    monkey_accuracy = 1 - np.count_nonzero(np.subtract(monkey_choice, correct_choice))/N
    
    return monkey_accuracy
#
#
# Analyze consecutive error streaks/overall accuracy
#
def consecutive_errors(choice, correct, indThresh=None):
    errors = np.subtract(np.array(choice, dtype=float), np.array(correct, dtype=float))
    numErr = np.count_nonzero(errors)
    accuracy = 1 - numErr/len(errors)
    consErr = 0
    consErrs = []
    long = []
    for i in range(len(errors)):
        if errors[i] != 0:
            consErr += 1
        if errors[i] == 0 and consErr != 0:
            if indThresh is not None and consErr>indThresh:
                long.append(i-consErr)
            consErrs.append(consErr)
            consErr = 0
    if indThresh is not None:      
        return consErrs, accuracy, long
    else:
        return consErrs,accuracy
#
#
#
def get_monkeyTaskAccuracy(trial_params):
    
    N = len(trial_params[:])
    monkey_choice = [ trial_params[i]['choice'][-1] for i in range(N) ]
    correct_task = [ trial_params[i]['task'][-1] for i in range(N) ]
    monkey_task = get_tasks(monkey_choice)
    monkey_taskaccuracy = 1 - np.count_nonzero(np.subtract(monkey_task, correct_task))/N
    
    return monkey_taskaccuracy
#
#
#
def get_monkeyPercAccuracy(trial_params):
    
    N = len(trial_params[:])
    monkey_choice = [ trial_params[i]['choice'][-1] for i in range(N) ]
    dsl = [ trial_params[i]['dsl'][-1] for i in range(N) ]
    dsf = [ trial_params[i]['dsf'][-1] for i in range(N) ]
    percerr_trials = []
    nc=0
    for i in range(N):
        if monkey_choice[i]==1 and dsf[i]>0:
            nc+=1
        elif monkey_choice[i]==2 and dsf[i]<0:
            nc+=1
        elif monkey_choice[i]==3 and dsl[i]<0:
            nc+=1
        elif monkey_choice[i]==4 and dsl[i]>0:
            nc+=1
        else:
            percerr_trials.append(i)
    perc_accuracy = nc/N
    
    return perc_accuracy, percerr_trials
#
#
# Get accuracies 1 to n trials after an error (followed by correct trials)
#
def accuracy_nAfterError(trial_params, n, acc_type):
    """
    Parameters
    ----------
    trial_errors : array
        1 if an error was made on the trial, 0 if the trial was correct.
    n : int
        The maximum trials after an error to analyze.

    Returns
    -------
    accuracies : array of length n
        An array of accuracies 1 to n trials after an error (followed by correct trials)
    """
    correct_count = np.zeros(n, dtype='int')
    total_trials = np.zeros(n, dtype='int')
    perc_errors = trial_params['r_DERR'][0]
    N = len(trial_params['r_CHOICE'][0])

    all_errors = np.zeros(N, dtype='int')
    for i in range(N):
        all_errors[i] = int(trial_params['r_CHOICE'][0][i] != trial_params['r_CORRECT'][0][i])
 
    error_inds = np.flatnonzero(all_errors)
        
    if acc_type=='perceptual':
        errors = perc_errors
    elif acc_type=='overall':
        errors = all_errors
    
    for i in error_inds:
        if i+n+1 < N:
            for j in range(1, n+1):
                total_trials[j-1] += 1     
                if errors[i+j]==0:
                    correct_count[j-1] += 1
                else:
                    break               
    accuracies = np.divide(correct_count,total_trials)
    return accuracies
#
#
#
def task_switches(task):
    """
    Parameters:
    task (array or list): an array of tasks
    
    Returns:
    switch (array): an array of task switches, 0 for no switch (same task as previous trial), 1 for switch
    numSwitches (int): total number of switches
    """
    switch = np.zeros(len(task), dtype='int8')
    for i in range(1,len(task)):
        if task[i] != task[i-1]:
            switch[i] = 1
    numSwitches = np.count_nonzero(switch)
    return switch, numSwitches
#
#
# Analyze consecutive task streaks
#
def task_streaks(task):
    switch = task_switches(task)
    consTask = 0
    taskStreaks = []
    
    for i in range(len(switch)):
        if switch[i] == 0:
            consTask += 1
        if switch[i] != 0 and consTask != 0:
            taskStreaks.append(consTask)
            consTask = 0
    return taskStreaks
#
#
#
def get_tasks(choices):
    choices=list(choices)
    tasks = np.zeros(len(choices), dtype='int8')
    for i in range(len(choices)):
        if choices[i]==3 or choices[i]==4:
            tasks[i] = 1
        elif choices[i]==1 or choices[i]==2:
            tasks[i] = 2
    return tasks
#
#
# Get the network model's TASK accuracy from model output and target output
#
def get_modelTaskAccuracy(model_output, trial_params):
    
    model_choice = np.argmax(model_output[:,-1,:], axis=1) + 1
    model_task = get_tasks(model_choice)
    correct_task = [ trial_params[i]['task'][-1] for i in range(len(trial_params[:])) ]
    
    numTaskErrs = np.count_nonzero(np.subtract(model_task, correct_task))
    taskaccuracy = 1 - numTaskErrs/len(model_task)
    
    return taskaccuracy
#
#
# Get the network model's PERCEPTUAL accuracy from model output and target output
#
def get_modelPercAccuracy(model_output, trial_params):
    
    N = len(trial_params[:])
    model_choice = np.argmax(model_output[:,-1,:], axis=1) + 1    
    dsl = [ trial_params[i]['dsl'][-1] for i in range(N) ]
    dsf = [ trial_params[i]['dsf'][-1] for i in range(N) ]
    
    nc=0
    for i in range(N):
        if model_choice[i]==1 and dsf[i]>0:
            nc+=1
        elif model_choice[i]==2 and dsf[i]<0:
            nc+=1
        elif model_choice[i]==3 and dsl[i]<0:
            nc+=1
        elif model_choice[i]==4 and dsl[i]>0:
            nc+=1
    
    perc_accuracy = nc/N
    
    return perc_accuracy
#   
#
#
def get_monkeyTaskAccuracy(trial_params):
    
    N = len(trial_params[:])
    monkey_choice = [ trial_params[i]['choice'][-1] for i in range(N) ]
    correct_task = [ trial_params[i]['task'][-1] for i in range(N) ]
    monkey_task = get_tasks(monkey_choice)
    monkey_taskaccuracy = 1 - np.count_nonzero(np.subtract(monkey_task, correct_task))/N
    
    return monkey_taskaccuracy
#
#
#
def get_monkeyPercAccuracy(trial_params):
    
    N = len(trial_params[:])
    monkey_choice = [ trial_params[i]['choice'][-1] for i in range(N) ]
    dsl = [ trial_params[i]['dsl'][-1] for i in range(N) ]
    dsf = [ trial_params[i]['dsf'][-1] for i in range(N) ]
    percerr_trials = []
    nc=0
    for i in range(N):
        if monkey_choice[i]==1 and dsf[i]>0:
            nc+=1
        elif monkey_choice[i]==2 and dsf[i]<0:
            nc+=1
        elif monkey_choice[i]==3 and dsl[i]<0:
            nc+=1
        elif monkey_choice[i]==4 and dsl[i]>0:
            nc+=1
        else:
            percerr_trials.append(i)
    perc_accuracy = nc/N
    
    return perc_accuracy, percerr_trials
#
#
#
def modelSwitchPercentage(model_output, trial_params):
    
    model_choice = np.argmax(model_output[:,-1,:], axis=1) + 1
    model_task = get_tasks(model_choice)
    prevMonkeyTask = get_tasks([trial_params[i]['choice'][-2] for i in range(len(trial_params[:]))])
    modelSwitches = np.zeros(len(model_task))
    for i in range(len(model_task)):
        if model_task[i] != prevMonkeyTask[i]:
            modelSwitches[i] = 1
    
    return np.count_nonzero(modelSwitches)/len(modelSwitches)
#
#
#
def monkeySwitchPercentage(trial_params):
    
    N = len(trial_params[:])
    monkey_choice = [ trial_params[i]['choice'][-1] for i in range(N) ]
    monkey_task = get_tasks(monkey_choice)
    monkey_prevTask = get_tasks([ trial_params[i]['choice'][-2] for i in range(N) ])
    monkey_switchprob = np.count_nonzero(np.subtract(monkey_task, monkey_prevTask))/N
    
    return monkey_switchprob
#
#
#
def probTaskSwitch_nAfterSwitch(choices, tasks, n):
    """
    Parameters:
    tasks (array or list): an array of tasks as defined by the experimentor
    choices (array or list): the model or monkey's choices
    n (int): how many trials after the task switch to consider
    
    Returns:
    switch_probs (array): an array of probabilities that the monkey first switches tasks 1 to n trials after the actual task switch
    """
    chosen_tasks = get_tasks(choices)
    actual_switches, numAct = task_switches(tasks)
    behavior_switches, numBeh = task_switches(chosen_tasks)
    actual_ind = np.flatnonzero(actual_switches)
    switch_probs = np.zeros(n)
    normalizer = 0
    
    for i in actual_ind:
        if i+n > len(behavior_switches)-1:
            continue
        #if chosen_tasks[i]==tasks[i]: # don't include switches where the monkey happened to already be doing the post-switch task on the switch trial
        #    continue
        if any([actual_switches[x]==1 for x in range(i+1, i+n+1)]): # don't include switches where the next switch is within n trials
            continue 
        
        normalizer += 1
        for j in range(1, n+1):
            if behavior_switches[i+j] == 1: # detect behavioral switches and break after the first behavioral switch
                switch_probs[j-1] += 1
                break
     
    switch_probs = np.divide(switch_probs, normalizer)
    
    return switch_probs
#
#
#
def probNoSwitchYet_nAfterSwitch(choices, tasks, n):
    """
    Parameters:
    tasks (array or list): an array of tasks as defined by the experimentor
    choices (array or list): the model or monkey's choices
    n (int): how many trials after the task switch to consider
    
    Returns:
    noswitch_probs (array): an array of probabilities that the monkey has not already switched tasks at the start of the 1st to nth trial after the task switch
    """
    firstSwitchProbs = probTaskSwitch_nAfterSwitch(choices, tasks, n)
    hasSwitched = np.zeros(n)
    for j in range(1,n):
        hasSwitched[j] = np.sum(firstSwitchProbs[:j])
    return 1-hasSwitched
#
#
#
def gendat_undetectedSwitch_nBefore(dat, n, K, cond_needSwitch=False):
    """
    Parameters
    ----------
    dat : dict
        Dataset from which to generate the test batch dataset.
    n : int
        The number of trials before the current one with an undetected task switch.
    K : int
        The total considered trials parameter of the StimHist model.
    cond_needSwitch: bool, optional
        If true, add the condition that the monkey was doing the correct pre-switch task before the switch. Default is False.

    Returns
    -------
    gendat : list (of dictionaries)
        A list of dictionaries, with each dictionary containing parameters for one trial with an undetected task switch n trials before the current one.
    """
    
    if n>K-1 or n<0:
        return print('n must be between 0 and K-1')
    
    trial_params = np.vstack((dat['r_TASK'][0], dat['r_CORRECT'][0], dat['r_CHOICE'][0], dat['r_DSL'][0], dat['r_DSF'][0]))
    
    switches, _ = task_switches(trial_params[0])
    switch_ind = np.flatnonzero(switches)
    chosen_task = get_tasks(trial_params[2])
    beh_switch, _ = task_switches(chosen_task)
    
    gendat = []
    
    for s in switch_ind:
        if cond_needSwitch==True and chosen_task[s-1] != trial_params[0][s-1]: # if specified, do not include trials where the monkey was already performing the post-switch task
                continue
        if s+n < len(switches):
            temp_params = np.zeros((5, K))
            for i in range(0, K): #store trial params around the switch
                if switches[s-(K-n-1)+i]==1 and i!=K-n-1: #break if we've reached another switch
                    break
                if K-n-1 <= i < K-1 and beh_switch[s-(K-n-1)+i]==1: #break if the monkey switches on or after the task switch, before the current trial
                    break
                temp_params[:, i] = trial_params[:,s-(K-n-1)+i]
            if all(t!=0 for t in temp_params[0,:]):
                gendat.append(dict(task = temp_params[0,:], correct = temp_params[1,:], choice = temp_params[2,:], dsl = temp_params[3,:], dsf = temp_params[4,:]))
                
    return gendat
#
#
#
def gendat_errorTrials(dat, K):
    """
    Parameters
    ----------
    dat : dict
        Dataset from which to generate the test batch dataset.
    K : int
        The total considered trials parameter of the StimHist model.

    Returns
    -------
    gendat_error : list (of dictionaries)
        A list of dictionaries, with each dictionary containing parameters for a trial on which the monkey made an error.
    gendat_correct : list (of dictionaries)
        A list of dictionaries, with each dictionary containing parameters for a trial on which the monkey made the correct choice.
    """
    
    trial_params = np.vstack((dat['r_TASK'][0], dat['r_CORRECT'][0], dat['r_CHOICE'][0], dat['r_DSL'][0], dat['r_DSF'][0]))
    
    gendat_error = []
    gendat_correct = []
    
    for i in range(len(trial_params[0])-K):
        temp_params = trial_params[:,i:i+K]
        if trial_params[1,i+K] == trial_params[2,i+K]:
            gendat_correct.append(dict(task = temp_params[0,:], correct = temp_params[1,:], choice = temp_params[2,:], dsl = temp_params[3,:], dsf = temp_params[4,:]))
        else:
            gendat_error.append(dict(task = temp_params[0,:], correct = temp_params[1,:], choice = temp_params[2,:], dsl = temp_params[3,:], dsf = temp_params[4,:]))
    
    return gendat_error, gendat_correct
#
#
#
def gendat_taskErrorTrials(dat, K):
    """
    Parameters
    ----------
    dat : dict
        Dataset from which to generate the test batch dataset.
    K : int
        The total considered trials parameter of the StimHist model.

    Returns
    -------
    gendat_error : list (of dictionaries)
        A list of dictionaries, with each dictionary containing parameters for a trial on which the monkey made a TASK error.
    gendat_correct : list (of dictionaries)
        A list of dictionaries, with each dictionary containing parameters for a trial on which the monkey made the correct TASK choice.
    """
    
    trial_params = np.vstack((dat['r_TASK'][0], dat['r_CORRECT'][0], dat['r_CHOICE'][0], dat['r_DSL'][0], dat['r_DSF'][0]))
    chosen_tasks = get_tasks(trial_params[2,:])
    
    gendat_error = []
    gendat_correct = []
    
    for i in range(len(trial_params[0])-K):
        
        temp_params = trial_params[:,i:i+K]
        
        if trial_params[0,i+K] == chosen_tasks[i+K]:
            gendat_correct.append(dict(task = temp_params[0,:], correct = temp_params[1,:], choice = temp_params[2,:], dsl = temp_params[3,:], dsf = temp_params[4,:]))
            
        else:
            gendat_error.append(dict(task = temp_params[0,:], correct = temp_params[1,:], choice = temp_params[2,:], dsl = temp_params[3,:], dsf = temp_params[4,:]))
    
    return gendat_error, gendat_correct
