#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 12:13:24 2021

@author: Sol
"""

from FourAFC_taskModels import StimHist_constant
from psychrnn.backend.models.basic import Basic
import numpy as np
import scipy.io
import visfunctions as vf
import csv
import matplotlib.pyplot as plt
from model_functions import get_modelAccuracy
from statsmodels.stats.weightstats import ztest

tParams_nDist = scipy.io.loadmat('./mat_files/tParams_nDist.mat')
#tParams = scipy.io.loadmat(r'tParams.mat')

N_rec = 100
N_batch = 100
dt = 10
tau = 100
T = 2000
K = 10

'''
# ---------------------- Set up a basic model ---------------------------

pd1 = StimHist_constant(dt, tau, T, N_batch, K, dat1=tParams_nDist, stim_noise=noise, trainChoice=False, nDist=True)
network_params1 = pd1.get_task_params() # get the params passed in and defined in pd
network_params1['name'] = 'nDist_correct' # name the model uniquely if running mult models in unison
network_params1['N_rec'] = N_rec # set the number of recurrent units in the model

pd2 = StimHist_constant(dt, tau, T, N_batch, K, dat1=tParams_nDist, stim_noise=noise, trainChoice=True, nDist=True)
network_params2 = pd2.get_task_params()
network_params2['name'] = 'nDist_choice'
network_params2['N_rec'] = N_rec

# ---------------Set connectivity if not fully connected ---------------
N_in = pd1.N_in
N_out = pd1.N_out

# Set the default connectivity as fully connected
input_connectivity = np.ones((N_rec, N_in))
rec_connectivity = np.ones((N_rec, N_rec))
output_connectivity = np.ones((N_out, N_rec))

# Specify certain connections to disallow

input_connectivity[N_rec//5:N_rec, -2:] = 0 #task region receives no stimulus inputs
input_connectivity[0:N_rec//5, 0:-2] = 0 #perception region receives no task input

network_params1['input_connectivity'] = input_connectivity
network_params1['rec_connectivity'] = rec_connectivity
network_params1['output_connectivity'] = output_connectivity

model1 = Basic(network_params1)

network_params2['input_connectivity'] = input_connectivity
network_params2['rec_connectivity'] = rec_connectivity
network_params2['output_connectivity'] = output_connectivity

model2 = Basic(network_params2)

# ---------------------- Train a basic model ---------------------------

train_params = {}
train_params['training_iters'] = 100000

losses1, initialTime1, trainTime1 = model1.train(pd1, train_params) # train model to perform pd task
#losses2, initialTime2, trainTime2 = model2.train(pd2, train_params)

# ---------------------- Save the model weights ---------------------------
model1.save('./saved_weights/nDist_correct')
model2.save('./saved_weights/nDist_choice')

# ---------------------- Test the trained model ---------------------------
x1, target_output1, mask1, trial_params1 = pd1.get_trial_batch() # get pd task inputs and outputs to test
model_output1, state_var1 = model1.test(x1) # run the model on input x

model_accuracy1 = get_modelAccuracy(model_output1, target_output1)
print('Accuracy, targeting on correct choice: ', model_accuracy1)

x2, target_output2, mask2, trial_params2 = pd2.get_trial_batch() # get pd task inputs and outputs to test
model_output2, state_var2 = model2.test(x2) # run the model on input x

model_accuracy2 = get_modelAccuracy(model_output2, target_output2)
print('Accuracy, targeting on monkey choice: ', model_accuracy2)
'''

# ---------------------- OR: Load and test from a saved model ---------------------------
pd1 = StimHist_constant(dt, tau, T, N_batch, K, dat1=tParams_nDist, in_noise=0.1, const_noise=0.0, trainChoice=False, nDist=True)
network_params1 = pd1.get_task_params() # get the params passed in and defined in pd
network_params1['name'] = 'nDist_choice' # name the model uniquely if running mult models in unison
network_params1['N_rec'] = N_rec # set the number of recurrent units in the model
network_params1['load_weights_path'] = './saved_weights/nDist_choice.npz'

model1 = Basic(network_params1)
x1, target_output1, mask1, trial_params1 = pd1.get_trial_batch() # get pd task inputs and outputs to test
model_output1, state_var1 = model1.test(x1) # run the model on input x

model_accuracy1 = get_modelAccuracy(model_output1, target_output1)
print('Accuracy, targeting correct choice: ', model_accuracy1)

# ---------------------- Behavioral analysis ---------------------------
from behavior_analysis import plot_behavioral_0

trial_params_large1 = trial_params1.copy()
model_output_large1 = model_output1.copy()

for i in range(999):
    x, target_output, mask, trial_params = pd1.get_trial_batch()
    model_output, state_var = model1.test(x)
    
    trial_params_large1 = np.concatenate((trial_params_large1, trial_params), axis=0)
    model_output_large1 = np.concatenate((model_output_large1, model_output), axis=0)

dsl_perf_aR1, dsl_perf_aNR1, dsf_perf_aR1, dsf_perf_aNR1 = plot_behavioral_0(trial_params_large1, model_output_large1, plot=False) #NEED LARGE DATASET

t_dsl1,p_dsl1 = ztest(dsl_perf_aR1, dsl_perf_aNR1)
t_dsf1,p_dsf1 = ztest(dsf_perf_aR1, dsf_perf_aNR1)

plt.figure()
plt.scatter(dsl_perf_aR1, dsl_perf_aNR1, edgecolors='m', facecolors='none', label='SL choices, p=%3.2e'%p_dsl1)
plt.scatter(dsf_perf_aR1, dsf_perf_aNR1, edgecolors='c', facecolors='none', label='SF choices, p=%3.2e'%p_dsf1)
plt.scatter(np.mean(dsl_perf_aR1), np.mean(dsl_perf_aNR1), s=60, edgecolors='k', facecolors='m')
plt.scatter(np.mean(dsf_perf_aR1), np.mean(dsf_perf_aNR1), s=60, edgecolors='k', facecolors='c')
X = np.linspace(0,1,100)
plt.plot(X,X, color='slategrey')
minperf = np.min(np.concatenate((dsl_perf_aR1, dsl_perf_aNR1, dsf_perf_aR1, dsf_perf_aNR1)))
plt.xlim(left = min(minperf-0.1, 0.8))
plt.ylim(bottom = min(minperf-0.1, 0.8))
plt.xlabel('After a rewarded trial', fontsize=14)
plt.ylabel('After an unrewarded trial', fontsize=14)
plt.title('Perceptual performance for the same stimulus', fontsize=14)
plt.legend(fontsize=10, loc='upper left')
plt.show()

# ---------------------- Plot the results ---------------------------
'''
plt.plot(losses1, label='loss training on correct choice')
plt.plot(losses2, label='loss training on monkey choice')
plt.title('Loss during training')
plt.ylabel('Loss')
plt.xlabel('Training iteration')
plt.legend()



trial = 0 #the trial number to plot

vf.plot_trial_inputs(x1, trial, dt, "Inputs, noise=%4.3f"%noise)

vf.plot_trial_outputs(model_output1, trial, dt, "Outputs")

vf.plot_trial_statevars(state_var1, trial, dt, "Evolution of State Variables")

weights = model1.get_weights()
vf.plot_weights(weights['W_rec'], "Recurrent weights")

vf.plot_weights(weights['W_in'], "Input weights")

vf.plot_weights(weights['W_out'], "Output weights")

nDist_correctCorrect = []
nDist_choiceChoice = []
nDist_correctChoice = []
nDist_choiceCorrect = []

for i in range(50):
    x, target_output, mask, trial_params = pd1.get_trial_batch() # get pd task inputs and outputs to test
    model_output, state_var = model1.test(x) # run the model on input x
    
    model_accuracy = get_modelAccuracy(model_output, target_output)
    nDist_correctCorrect.append(model_accuracy)
    #
    #
    x, target_output, mask, trial_params = pd2.get_trial_batch() # get pd task inputs and outputs to test
    model_output, state_var = model2.test(x) # run the model on input x
    
    model_accuracy = get_modelAccuracy(model_output, target_output)
    nDist_choiceChoice.append(model_accuracy)
    #
    #
    x, target_output, mask, trial_params = pd2.get_trial_batch() # get pd task inputs and outputs to test
    model_output, state_var = model1.test(x) # run the model on input x
    
    model_accuracy = get_modelAccuracy(model_output, target_output)
    nDist_correctChoice.append(model_accuracy)
    #
    #
    x, target_output, mask, trial_params = pd1.get_trial_batch() # get pd task inputs and outputs to test
    model_output, state_var = model2.test(x) # run the model on input x
    
    model_accuracy = get_modelAccuracy(model_output, target_output)
    nDist_choiceCorrect.append(model_accuracy)

with open('model_trainTestAccuracies.csv', 'a') as file:
    writer = csv.writer(file)
    writer.writerow(['decodedStim_modularIn'])
    writer.writerow(['nDist_correctCorrect', [str(x) for x in nDist_correctCorrect]])
    writer.writerow(['nDist_choiceChoice', [str(x) for x in nDist_choiceChoice]])
    writer.writerow(['nDist_correctChoice', [str(x) for x in nDist_correctChoice]])
    writer.writerow(['nDist_choiceCorrect', [str(x) for x in nDist_choiceCorrect]])


accuracies = np.array([np.mean(nDist_correctCorrect), np.mean(nDist_choiceCorrect), np.mean(nDist_choiceChoice), np.mean(nDist_correctChoice)])
error = np.array([np.std(nDist_correctCorrect), np.std(nDist_choiceCorrect), np.std(nDist_choiceChoice), np.std(nDist_correctChoice)])
fig, ax = plt.subplots()
ax.bar(np.arange(len(accuracies)), accuracies, yerr=error, align='center', alpha=0.5)
ax.set_xticks(np.arange(len(accuracies)))
ax.set_xticklabels(['correct-correct', 'monkey-correct', 'monkey-monkey', 'correct-monkey'])
ax.set_title('Train-Test Accuracies for Decoded Perceptual Inputs')
ax.yaxis.grid(True)

t1,p1 = ztest(nDist_choiceChoice, nDist_correctChoice)
print('monkey-monkey, correct-monkey, p =', p1)
'''