#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 18:04:15 2021

@author: Sol
"""

from FourAFC_taskModels import PerceptionModule
from psychrnn.backend.models.basic import Basic
import numpy as np
import scipy.io
import visfunctions as vf
import csv
import matplotlib.pyplot as plt

N_rec = 50
N_batch = 100
dt = 10
tau = 100
T = 2000

constNoises = [0.8,0.8]

tParams = scipy.io.loadmat(r'tParams.mat')

dat=tParams

for const_noise in constNoises:
    # ---------------------- Set up a basic model ---------------------------
    pd = PerceptionModule(dt, tau, T, N_batch, dat, in_noise=0.1, const_noise=const_noise)
    network_params = pd.get_task_params() # get the params passed in and defined in pd
    network_params['name'] = 'model' # name the model uniquely if running mult models in unison
    network_params['N_rec'] = N_rec # set the number of recurrent units in the model
    network_params['autapses'] = False #no autapses
    network_params['dale_ratio'] = .8 #80 percent excitatory, 20 percent inhibitory neurons
    network_params['rec_noise'] = 0.1

    model = Basic(network_params) # instantiate a basic vanilla RNN
    
    # ---------------------- Train a basic model ---------------------------
    train_params = {}
    train_params['training_iters'] = 50000
    model.train(pd, train_params) # train model to perform pd task
    
    # ---------------------- Test the trained model ---------------------------
    x, target_output, mask, trial_params = pd.get_trial_batch() # get pd task inputs and outputs
    model_output, state_var = model.test(x) # run the model on input x
    
    
    pred=model_output[:,-1,:]
    target=target_output[:,-1,:]
    dww=np.concatenate(([np.argmax(pred, axis=1)],[np.argmax(target,axis=1)]),axis=0)
    loss=np.diff(np.transpose(dww))
    model_accuracy = 1.0 - np.count_nonzero(loss)/len(loss)
    print("accuracy: ", model_accuracy, "const_noise: ", const_noise)
    
    # add datapoint to csv file
    with open('model_noise_accuracy_data.csv', 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['perceptionModule_Nrec50dale_recnoise.1', str(const_noise), str(model_accuracy)])

    model.destruct()

toPlot = []
with open('model_noise_accuracy_data.csv', 'r') as file:
    noiseFile = csv.reader(file)
    
    for line in noiseFile:
        if line[0] != 'model' and line[0] == 'perceptionModule_Nrec50dale_recnoise.1':
            point = [float(x) for x in line[1:3]]       
            toPlot.append(point)

X = np.array([x[0] for x in toPlot])
Y = np.array([x[1] for x in toPlot])

#Y = Y[X < 8]
#X = X[X < 8]
 
#c2,c1,c0 = np.polyfit(X, Y, 2)

plt.scatter(X, Y)
#X_ = np.linspace(np.min(X)-0.5, np.max(X)+0.5, 100)
#plt.plot(X_, c2*X_**2+c1*X_+c0, color='gray')
plt.title('Visual noise vs. model accuracy, bioconstrained perception module (N_rec=50)')
plt.xlabel('Noise parameter (const_noise)')
plt.ylabel('Model accuracy')
