#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 11:48:32 2021

@author: Sol
"""
import numpy as np
from model_functions import probTaskSwitch_nAfterSwitch, probNoSwitchYet_nAfterSwitch
import matplotlib.pyplot as plt
import scipy.io
import model_functions as mf
import visfunctions as vf

tParams = scipy.io.loadmat(r'tParams.mat')

tParams_switchProbs = probTaskSwitch_nAfterSwitch(tParams['r_CHOICE'][0], tParams['r_TASK'][0], 8)

fig, ax = plt.subplots()
ax.bar(np.arange(len(tParams_switchProbs)), tParams_switchProbs, align='center', alpha=1)
ax.set_xticks(np.arange(len(tParams_switchProbs)))
ax.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8'])
ax.set_title('Monkey switch probabilities following a task switch')
ax.set_xlabel('Number of trials after task switch')
ax.set_ylabel('P(switch)')
vf.add_value_labels(ax)

# Correct switches
choice = tParams['r_CHOICE'][0]
task = tParams['r_TASK'][0]
beh_switch, _ = mf.task_switches(mf.get_tasks(choice))
switch, _ = mf.task_switches(task)
beh_switchind = np.flatnonzero(beh_switch)
switchind = np.flatnonzero(switch)

# Conditioned on no switch yet

prob_noSwitchYet = probNoSwitchYet_nAfterSwitch(tParams['r_CHOICE'][0], tParams['r_TASK'][0], 8)
switchProbsGivenNoSwitchYet = np.divide(tParams_switchProbs, prob_noSwitchYet)

print(switchProbsGivenNoSwitchYet)

fig, ax = plt.subplots()
ax.bar(np.arange(len(switchProbsGivenNoSwitchYet)), switchProbsGivenNoSwitchYet, align='center', alpha=1)
ax.set_xticks(np.arange(len(switchProbsGivenNoSwitchYet)))
ax.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8'])
ax.set_title('Conditional monkey switch probabilities')
ax.set_ylabel('P(switch | has not switched already)')
ax.set_xlabel('Number of trials after task switch')
vf.add_value_labels(ax)

'''
switch_nodelay = 0
for i in switchind:
    for j in beh_switchind:
        if i==j:
            switch_nodelay += 1
print(switch_nodelay) #493

switch_1before = 0
for i in switchind:
    for j in beh_switchind:
        if i==j+1:
            switch_1before += 1
print(switch_1before) #367

switch_2before = 0
for i in switchind:
    for j in beh_switchind:
        if i==j+2:
            switch_2before += 1
print(switch_2before) #333

switch_3before = 0
for i in switchind:
    for j in beh_switchind:
        if i==j+3:
            switch_3before += 1
print(switch_3before) #385
'''