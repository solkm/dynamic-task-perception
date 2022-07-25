#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 15 14:33:42 2022

@author: Sol
"""
import numpy as np
from FourAFC_taskModels import WM_TaskRule_varDelay
from psychrnn.backend.models.basic import Basic
import pandas as pd
#%%

weights_path = './WM_taskrule/vardelay100/Weights/vardelay100.npz'

weights = np.load(weights_path)

modeltask = WM_TaskRule_varDelay(dt=10, tau=100, T=1800, N_batch=100, in_noise=0.7, on_rule=1, off_rule=0, varDelay=True)
network_params = modeltask.get_task_params()
network_params['name'] = 'vardelay100'
network_params['N_rec'] = 100
network_params['rec_noise'] = 0.3
network_params['load_weights_path'] = weights_path
model = Basic(network_params)