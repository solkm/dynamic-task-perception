#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 01:09:50 2021

@author: Sol
"""

import numpy as np
import matplotlib.pyplot as plt
import visfunctions as vf

weights = np.load('./WM_taskrule/WM_ContextSignal/ctxtsig_vardelay.npz')
#%%
vf.plot_weights(weights['W_in'],title='Input weights')
#plt.savefig('./WM_taskrule/model3b_1000_Win.png',dpi=300, bbox_inches='tight')

vf.plot_weights(weights['W_rec'],title='Recurrent weights')
#plt.savefig('./WM_taskrule/model3b_1000_Wrec.png',dpi=300, bbox_inches='tight')
#%%