#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 21:34:59 2021

@author: Sol
"""

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

def plot_weights(weights, title="", colorbar=True):
    plt.rcParams.update({'font.size':6})
    mpl.rcParams['xtick.major.width'] = .5
    mpl.rcParams['ytick.major.width'] = .5
    plt.matshow(weights, norm=Normalize(vmin=-.5, vmax=.5), cmap='RdBu')
    if colorbar == True:
        plt.colorbar(shrink=0.5)
    plt.title(title)
    
def plot_activities(activities, trial, dt, title="", label=None, ylabel="Activity of units", colormap=plt.cm.tab20):
    plt.rcParams.update({'font.size':12})
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    plt.plot(range(0,len(activities[trial,:,:])*dt,dt), activities[trial,:,:])
    plt.xlabel("Time (ms)")
    plt.ylabel(ylabel)
    plt.title(title)
    if colormap is not None:
        colors = [colormap(i) for i in np.linspace(0, 1,len(ax1.lines))]
        for i,j in enumerate(ax1.lines):
            j.set_color(colors[i])
    if label is not None:
        plt.legend(label)
    
def proj_ortho_basis(basis_mat, vec):
    v = vec
    dim = basis_mat.shape[0]
    x = np.zeros(basis_mat.shape[1])
    for i in range(dim):
        b = basis_mat[i]
        p = np.dot(v, b) / np.dot(b, b) * b
        x += p
    
    return x

def add_value_labels(ax, spacing=10, dec=2, fontsize=10):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number
        if dec >=1:
            label = np.around(y_value, dec)
        else:
            label = int(np.rint(y_value))
              
        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            fontsize=fontsize,
            va=va)                      # Vertically align label differently for positive and negative values.