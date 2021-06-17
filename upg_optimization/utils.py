import os 
import pickle 
import glob

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import logging

logging.basicConfig(
    level=logging.INFO,
    format=' %(asctime)s - %(levelname)s- %(message)s'
    )


def plot_style():
    plt.rcParams['figure.figsize'] = [8, 4]
    plt.rcParams['lines.linewidth'] = 1.7
    plt.rcParams['font.size'] = 14
    plt.rcParams['legend.fontsize'] = 10

    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.left'] = True
    plt.rcParams['axes.spines.bottom'] = True

    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['legend.frameon'] = False

    plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.ymargin'] = 0

    colors = ['#661100', '#117733', '#6699CC', '#CC6677', '#DDCC77']
    return colors

def load_joint_angles():
    """ Loads the grooming joint angles. """
    path = glob.glob('./data/joint_angles*.pkl')[0]
    return pd.read_pickle(path)

def save_pickle(variable, file_name):
    """ Saves file in a pkl format. """
    file_name += '.pkl' if not file_name.endswith('.pkl') else ''
    with open(file_name,'wb') as f:
        pickle.dump(variable,f)
        logging.info('Loaded successfully!')

def plot_data(t,x,xlabel,y,ylabel,title, ax=None):
    colors = plot_style()
    ax = plt.gca() if ax is None else ax
    ax.plot(t, np.array(x) * 180/np.pi, 'r-', c= colors[0], linewidth = 2, label = xlabel)
    ax.plot(t, np.array(y) * 180/np.pi, 'b', c=colors[1], linewidth =1, label = ylabel)
    ax.set_title(title)
    ax.set_ylabel('Joint Angles (rad)')
    #ax.yaxis.tick_left()
    ax.set_xlabel('Time (s)')
    ax.grid(True)
    ax.legend(loc='best')


def plot_params(t, omega, mu, target, title, ax= None):
    colors = plot_style()

    ax = plt.gca() if ax is None else ax
    ax[0].plot(t, np.array(omega), c= colors[2], label = 'omega')
    ax[1].plot(t, np.array(mu), c=colors[3], label = 'amplitude')
    ax[2].plot(t, np.array(target), c=colors[4], label = 'offset')

    plt.suptitle(title)
    ax[0].set_ylabel('Omega')
    ax[1].set_ylabel('Amplitude')
    ax[2].set_ylabel('Target')

    ax[2].set_xlabel('Time (s)')


def apply_filter(signal = None, window_length=51, degree=2):
    """ Applies filter to joint angles. """
    return savgol_filter(signal, window_length, degree)


def apply_filter_to_dict(joint_angles, window_length=51, degree=2, cut_ind = 3000):
    """ Filters the dictionary of joint angles. """
    smoothed_signal = {
        leg: {joint: list()}
        for leg in joint_angles.keys()
        for joint in joint_angles[leg].keys()
        }

    for leg in joint_angles.keys():
        for joint in joint_angles[leg].keys():
            smoothed_signal[leg][joint] = apply_filter(
                np.array(joint_angles[leg][joint]),
                window_length, degree
                )[:cut_ind]

    return smoothed_signal

