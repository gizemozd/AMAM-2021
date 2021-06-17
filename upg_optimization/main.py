import pandas as pd 
import numpy as np

from upg import optimization
from utils import save_pickle

if __name__ == '__main__':

    grooming_angles = pd.read_pickle('../data/smoothed_joint_angles.pkl')
    time_intervals = pd.read_pickle('../data/time_intervals.pkl')

    opt_angles = {
        leg: {joint: list()}
        for leg in grooming_angles.keys()
        for joint in grooming_angles[leg].keys()
        }

    opt_params = {
        leg: {joint: list()}
        for leg in grooming_angles.keys()
        for joint in grooming_angles[leg].keys()
        }

    for leg in grooming_angles.keys():
        for joint in grooming_angles[leg].keys():
            time_interval = time_intervals['_'.join((leg, joint))]
            angles = grooming_angles[leg][joint]
            
            best_initial = np.array(
                [0, 0.05, 0, 0.005, 
                grooming_angles[leg][joint][0],
                grooming_angles[leg][joint][0]
                ]
            ) #1.8, 1.5])

            best_parameters  = (
                0.15,53,
                grooming_angles[leg][joint][0],
                80)

            opt_angles[leg][joint], opt_params[leg][joint] = optimization(
                grooming_angles,
                leg,joint,
                time_interval,
                best_initial,
                best_parameters
                )

    save_pickle(opt_angles, '../data/fitted_angles.pkl')
    save_pickle(opt_params, '../data/opt_parameters.pkl')




    
