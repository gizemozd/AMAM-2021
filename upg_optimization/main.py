import pandas as pd
import numpy as np

from upg import upg_optimization
from utils import save_pickle
import logging

import argparse

logging.basicConfig(
    level=logging.INFO,
    format=' %(asctime)s - %(levelname)s- %(message)s'
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--leg', default='RF')
    args = parser.parse_args()

    grooming_angles = pd.read_pickle('../data/smoothed_joint_angles.pkl')
    time_intervals = pd.read_pickle('../data/time_intervals.pkl')

    opt_angles = {}
    opt_params = {}

    opt_angles[args.leg + '_leg'] = {
        joint: list()
        for joint in grooming_angles[args.leg + '_leg'].keys()
        }

    opt_params[args.leg + '_leg'] = {
        joint: list()
        for joint in grooming_angles[args.leg + '_leg'].keys()
        }

    for leg in grooming_angles.keys():
        for joint in grooming_angles[leg].keys():
            if args.leg in leg:
                logging.info('Optimizing for {} {}'.format(leg,joint))
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

                opt_angles[leg][joint], opt_params[leg][joint] = upg_optimization(
                    grooming_angles,
                    leg,joint,
                    time_interval,
                    best_initial,
                    best_parameters
                    )

    save_pickle(opt_angles, '../data/fitted_angles_{}.pkl'.format(args.leg))
    save_pickle(opt_params, '../data/opt_parameters_{}.pkl'.format(args.leg))