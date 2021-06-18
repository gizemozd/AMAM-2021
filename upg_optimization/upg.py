import time

import numpy as np
from scipy.integrate import odeint
from lmfit import minimize, Parameters, report_fit
import matplotlib.pyplot as plt

import utils
import logging

logging.basicConfig(
    level=logging.INFO,
    format=' %(asctime)s - %(levelname)s- %(message)s'
)
#: Global variables
legs = ['LF_leg', 'RF_leg']
joints = [
    'ThC_yaw',
    'ThC_pitch',
    'ThC_roll',
    'CTr_pitch',
    'FTi_pitch',
    'CTr_roll',
    'TiTa_pitch']

#: Log optimization results
grooming_angles_optimization = {
    leg: {joint: list() for joint in joints} for leg in legs}
grooming_params_optimization = {
    leg: {joint: list() for joint in joints} for leg in legs}

def upg_optimization(
    grooming_angles,
    name, key,
    time_intvs,
    best_initial,
    best_parameters,
    show_fig=True
):
    """Optimization using AMPGO to fit the joint angles.

    Args:
        grooming_angles (array_like): angles to be fitted in the optimization.
        name (string): name of the leg, i.e. 'LF_leg' or 'RF_leg'
        key (string): name of the joint, i.e. 'TiTa_pitch' or 'ThC_yaw'
        time_intvs (array_like): time intervals indicating the starting and ending points of the optimization
            Example: np.array((0,100),(100,340), ...)
        best_initial (array_like): initial condition for the first iteration
        best_parameters (array_like): parameters (mu, amplitude, target, A) for the first iteration
        show_fig (bool): shows figures
    """

    #: Dynamical system
    def f(xs, t, ps):
        """ Unit Pattern Generator."""
        try:
            mu = ps['mu'].value
            omega = ps['omega'].value
            target = ps['target'].value
            A = ps['A'].value
        except BaseException:
            mu, omega, target, A = ps
        # Constant Variables
        B = 10
        C = 10

        # States: h, y, v, m, x, z
        #: Discrete
        h_d = 1 - xs[0]
        y_d = xs[2]
        v_d = xs[0] * B * (xs[0] * B * 0.25 * (target - xs[1]) - xs[2])
        # Radius: sqrt((x-y)^2 + z^2)
        r = np.sqrt((xs[4] - xs[1])**2 + xs[5]**2)
        #: Ryhtmic
        m_d = C * (mu - xs[3])
        x_d = A * 1.0 / np.abs(mu) * \
            (xs[3] - r**2) * (xs[4] - xs[1]) - omega * xs[5]
        z_d = A * 1.0 / np.abs(mu) * \
            (xs[3] - r**2) * xs[5] + omega * (xs[4] - xs[1])
        # Output of the UPG: x_d
        return [h_d, y_d, v_d, m_d, x_d, z_d]


    def g(t, x0, ps):
        """ Solution to the ODEs."""
        x = odeint(f, x0, t, args=(ps,))
        return x

    def residual(ps, ts, data):
        """ Objective function."""
        x0 = ps['h0'].value, ps['y0'].value, ps['v0'].value, ps['m0'].value, ps['x0'].value, ps['z0'].value
        model = g(duration, x0, ps)
        global cost
        cost = (np.square(model[:, 4] - data)).ravel()
        return cost

    def log_cost(ps, iter, residual, *args, **kws):
        """ Prints the loss values in every 100 iteration. """
        if iter % 5 == 0:
            logging.info("Iteration: {} and cost: {}".format(iter, sum(cost)))
        cost_values.append(sum(cost))
        parameters.append(
            [ps['mu'].value, ps['omega'].value, ps['target'].value])

    param_result = np.empty_like((1, 1, 1))
    optimization_result = np.empty_like(np.zeros((1,6)))

    dt = 0.001

    starting_time = time.time()

    for begin, end in time_intvs:

        time_ = end - begin
        yData = grooming_angles[name][key][begin:end]
        duration = np.arange(0, time_ * dt, dt)

        parameters = list()
        cost_values = list()

        data = g(duration, best_initial, best_parameters)

        # Set parameters incluing bounds
        params = Parameters()
        params.add('h0', value=float(data[0, 0]),
                   min=0, max=1, vary=True)  # False
        params.add('y0', value=float(data[0, 1]), min=-3, max=2, vary=True)
        params.add('v0', value=float(data[0, 2]),
                   min=-3, max=2, vary=True)  # vary=False
        params.add('m0', value=float(data[0, 3]), min=0, max=2, vary=True)
        params.add('x0', value=float(data[-1, 4]), min=-3, max=2, vary=True)
        params.add('z0', value=float(data[-1, 5]), min=-3, max=2, vary=True)
        params.add('mu', value=0.1, min=-0.1, max=0.5)
        params.add('omega', value=53, min=40, max=90)
        params.add('target', value=1.5, min=0, max=3)
        params.add('A', value=80, min=1, max=90, vary=True)
        params.pretty_print()

        # fit model and find predicted values
        result = minimize(residual, params, args=(duration, yData),
                          iter_cb=log_cost, method='AMPGO', nan_policy='omit')

        best_initial = (result.params['h0'].value,
                        result.params['y0'].value,
                        result.params['v0'].value,
                        result.params['m0'].value,
                        result.params['x0'].value,
                        result.params['z0'].value
                        )

        final = g(duration, best_initial, result.params)

        # Display fitting statistics
        report_fit(result)

        temp_param = np.ones((time_, 1)) * (
            result.params['mu'].value,
            result.params['omega'].value,
            result.params['target'].value
        )

        param_result = np.vstack((param_result, temp_param))
        optimization_result= np.append(optimization_result, final, axis=0)

        logging.info(
            "Param length: {}, Result length: {}".format(
                len(param_result),
                len(optimization_result)))

    grooming_angles_optimization[name][key] = optimization_result
    grooming_params_optimization[name][key] = param_result[1:]

    # utils.save_pickle(grooming_angles_optimization, '../data/opt_{}_{}'.format(key,name))
    # utils.save_pickle(grooming_params_optimization, '../data/param_{}_{}'.format(key,name))

    logging.info("Optimization took: {}\n".format((time.time() - starting_time)/3600))

    if show_fig:

        colors = utils.plot_style()

        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(12, 8))
        ax1.plot(grooming_angles[name][key], label='processed', c=colors[0])
        ax1.plot(
            utils.apply_filter(
                grooming_angles_optimization[name][key][:,4]),
            '--',
            label='optimization', c=colors[1])
        ax1.set_title('Regression versus Real Data ({} {})'.format(name, key))
        ax1.set_xlabel('Joint Angles (rad)')
        ax1.legend()

        ax2.plot(grooming_params_optimization[name][key][:, 0], label='Amplitude', c=colors[2])
        ax2.plot(grooming_params_optimization[name][key][:, 1], label='Omega', c=colors[3])
        ax2.plot(grooming_params_optimization[name][key][:, 2], label='Target', c=colors[4])
        ax2.set_title('Control Parameters')
        ax2.set_xlabel('Time (msec)')
        ax2.set_ylabel('Control Signals')
        ax2.legend()
        try:
            plt.savefig('../docs/paramresults_{}_{}.eps'.format(key,name))
        except:
            logging.ingo('File does not exist... Could not save')

        fig, ax1 = plt.subplots(figsize=(11, 6))
        ax1.plot(cost_values, label='loss')
        ax1.set_title('Loss over iterations')
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Loss values')
        ax1.legend()
        try:
            plt.savefig('../docs/loss_{}_{}.eps'.format(key,name))
        except:
            logging.info('File does not exist... Could not save')

        return grooming_angles_optimization, grooming_params_optimization