#!/usr/bin/env python

from __future__ import (print_function, division)

import numpy as np
import matplotlib.pyplot as plt
import celerite
from integrated_terms import IntegratedSHOTerm

def create_data():
    """ example data from celerite documentation """
    np.random.seed(42)

    # time in units of minutes
    dead_time        = 1/60
    integration_time = 2/60
    dt               = integration_time + dead_time

    # units of hours
    t = np.append(np.arange(0, 3.8, dt), np.arange(5.5, 10, dt))

    yerr = np.random.uniform(0.08, 0.22, len(t))
    y    = np.sin(3*t + 0.1*(t-5)**2) + yerr * np.random.randn(len(t))

    true_t = np.linspace(0, 10, 5000)
    true_y = np.sin(3*true_t + 0.1*(true_t-5)**2)

    return (t, y, yerr), (true_t, true_y)

def print_stuff(gp, y):
    print("Initial log likelihood: {0}".format(gp.log_likelihood(y)))
    print("parameter_dict:\n{0}\n".format(gp.get_parameter_dict()))
    print("parameter_names:\n{0}\n".format(gp.get_parameter_names()))
    print("parameter_vector:\n{0}\n".format(gp.get_parameter_vector()))
    print("parameter_bounds:\n{0}\n".format(gp.get_parameter_bounds()))
    print('complex coefficients inside SHOTerm: ', 
            gp.kernel.get_complex_coefficients((log_S0, log_Q, log_omega0)))

if __name__ == "__main__":
    # data from asteroseismic example in the original celerite paper
    data, true = create_data()
    (t, y, yerr), (true_t, true_y) = data, true

    # optimal parameters from least squares minimisation
    log_S0     = -4.16
    log_Q      = 2.34
    log_omega0 = 1.13

    # original SHO kernel
    k_sho = celerite.terms.SHOTerm(log_S0=log_S0, log_Q=log_Q, log_omega0=log_omega0)
    gp_sho = celerite.GP(k_sho, mean=np.mean(y), fit_mean=False)
    gp_sho.compute(t, yerr)
    print_stuff(gp_sho, y)

    # integrated SHO kernel
    t_exp = 2/60 # 2 minute exposures
    k_isho = IntegratedSHOTerm(log_S0=log_S0, log_omega0=log_omega0,
                               log_Q=log_Q, t_exp=t_exp)
    gp_isho = celerite.GP(k_isho, mean=np.mean(y), fit_mean=False)
    gp_isho.compute(t, yerr)
    print_stuff(gp_isho, y)

    # make predictions
    mu_sho, var_sho = gp_sho.predict(y, true_t, return_var=True)
    mu_isho, var_isho = gp_isho.predict(y, true_t, return_var=True)

    # data and true
    plt.plot(true_t, true_y, "k", lw=1.5, alpha=0.3)
    plt.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(0, 10)
    plt.ylim(-2.5, 2.5)

    # models
    colors = "#ff7f0e", "#1f77b4"
    plt.plot(true_t, mu_sho, color=colors[0])
    plt.plot(true_t, mu_isho, color=colors[1])
    plt.show()

