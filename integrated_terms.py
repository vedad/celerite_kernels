#! /usr/bin/env python

from __future__ import (division, print_function)

import numpy as np
from celerite.terms import SHOTerm, Term

class IntegratedSHOTerm(SHOTerm, Term):

    parameter_names = ("log_S0", "log_Q", "log_omega0", "t_exp")

    def __repr__(self):
        return "IntegratedSHOTerm({0.log_S0}, {0.log_Q}, {0.log_omega0}, {0.t_exp})".format(self)

    def get_real_coefficients(self, params):
        a, c = super(IntegratedSHOTerm, self).get_real_coefficients(params[:-1])

        # new coefficients accounting for attenuation
        a_tilde = a * self.get_real_attenuation(c)
        
        return (a_tilde, c)
        
    def get_complex_coefficients(self, params):
        a, b, c, d = super(IntegratedSHOTerm,
                           self).get_complex_coefficients(params[:-1])

        A, B = self.get_complex_attenuation((a, b, c, d))
        f    = complex(a, b) * complex(A, B)

        # new coefficients accounting for attenuation
        a_tilde, b_tilde = f.real, f.imag

        return (a_tilde, b_tilde, c, d)

    def get_real_attenuation(self, c):
        r"""
        Calculate attentuation due to finite integration time for real term, as
        shown by Dan Foreman-Mackey,

        https://gist.github.com/dfm/54861ee2c05fd147234c4ac1a712d53a
        """
        dt = self.t_exp
        f  = 2.0/(c*dt)**2 * (np.cosh(c*dt) - 1)

        return f

    def get_complex_attenuation(self, coeffs):
        r"""
        Calculate attentuation due to finite integration time for complex term
        as shown by Dan Foreman-Mackey,

        https://gist.github.com/dfm/54861ee2c05fd147234c4ac1a712d53a

        """
        dt         = self.t_exp
        a, b, c, d = coeffs

        C_1 = 2*(a*c**2 - a*d**2 + 2*b*c*d)
        C_2 = 2*(b*c**2 - b*d**2 - 2*a*c*d)

        # real part
        A = (C_1 * (np.cosh(c*dt) * np.cos(d*dt) - 1) -
             C_2 * np.sinh(c*dt) * np.sin(d*dt)) / (dt**2 * (c**2 + d**2)**2)

        # imaginary part
        B = (C_2 * (np.cosh(c*dt) * np.cos(d*dt) - 1) + 
             C_1 * np.sinh(c*dt) * np.sin(d*dt)) / (dt**2 * (c**2 + d**2)**2)

        return (A, B)

