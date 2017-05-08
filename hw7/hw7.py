#!/usr/bin/env python3

import numpy as np
import scipy.stats as stats

data = np.genfromtxt("hw7_q1.dat")
x = data[:,0]
y = data[:,1]
z = x - y
varx = 2.5
vary = 2.5
varz = varx + vary
N = len(z)

print(np.mean(x), np.mean(y))

chi2N = np.sum(z**2)/varz
PTE = 1. - stats.chi2.cdf(chi2N, N)
DOF = N
print("chi2N = {:.2f}, DOF = {:d}, PTE = {:.3f}".format(chi2N, DOF, PTE))

t = np.mean(z)/np.sqrt(varz/N)
PTE = 1. - stats.t.cdf(t, N)
DOF = N - 1
print("t = {:.2f}, DOF = {:d}, PTE = {:.3f}".format(t, DOF, PTE))

