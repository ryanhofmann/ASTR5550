#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

def f_base(x, c1=1):

  x0, x1, x2, x3 = np.exp(-3), 0.1, 1, np.exp(2)

  if x < x0:
    return 0.
  elif x < x1:
    return 10.*c1
  elif x < x2:
    return c1/x
  elif x < x3:
    return c1/x**2
  else:
    return 0.


def F(x):

  return .4*x**-1


def get_x0():

  return np.exp(5.*np.random.random() - 3)


if __name__=="__main__":

  np.random.seed(42)
  x = np.arange(np.exp(-3),np.exp(2),0.001)
  f = np.vectorize(f_base)
  norm = simps(f(x), x)
  print(norm)
  print(simps(f(x, c1=1./norm), x))
  y = f(x, c1=1./norm)
  plt.plot(x, y, linewidth=3)
  plt.plot(x, F(x))
  plt.loglog()
  plt.savefig("broken_powerlaw.png")

  N = 100000
  x0 = np.random.random(size=N)*np.exp(2)
  a_f_x0 = np.random.random(size=N)*.4*x0**-1
  ind = np.where(a_f_x0 < f(x0, c1=1./norm))
  randoms = a_f_x0[ind]
  plt.scatter(x0[ind], randoms)
  plt.savefig("broken_powerlaw_scatter.png")

