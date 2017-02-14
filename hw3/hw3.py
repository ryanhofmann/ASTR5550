#!/usr/bin/env python3

import numpy as np
from scipy import integrate
from scipy import interpolate
import matplotlib.pyplot as plt

def f(x, p=2):
  """
  Distribution function
  """

  return x**(-p)


def x(a, p=2):
  """
  Inverse transform function
  """

  return ((-p + 1)*a + 1)**(1/(-p + 1))


def P(k, lam=10):
  """
  Poisson distribution function
  """

  return lam**k*np.exp(-lam)/np.math.gamma(k+1)


def rand_P(lam=10):
  """
  Poisson RNG function
  Returns x(alpha)
  """

  x = np.linspace(0,20,1000)
  y = [P(i) for i in x]
  alpha = integrate.cumtrapz(y, x, initial=0)
  x_int = interpolate.interp1d(alpha, x, fill_value='extrapolate')

  return x_int


if __name__=="__main__":

  # Explicitly initialize RNG
  np.random.seed(42)

  # Problem 2
  # Numerically integrate f(x) to obtain alpha
  x_sup = np.linspace(1,100,num=1000)
  alpha = integrate.cumtrapz(f(x_sup), x_sup, initial=0)

  # Interpolate alpha
  x_int = interpolate.interp1d(alpha, x_sup, fill_value='extrapolate')

  # Draw 10^5 random numbers in [0,1)
  N = 100000
  a = np.random.rand(N)

  # Transform the uniform random numbers to the PDF
  x_an = x(a)
  a = np.random.rand(N)
  x_num = x_int(a)

  # Create analytic histograms
  bins = np.arange(100) + 1.
  y = N*bins**-2
  figsize = (8,4)
  plt.figure(figsize=figsize)
  plt.subplot(121)
  plt.hist(x_an, bins=bins, color='blue')
  plt.plot(bins, N*f(bins), '-r', linewidth=2)
  plt.subplot(122)
  plt.hist(x_an, bins=bins, color='blue')
  plt.plot(bins, N*f(bins), '-r', linewidth=2)
  plt.loglog()
  plt.tight_layout()
  plt.savefig("analytic.eps")
  plt.savefig("analytic.png")
  plt.clf()

  # Create numerical histograms
  plt.figure(figsize=figsize)
  plt.subplot(121)
  plt.hist(x_num, bins=bins, color='blue')
  plt.plot(bins, N*f(bins), '-r', linewidth=2)
  plt.subplot(122)
  plt.hist(x_num, bins=bins, color='blue')
  plt.plot(bins, N*f(bins), '-r', linewidth=2)
  plt.loglog()
  plt.tight_layout()
  plt.savefig("numerical.eps")
  plt.savefig("numerical.png")
  plt.clf()


  # Problem 3
  # Draw 10^5 random numbers from Poisson distribution
  x_int = rand_P()  # Creates lookup table for x(alpha)
  a = np.random.rand(N)
  k = x_int(a)  # Transforms uniform randoms to distribution

  # Create histogram 1
  bins = np.arange(100)/5.
  y = np.array([N*P(i) for i in bins])
  plt.figure(figsize=figsize)
  plt.subplot(121)
  plt.hist(k, bins=bins, color='blue')
  plt.plot(bins, y/5., '-r', linewidth=2)
  ax = plt.subplot(122)
  ax.hist(k, bins=bins, color='blue')
  ax.plot(bins, y/5., '-r', linewidth=2)
  ax.set_yscale("log")
  plt.tight_layout()
  plt.savefig("P1.eps")
  plt.savefig("P1.png")
  plt.clf()

  # Create histogram 2
  k_np = np.random.poisson(lam=10, size=N)
  plt.figure(figsize=figsize)
  plt.subplot(121)
  plt.hist(k_np, bins=bins, color='blue')
  plt.plot(bins, y, '-r', linewidth=2)
  ax = plt.subplot(122)
  ax.hist(k_np, bins=bins, color='blue')
  ax.plot(bins, y, '-r', linewidth=2)
  ax.set_yscale("log")
  plt.tight_layout()
  plt.savefig("P2.eps")
  plt.savefig("P2.png")
  plt.clf()

  # Create histogram 3
  a = np.random.rand(N)
  k = np.rint(x_int(a))
  y = np.array([N*P(i) for i in bins])
  plt.figure(figsize=figsize)
  plt.subplot(121)
  plt.hist(k, bins=bins, color='blue')
  plt.plot(bins, y, '-r', linewidth=2)
  ax = plt.subplot(122)
  ax.hist(k, bins=bins, color='blue')
  ax.plot(bins, y, '-r', linewidth=2)
  ax.set_yscale("log")
  plt.tight_layout()
  plt.savefig("P3.eps")
  plt.savefig("P3.png")
  plt.clf()

