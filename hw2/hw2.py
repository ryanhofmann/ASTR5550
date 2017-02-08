#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def image(p=0.3):
  """
  Uses np.random.random() to simulate a random event.
  If random number is less than p, the event happens.
  Returns True or False.
  """

  if np.random.random() < p:
    return True
  else:
    return False


def observation(p=0.3):
  """
  Runs image() repeatedly until it returns True.
  Returns number of iterations.
  """

  i = 0
  while True:
    i += 1
    if image(p):
      break
  return i


def trials(p=0.3, N=1000):
  """
  Runs N trials of observation().
  Returns array containing number of images for each trial.
  """

  k = np.zeros(N)
  for i in range(0, N):
    k[i] = observation(p)

  return k


def analytic(k, p=0.3):
  """
  Analytic solution for P[k]. Returns float.
  """

  return p*(1-p)**(k-1)


def plot(p=0.3, N=1000):
  """
  Plots histogram of frequency vs. number of images k.
  """

  k = trials(p, N)
  x = np.arange(1, np.max(k)+1)
  y = analytic(x, p)*N

  plt.plot(x, y, linewidth=2, color='red')
  plt.hist(k, bins=np.max(k)-1, align='left', color='blue')
  plt.xlabel("k")
  plt.ylabel("N*P[k]")
  plt.savefig("p_{:.1f}_N_{:d}.eps".format(p, N))
  plt.savefig("p_{:.1f}_N_{:d}.png".format(p, N))
  plt.clf()


if __name__=="__main__":

  # Explicitly initialize pseudorandom number generator
  np.random.seed(42)

  # Create plots for cluster hunt
  plot(0.3, 1000)
  plot(0.3, 10)
  plot(0.3, 100)
  plot(0.3, 10000)
  plot(0.1, 10000)

  # Plot PDF for third problem
  x = np.arange(0,10,.1)
  y = x*np.exp(-x)
  plt.plot(x, y, linewidth=2)
  plt.xlabel("x")
  plt.ylabel("f(x)")
  plt.savefig("PDF.eps")
  plt.savefig("PDF.png")
  plt.clf()

