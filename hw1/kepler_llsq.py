#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import hw1

def llsq(t, b, N=3):
  """
  Fit a polynomial of degree N to the light curve b(t).
  b, t are 1D arrays of identical size.
  Outputs array of coefficients, lowest order first.
  """

  # Create arrays A and A_T, A_T*A*x = A_T*b
  A = np.zeros((len(b), N+1))
  for i in range(N+1):
    np.copyto(A[:,i], t**i)
  A_T = np.transpose(A)

  # Multiply matrices
  ATA = np.dot(A_T, A)
  ATb = np.dot(A_T, b)

  # Solve for coefficients
  x = np.linalg.solve(ATA, ATb)

  return x


def detrend(t, b, N=3):
  """
  Remove polynomial of degree N from data b(t).
  """

  # Calculate coefficients
  x = llsq(t, b, N=N)

  # Generate trendline A*x = fit
  A = np.zeros((len(t), N+1))
  for i in range(N+1):
    np.copyto(A[:,i], t**i)
  fit = np.dot(A, x)

  # Detrend lightcurve
  b_fit = b - fit
  return b_fit


if __name__=="__main__":

  # Read data from file
  f = "kplr007200111-2009350155506_llc.fits"
  t, b, bk = hw1.read_data(f)

  # Plot raw data
  plt.plot(t, b)
  x_label = "Time [BJD-2454833]"
  y_label = "Flux [e-/sec]"
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.savefig("raw.eps")
  plt.savefig("raw.png")
  plt.clf()

  # Generate trendlines
  Nmax = 4
  A = np.zeros((len(t), Nmax+1))
  x = np.zeros(Nmax+1)
  Ax = np.zeros((len(t), Nmax))
  for i in range(Nmax+1):
    np.copyto(A[:,i], t**i)
  for i in range(Nmax):
    x[:i+2] = llsq(t, b, N=i+1)
    np.copyto(Ax[:,i], np.dot(A[:,:i+2], x[:i+2]))

  # Plot data with trendlines
  plt.plot(t, b, label="data")
  for i in range(Nmax):
    plt.plot(t, Ax[:,i], label="N = {:d}".format(i+1))
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.legend(loc="lower left")
  plt.savefig("trends.eps")
  plt.savefig("trends.png")
  plt.clf()

  # Detrend data with N=3 polynomial, plot with Kepler detrended data
  b_3 = detrend(t, b, N=3)
  plt.plot(t, b_3, label="N = 3")
  plt.plot(t, bk - np.mean(bk), label="PDC")
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.legend(loc="upper left")
  plt.savefig("detrend.eps")
  plt.savefig("detrend.png")
  plt.clf()

