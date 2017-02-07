#!/usr/bin/env python3

import hw1
import kepler_llsq as kllsq
import numpy as np
import matplotlib.pyplot as plt

def resample(t, f, nt=8192):
  """
  Resamples f(t) to a uniform grid of length nt.
  """

  # Create time array with uniform spacing
  t_lin = np.linspace(t[0], t[-1], num=nt)

  # Perform interpolation
  f_lin = np.interp(t_lin, t, f)

  return t_lin, f_lin


def fft(t, f):
  """
  Compute the FFT of f(t). Returns power spectrum as two arrays.
  """

  a = np.fft.fft(f)  # amplitude
  P = np.abs(a)**2  # power spectrum
  w = np.fft.fftfreq(t.size, t[1]-t[0])  # frequencies

  return w, P


if __name__=="__main__":

  # Read in data
  fin = "kplr007200111-2009350155506_llc.fits"
  t, f, fk = hw1.read_data(fin)

  # Detrend and FFT
  f3 = kllsq.detrend(t, f, N=3)
  t_lin, f3_lin = resample(t, f3, nt=8192)
  w, P = fft(t_lin, f3_lin)

  # Plot power spectrum
  plt.plot(w[1:4095], P[1:4095])
  plt.xlabel("Frequency [1/days]")
  plt.ylabel("Power [(e-/s)^2]")
  plt.loglog()
  plt.savefig("fft.eps")
  plt.savefig("fft.png")
  plt.show()
  plt.clf()

