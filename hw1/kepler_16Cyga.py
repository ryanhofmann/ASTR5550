#!/usr/bin/env python3

import hw1
import kepler_llsq as kllsq
import kepler_fft as kfft
import numpy as np
import matplotlib.pyplot as plt

# Read in data
fin = "kplr100002741-2010296114515_slc.fits"
t, f, fk = hw1.read_data(fin)

# Detrend and FFT
f3 = kllsq.detrend(t, f, N=3)
t_lin, f3_lin = kfft.resample(t, f3, nt=65536)
w, P = kfft.fft(t_lin, f3_lin)

# Plot full power spectrum
plt.plot(w[1:32767], P[1:32767])
plt.xlabel("Frequency [1/days]")
plt.ylabel("Power [(e-/s)^2]")
plt.loglog()
plt.savefig("16Cyga_fft_log.eps")
plt.savefig("16Cyga_fft_log.png")
plt.clf()

# Plot zoomed-in power spectrum
ws = w/86400.
plt.plot(ws[1:32767], P[1:32767])
plt.xlabel("Frequency [(1/seconds)]")
plt.ylabel("Power [(e-/s)^2]")
plt.xlim(1.5e-3, 3.0e-3)
plt.ylim(0, 1.2e14)
plt.savefig("16Cyga_fft_lin.eps")
plt.savefig("16Cyga_fft_lin.png")
plt.clf()

