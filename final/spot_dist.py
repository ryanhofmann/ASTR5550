#!/usr/bin/env python3

# Ryan Hofmann
# ASTR 5550 final project
# 
# WARNING: pymacula does not work with python 3 by default
# To fix, modify macula.py, replacing xrange() with range()

"""
Generate 2D plots of starspot model properties.
X-axis: number of spots visible in model, range 1-10.
        spots are of equal radius alpha_max.
Y-axis: inclination of star, range 0-90 degrees.
Plot a dot for every lightcurve with given amplitude.

Mean and sigma of 1/f
1/f vs fractional spot coverage
    
1/f vs amplitude
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib
import pymacula
import pickle
from tqdm import *

# Initialize RNG
np.random.seed(547865)

# Set simulation parameters
N = 100000
inc = np.random.rand(N)*np.pi/2
nspots_max = 20
nspots = np.random.randint(1,nspots_max+1,N)
fname = "N_{:d}_nspots_{:.0f}.pkl".format(N, nspots_max)
alpha_max = 10
spot_area = 0.5*(1-np.cos(alpha_max*np.pi/180))  # fraction of star covered by spot
data = np.zeros((N, 5))
amps = np.linspace(0,10,11)/100  # amplitude bins

# Generate simulation data
for i in tqdm(range(N)):
  try:
    # Create starspot model
    lat = np.arccos(2*(np.random.rand(nspots[i])-0.5))
    spots = [pymacula.Spot(lat=lat[j], alpha_max=alpha_max) for j in range(nspots[i])]
#    spots = [pymacula.Spot(alpha_max=alpha_max) for j in range(nspots[i])]
    model = pymacula.MaculaModel(spots=spots, nspots=nspots[i])
#    model = pymacula.MaculaModel(nspots=nspots[i])
    model.star.incl = inc[i]
    for spot in model.spots:
      if spot.lat > np.pi/2 and spot.lat - np.pi < (-inc[i] - alpha_max*np.pi/180):
        nspots[i] -= 1
    ts = np.arange(0,500,0.05)

    # Compute lightcurve amplitude
    amp = 0.5*(np.max(model(ts)) - np.min(model(ts)))

    # Compute TDV mean and sigma (units of Rp/R*)
    TDV = model(ts)**-1
    TDV_mean = np.mean(TDV)
    TDV_sigma = np.std(TDV)

    # Record data in array
    data[i] = [nspots[i], inc[i], amp, TDV_mean, TDV_sigma]

    # If last iteration, pickle data
    if i == N-1:
      pickle.dump(data, open(fname, "wb"))

  except KeyboardInterrupt:
    break

#for i in range(N):
#  print("{:.0f}\t{:.0f}\t{:.1f} %".format(data[i,0], data[i,1]*180/np.pi, data[i,2]*100))

print("Amplitude range: {:.1f} - {:.1f} %".format(np.min(data[:,2])*100, np.max(data[:,2])*100))

# Plot 2D distributions with marginalized distributions at top and left
data = pickle.load(open(fname, "rb"))
matplotlib.rcParams.update({'font.size': 22})
for i in range(len(amps)-1):
  points = data[np.where(np.abs(data[:,2] - (amps[i+1]+amps[i])/2) < (amps[i+1]-amps[i])/2)]
#  print(len(points))
  fig = plt.figure(figsize=(10,10))
  ax1 = plt.subplot2grid((8,8), (0,2), rowspan=2, colspan=6)
  plt.xlabel("number of spots")
  ax2 = plt.subplot2grid((8,8), (2,0), rowspan=6, colspan=2)
  ax3 = plt.subplot2grid((8,8), (2,2), rowspan=6, colspan=6, sharex=ax1, sharey=ax2)
  plt.ylabel("inclination")
  xbins = np.arange(0, nspots_max+1) + 0.5
  ybins = np.linspace(0, 90, nspots_max+1)
  f1 = ax1.hist(points[:,0], bins=xbins, color='grey')
  ax1.locator_params(axis='y', nbins=4)
  f2 = ax2.hist(points[:,1]*180/np.pi, bins=ybins, orientation='horizontal', color='grey')
  ax2.invert_xaxis()
  ax2.locator_params(axis='x', nbins=4)
  plt.setp(ax2.get_xticklabels(), rotation=45, horizontalalignment='right')
  f3 = ax3.hist2d(points[:,0], points[:,1]*180/np.pi, bins=[xbins,ybins], cmap='Greys', norm=colors.LogNorm(), vmax=300)
#  print("hist2d peak: {:.0f}".format(np.max(f3[0])))
  fig.suptitle("{:.2f} < amp < {:.2f}".format(amps[i], amps[i+1]))
  plt.tight_layout()
  fig.subplots_adjust(top=0.94)
  plt.savefig("N_{:d}_amp_{:.1f}_{:.1f}.png".format(N, amps[i]*100, amps[i+1]*100))
  plt.close()

# Plot mean TDVs vs mean spot coverage
area = data[:,0]*(1 - 0.5*np.sin(data[:,1]))*spot_area
counts, xbins, ybins, image = plt.hist2d(area, data[:,3], bins=nspots_max, cmap='Greys', norm=colors.LogNorm())
plt.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],linewidths=3)
plt.xlabel("mean spot coverage")
plt.ylabel("mean TDV")
plt.title("N = {:d}, alpha_max = {:d}".format(N, alpha_max))
plt.tight_layout()
plt.savefig("N_{:d}_TDV_vs_area.png".format(N))
plt.close()

# Plot mean TDVs vs amplitude
counts, xbins, ybins, image = plt.hist2d(data[:,2], data[:,3], bins=3*nspots_max, cmap='Greys', norm=colors.LogNorm())
#plt.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],linewidths=3)
plt.xlabel("lightcurve amplitude")
plt.ylabel("mean transit depth variation")
#plt.plot([0,0.4], [1,1.4], color='goldenrod', linewidth=3)
c1, c2 = 9.22, 1.63
plt.plot(np.linspace(0.005,0.095,10), 1+c1*np.linspace(0.005,0.095,10)**c2, color='goldenrod', linewidth=3)
plt.xlim((0,0.1))
plt.ylim((1,1.3))
plt.title("mode fit: y-1 = {:.2f}*x^{:.2f}".format(c1,c2))
plt.tight_layout()
plt.savefig("N_{:d}_TDV_vs_amplitude.png".format(N))
plt.close()

# Plot mean spot coverage vs amplitude
counts, xbins, ybins, image = plt.hist2d(data[:,2], area, bins=nspots_max, cmap='Greys', norm=colors.LogNorm())
plt.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],linewidths=3)
plt.plot([0,0.12], [0,0.12], color='goldenrod', linewidth=3)
plt.xlabel("lightcurve amplitude")
plt.ylabel("mean spot coverage")
plt.axis('equal')
plt.title("N = {:d}, alpha_max = {:d}".format(N, alpha_max))
plt.tight_layout()
plt.savefig("N_{:d}_area_vs_amplitude.png".format(N))
plt.close()

# Compute power-law fit to TDV vs amplitude using least squares
amax = 10
x = np.linspace(0.005, 0.01*amax-0.005, amax)
y = np.zeros(amax)
sigma = np.zeros(amax)
for i in range(amax):
  y[i] = np.mean(data[np.where(np.abs(data[:,2] - x[i]) < x[0]/2)][:,3])
  sigma[i] = np.std(data[np.where(np.abs(data[:,2] - x[i]) < x[0]/2)][:,3])
Y = np.transpose(np.log(y-1))
A = np.ones((amax,2))
A[:,1] = np.log(x)
C = np.zeros((amax,amax))
for i in range(amax):
  C[i,i] = np.log(sigma[i])**2
A_T = np.transpose(A)
C1 = np.linalg.inv(C)
ACA = np.dot(A_T, np.dot(C1, A))
ACY = np.dot(A_T, np.dot(C1, Y))
X = np.dot(np.linalg.inv(ACA), ACY)
#ATA = np.dot(A_T, A)
#ATY = np.dot(A_T, Y)
#X = np.dot(np.linalg.inv(ATA), ATY)
c1 = np.exp(X[0])
c2 = X[1]
plt.errorbar(x, y, yerr=sigma, fmt='o')
plt.plot(x, 1+c1*x**c2)
plt.xlabel("lightcurve amplitude")
plt.ylabel("mean transit depth variation")
plt.title("mean fit: y-1 = {:.2f} x^{:.2f}".format(c1, c2))
plt.tight_layout()
plt.savefig("llsq.png")

