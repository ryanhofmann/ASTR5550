#!/usr/bin/env python3

import numpy as np
import os
import pyfits
import matplotlib.pyplot as plt

# Read FITS files to header and data arrays
raw_names = []
for name in os.listdir("."):
  if name.endswith("fixed_header.fits"):
    raw_names.append(name)
raw_names.sort()
headers = []
datas = []
for name in raw_names:
  image = pyfits.open(name)
  headers.append(image[0].header)
  datas.append(image[0].data)

# Print filters, integration times, data types, and ranges
print("FILTER\tITIME\tDTYPE\tMIN\tMAX")
for i in range(len(headers)):
  FILTER = headers[i]['FILTER']
  EXPOSURE = headers[i]['EXPOSURE']
  DTYPE = datas[i].dtype.name
  MIN = np.min(datas[i])
  MAX = np.max(datas[i])
  print("{}\t{:.0f}\t{}\t{:d}\t{:d}".format(FILTER,EXPOSURE,DTYPE,MIN,MAX))

# Combine darks
imtype = float
darks = np.array(datas[6:9]).astype(imtype)
dark_master = np.median(darks, axis=0)

# Plot dark_master
from matplotlib.colors import LogNorm
dpi = 80
margin = 0.05
xpixels, ypixels = len(dark_master[0]), len(dark_master)
figsize = (1 + margin)*xpixels/dpi, (1 + margin)*ypixels/dpi
fig = plt.figure(figsize=figsize, dpi=dpi)
ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
i = ax.imshow(dark_master, cmap='gray', interpolation=None, norm=LogNorm(), vmax=10*np.mean(dark_master))
ax.invert_yaxis()
plt.title("dark_master")
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(i, cax=cax)
plt.savefig("dark_master.jpg")
plt.clf()

# Combine biases
biases = np.array(datas[0:3]).astype(imtype)
bias_master = np.median(biases, axis=0)

# Plot bias_master
fig = plt.figure(figsize=figsize, dpi=dpi)
ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
i = ax.imshow(bias_master, cmap='gray', interpolation=None, vmax=0.3*np.max(bias_master))
ax.invert_yaxis()
plt.title("bias_master")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(i, cax=cax)
plt.savefig("bias_master.jpg")
plt.clf()

# Combine flats
flats = np.array(datas[3:6]).astype(imtype)
flats_bsub = flats - bias_master
from scipy import stats
repeats = [stats.find_repeats(flats_bsub[i]) for i in range(3)]
modes = np.array([repeats[i].values[np.where(repeats[i].counts==np.max(repeats[i].counts))] for i in range(3)])
flats_norm = np.array([flats_bsub[i]/modes[i] for i in range(3)])
flats_comb = np.median(flats_norm, axis=0)
repeats = stats.find_repeats(flats_comb)
mode = repeats.values[np.where(repeats.counts==np.max(repeats.counts))]
flat_master = flats_comb / mode

# Plot flat_master
fig = plt.figure(figsize=figsize, dpi=dpi)
ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
i = ax.imshow(flat_master, cmap='gray', interpolation=None)
ax.invert_yaxis()
plt.title("flat_master")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(i, cax=cax)
plt.savefig("flat_master.jpg")
plt.clf()

# Reduce M57 frames
M57 = np.array(datas[9:13]).astype(imtype)
M57_reduced = (M57 - dark_master)/flat_master

# Plot data images
c = ["L", "R", "G", "B"]
for i in range(4):
  fig = plt.figure(figsize=figsize, dpi=dpi)
  ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
  ax.imshow(M57_reduced[i], cmap='gray', norm=LogNorm(vmin=0.4*np.median(M57_reduced[i]), vmax=0.2*M57_reduced[i].max()), interpolation=None)
  ax.invert_yaxis()
  plt.title("M57 {}".format(c[i]))
  plt.savefig("M57_{}.jpg".format(c[i]))
  plt.clf()

# Compute mean and sigma of first bias frame
bias_mean = np.mean(biases[0], axis=None)
bias_sigma = np.std(biases[0], axis=None)
print("\nBias 1: mean= {:.1f}, sigma= {:.1f}".format(bias_mean, bias_sigma))

# Plot histogram of first bias image
plt.figure(figsize=(16,12), dpi=None)
plt.hist(biases[0].flatten(), bins=int(biases[0].max()-biases[0].min()))
plt.yscale('log')
plt.ylim((0.1,1e6))
plt.title("bias 1 histogram")
plt.savefig("bias_1_hist.jpg")
plt.clf()

# Plot histogram of difference bias1 - bias2
diff = biases[0] - biases[1]
plt.hist(diff.flatten(), bins=int(diff.max()-diff.min()))
plt.yscale('log')
plt.ylim((0.1,1e6))
plt.title("bias difference histogram")
plt.savefig("bias_diff_hist.jpg")
plt.clf()

# Use difference to compute sigma
diff_sigma = np.std(diff, axis=None)
bias_sigma_diff = diff_sigma/np.sqrt(2)
print("Diff: sigma= {:.1f}; derived single: sigma= {:.1f}".format(diff_sigma, bias_sigma_diff))

# Compute readnoise
readnoise = bias_sigma_diff*np.sqrt(3)/3
print("Readnoise= {:.1f}".format(readnoise))

# Use difference of two flats to calculate gain
diff = flats[0] - flats[1]
gains = np.zeros(10)
for i in range(10):
  diff_var = np.var(diff[600:850,800+50*i:800+50*(i+1)])*0.5
  flat_mean = np.mean(flats[0][600:850,800+50*i:800+50*(i+1)])
  gains[i] = flat_mean / diff_var
gain = np.mean(gains)
gain_sigma = np.std(gains)
print("\nGain= {:.2f} +/- {:.2f}".format(gain, gain_sigma))
print("from header: gain= {:.2f}".format(headers[3]['E-GAIN']))

# Plot histograms of dark frame
plt.hist(darks[0].flatten(), bins=200)
plt.yscale('log')
plt.ylim((0.1,4e6))
plt.title("dark 1 histogram")
plt.savefig("dark_1_hist.jpg")
plt.clf()
plt.hist(darks[0].flatten(), bins=np.arange(451).astype(int))
plt.yscale('log')
plt.xlim(0,450)
plt.ylim((0.1,1e6))
plt.title("dark 1 histogram, cropped to same limits as bias 1")
plt.savefig("dark_1_hist_crop.jpg")
plt.clf()

# Compute mean and sigma of dark frame
dark_mean = np.mean(darks[0])
dark_sigma = np.std(darks[0])
print("\nDark 1: mean= {:.1f}, sigma= {:.1f}".format(dark_mean, dark_sigma))

# Plot histograms of difference of two darks
diff = darks[0] - darks[1]
plt.hist(diff.flatten(), bins=200)
plt.yscale('log')
plt.ylim(0.1,4e6)
plt.title("dark difference histogram")
plt.savefig("dark_diff_hist.jpg")
plt.clf()
plt.hist(diff.flatten(), bins=np.arange(-80,81).astype(int))
plt.yscale('log')
plt.xlim(-80,80)
plt.ylim(0.1,1e6)
plt.title("dark difference histogram, cropped to same limits as bias difference")
plt.savefig("dark_diff_hist_crop.jpg")
plt.clf()

# Compute mean and sigma of difference of two darks
diff_mean = np.mean(diff)
diff_sigma = np.std(diff)
print("Diff: mean= {:.1f}, sigma= {:.1f}".format(diff_mean, diff_sigma))

# Calculate contributions from readnoise and dark current
diff_sigma_dark = np.sqrt((diff_sigma**2 - 2*bias_sigma_diff**2)/2)
print("Total= {:.1f}; readnoise= {:.1f}, dark current= {:.1f}".format(diff_sigma, np.sqrt(2)*bias_sigma_diff, np.sqrt(2)*diff_sigma_dark))

# Estimate dark current contribution to uncertainty
dark_current = np.sqrt(diff_sigma_dark**2/3 + diff_sigma_dark**2)
print("Dark current= {:.1f}".format(dark_current))

# Plot histogram of sky background; use L image
sky = M57_reduced[0][950:1000,200:300]
plt.hist(sky.flatten(), bins=100)
plt.title("sky histogram")
plt.savefig("sky_histogram.jpg")
plt.clf()

# Compute mean and sigma of sky background
sky_mean = np.mean(sky)
sky_sigma = np.std(sky)
print("\nSky: mean= {:.1f}, sigma= {:.1f}".format(sky_mean, sky_sigma))

# Calculate sky noise
sky_noise = np.sqrt(sky_sigma**2 - readnoise**2 - dark_current**2)
sky_noise_expected = np.sqrt(np.mean(sky))
print("Calculated noise= {:.1f}; expected noise= {:.1f}".format(sky_noise, sky_noise_expected))

# Estimate total uncertainty in star peak amplitude
star = M57_reduced[0][475:500,1425:1450]
star_peak = np.max(star)
total_sigma = np.sqrt(readnoise**2 + dark_current**2 + sky_noise**2 + star_peak)
print("\nTotal= {:.1f}, readnoise= {:.1f}, dark current= {:.1f}, sky noise= {:.1f}, Poisson noise= {:.1f}".format(total_sigma, readnoise, dark_current, sky_noise, np.sqrt(star_peak)))

