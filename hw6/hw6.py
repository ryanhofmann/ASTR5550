#!/usr/bin/env python3

import numpy as np

def sampleStats(n=10):

  x = np.random.normal(0,1,n)
  mu = np.mean(x)
  var1 = 1./n*np.sum((x - 0)**2)
  var2 = 1./(n-1)*np.sum((x - mu)**2)
  varmu = var1/n
  varvar = 2.*var1/n

  return mu, var1, var2, varmu, varvar


if __name__=="__main__":

  n = 10
  np.random.seed(314)
  mu, var1, var2, varmu, varvar = sampleStats(n)
  print("mean: {:.3f}\nvariance given mu=0: {:.3f}\nvariance not given mu: {:.3f}\nvariance of mean: {:.3f} [0.100]\nvariance of variance: {:.3f} [0.200]".format(mu, var1, var2, varmu, varvar))

  N = 1000
  success = 0
  for i in range(N):
    mu, var1, var2, varmu, varvar = sampleStats(n)
    x1 = mu - varmu**0.5
    x2 = mu + varmu**0.5
    if x1 < 0 and x2 > 0:
      success += 1

  print("Success rate = {:.3f}".format(success/N))

