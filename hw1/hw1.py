#
# hw1.py
# Benjamin Brown
# ASTR 5550
# January 24, 2017
#
import numpy as np
import matplotlib.pyplot as plt
import time
import pyfits

def read_data(file):
    file = pyfits.open(file)
    data = file['LIGHTCURVE'].data
    raw_times = data['TIME']
    raw_lightcurve = data['SAP_FLUX']
    raw_PDC_lightcurve = data['PDCSAP_FLUX']
    
    # clean out NANs and infs in raw data stream
    # slightly more points are bad in "PDC" version.
    good_data = np.isfinite(raw_PDC_lightcurve)
    lightcurve = raw_lightcurve[good_data]
    PDC_lightcurve = raw_PDC_lightcurve[good_data]
    times = raw_times[good_data]

    N_good_points = len(lightcurve)
    N_bad_points = len(raw_lightcurve)-N_good_points
    print("{:d} good points and "
          "{:d} bad points in lightcurve".format(N_good_points, N_bad_points))

    # note: the PDC_lightcurve is a corrected lightcurve
    # from the Kepler data pipleine, that fixes some errors. 
    # PDC means "Pre-Search Data Conditioning"
    return times, lightcurve, PDC_lightcurve
    
if __name__ == "__main__":

    filename = 'kplr007200111-2009350155506_llc.fits'
    times, lightcurve, PDC_lightcurve = read_data(filename)

