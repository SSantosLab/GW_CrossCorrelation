#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# First some imports that we'll use below
from __future__ import print_function
import treecorr
#import fitsio
import numpy
import math
import time
import pprint
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import StrMethodFormatter, NullFormatter 
from scipy import signal
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import Distance
from astropy.cosmology import WMAP5 as cosmo5
from astropy.cosmology import WMAP9 as cosmo9
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import Planck13, z_at_value
from astropy import cosmology
import healpy as hp
import random
from collections import Counter
from astropy.io import ascii 
import os
import mockmaps
from scipy import stats
from scipy import signal
from scipy.signal import find_peaks
from scipy.stats import norm
import matplotlib.mlab as mlab
import multiprocessing as mp
from joblib import Parallel, delayed, parallel_backend
import NK_Correlation_GW_multEvents_Final as NK_GW
import BCCsims_PreProcessing as NK_PreProcessing
import Plotting
import pickle


# In[ ]:


os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
#treecorr.set_omp_threads(5)

# # Load and Run NK Corr on GW events

# In[ ]:



#h0_values = numpy.arange(40,72,2)
#h0_values =numpy.arange(72,102,2)
h0_values =numpy.arange(40,102,2)
min_completeness = 0.7

gw_dir = 'des40a/antonella_files/massmin_12.0_massmax_13.0_1/rotated_1570_117001126777.fits'
cat_dir = 'des40a/antonella_files/truth_all/'
outdir_all = 'des40a/Final_BCC_NKObjects/'

gw_files = []
cat_files = []

if (os.path.isfile(gw_dir) and os.path.isfile(cat_dir)):
    gw_files.append(gw_dir)
    cat_files.append(cat_dir)
    
elif (os.path.isdir(gw_dir) and os.path.isfile(cat_dir)):
    for r, di, f in os.walk(gw_dir):
        for file in f:
            cat_files.append(cat_dir)
            gw_files.append(os.path.join(r, file))
elif (os.path.isfile(gw_dir) and os.path.isdir(cat_dir)):
    file = gw_dir
    num = file.split('/')[3].split('_')[2].split('.')[0]
    num = str(num[0:len(num) -9])
    cat_files.append(cat_dir + 'Chinchilla-0Y3_v1.6_truth.' + num +'_hpix.fits')
    gw_files.append( file)

else:
    for r, di, f in os.walk(gw_dir):
            for file in f:
                num = file.split('_')[2].split('.')[0]
                num = str(num[0:len(num) -9])
                cat_files.append(cat_dir + 'Chinchilla-0Y3_v1.6_truth.' + num +'_hpix.fits')
                gw_files.append(os.path.join(r, file))

for gw_file, cat_file in zip(gw_files, cat_files):
    nest = False
    GW_info = NK_PreProcessing.runPreProcessing(gw_file, cat_file, nest)
    z_pdf = GW_info[0]
    lumd_pdf = GW_info[1]
    weights_pdf = GW_info[2]
    cat_ra = GW_info[3]
    cat_dec = GW_info[4]
    gw_prob = GW_info[5]
    NSIDE = GW_info[6]
    GW_ID = GW_info[7]
    pixels = GW_info[8]
    gw_distances = GW_info[9]
    distsigma = GW_info[10]
    distnorm = GW_info[11]
    completeness = GW_info[12]
    if (completeness > min_completeness):
        outdir = outdir_all + GW_ID
        if (os.path.isdir(outdir) == False):
            os.mkdir(outdir)
        outdir += '/output_noRandoms'
        if (os.path.isdir(outdir) == False):
            os.mkdir(outdir)
        outdir += '/'
        
        outdir_all += '/all_events'
        if (os.path.isdir(outdir_all) == False):
            os.mkdir(outdir_all)
        if (os.path.isdir(outdir_all + '/NKObjects') == False):
            os.mkdir(outdir_all+ '/NKObjects')
        outdir_all += '/'
        

        NK_GW.run(h0_values, z_pdf, lumd_pdf, weights_pdf, cat_ra, cat_dec, gw_prob, NSIDE, outdir, outdir_all, GW_ID, gw_distances, distsigma, nest,  pixels)
       
        
        fig, axs = plt.subplots(2, 2, figsize=(18, 12))
        axs[0, 0] = Plotting.plotGW(axs[0, 0], cat_file, gw_prob, pixels, NSIDE, completeness)
        axs[0, 1] = Plotting.plotGWPrior( axs[0, 1], gw_prob, gw_distances, distsigma, distnorm )
        axs[1, 0]  = Plotting.plotGalCatZ(axs[1, 0], z_pdf)
        axs[1, 1]  = Plotting.plotMaxCorr(axs[1,1], h0_values, outdir)
        fig.savefig(outdir + 'Analysis_Plots_noRandoms')

        fig_ind = Plotting.plotIndCorr(h0_values, outdir)
        fig_ind.savefig(outdir + 'All_H0_NKCross_Plots')
        
        fig_combine = Plotting.plotCombineAverage(h0_values, (outdir_all + 'NKObjects/'))
        fig_combine.savefig(outdir_all + 'combine_MaxCorr_Plot_noRandoms')
