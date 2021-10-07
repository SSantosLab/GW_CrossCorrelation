# Author: Federico Berlfein
# coding: utf-8
# This python file is used to generate all plots used in the 'RunGWEvents_Final.py'


# First some imports that we'll use below
from __future__ import print_function
import treecorr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
from astropy.table import Table
import healpy as hp
import os
import NK_Correlation_GW_multEvents_Final as NK_GW
import ligo.skymap as skymap
from ligo.skymap import distance
import pickle

# set some global variables from the configuration file

config_file = 'NKCorr_Settings_zerr.yaml'
config = treecorr.read_config(config_file)
RA_COL = config['ra_col_name']
DEC_COL = config['dec_col_name']
REDSHIFT_COL = config['redshift_col_name']
MIN_SEP = config['min_sep']
MAX_SEP = config['max_sep']
NBINS = config['nbins']

os.environ['KMP_DUPLICATE_LIB_OK']='True' # this is done to run skymap.distance



# Given the h_0 values, the directory of the cross-correlation with no randoms and the cross-correlation of only randoms if given
# returns a figure with the finalized cross-correlation for every cosmology
def plotIndCorr(fig, h0_values, directory, rand_directory = None):
    num_rows = int(len(h0_values)/4.0 + 1)
    axs = []
    r, xi = plotNKFinalized(directory, h0_values, rand_directory)
    for i in range(len(h0_values)):
        axs.append(fig.add_subplot(num_rows, 4, i+1))
        axs[-1].scatter (r[i], xi[i]) 
        axs[-1].set_ylabel(r'$\xi$ (r)')
        axs[-1].set_xlabel('r (Mpc)')
        axs[-1].set_title('NKCorrelation for H_0 = ' + str(h0_values[i]))
        if (np.min(xi) > 0):
            axs[-1].set_ylim(0, 16/15*np.max(xi))
        else:
            axs[-1].set_ylim(np.min(xi), 16/15*np.max(xi))
    plt.close(fig)
    return fig



# Using the TreeCorr adding method, this function combines all the cross correlations in a given directory for a given
# H_0 values, and does the same if a randoms directory is given. It then finalizes the cross correlation using ranomds
# if given and returns the finalized cross-correlation result in xi_nk_tot for each radius of separation r_nk_tot.
def plotNKFinalized(directory, h0_values, rk_dir = None):
    nk_list = []
    for r, di, f in os.walk(directory):
        for file in f:
            if '_NKObject' in file:
                pickle_in = open(os.path.join(r, file) ,'rb')
                nk_list.append(pickle.load(pickle_in))
    if (rk_dir is not None):
        rk_list = []
        for r, di, f in os.walk(rk_dir):
            for file in f:
                if '_NKObject_Randoms' in file:
                    pickle_in = open(os.path.join(r, file) ,'rb')
                    rk_list.append(pickle.load(pickle_in))
    r_nk_tot = []
    xi_nk_tot = []
    #max_r_tot = []
    #max_xi_tot = []
    for i in range(len(h0_values)):
        nk_combine = treecorr.NKCorrelation(min_sep= MIN_SEP, max_sep=MAX_SEP , nbins=NBINS)
        rk_combine = treecorr.NKCorrelation(min_sep= MIN_SEP, max_sep=MAX_SEP , nbins=NBINS)
        for j in range(len(nk_list)):
            nk_combine.__iadd__(nk_list[j][i]) # combines nk_cross correlations for same cosmology
            if(rk_dir is not None):
                rk_combine.__iadd__(rk_list[j][i]) # combines nk_cross correlations for the randoms
        rand_nk, xirand_nk, rk_combine, varrand_xi= NK_GW.nk_finalize(rk_combine, 1)
        r_nk, xi_nk, nk_one, var_xi= NK_GW.nk_finalize(nk_combine, 1, rk_combine)
        #max_xi = np.max(xi_nk)
        #max_r = r_nk[np.argmax(xi_nk)]
        r_nk_tot.append(r_nk)
        xi_nk_tot.append(xi_nk)
        #max_r_tot.append(max_r)
        #max_xi_tot.append(max_xi)
    return r_nk_tot, xi_nk_tot


# This function plots the GW Event and Galaxy catalog overlaped (in 2D). It also shows the percentage of pixels in the GW
# Event that also have galaxies in the Galaxy catalog. For the galaxy catalog it shows all galaxies with redshift < 1.
# Returns axis given of figure object given as input
def plotGW(axs, cat_file, prob,m, NSIDE, completeness, nest = False):
    data_cat = Table.read(cat_file)
    cat_ra = np.array(data_cat[RA_COL])
    cat_dec = np.array(data_cat[DEC_COL])
    z_pdf = np.array(data_cat[REDSHIFT_COL])
    cut = z_pdf<1.0
    z_pdf = [z_pdf[cut]]
    cat_ra = cat_ra[cut]
    cat_dec = cat_dec[cut]
    n_pixels = np.arange(hp.nside2npix(NSIDE))
    clevel = 0.95
    prob_tot = np.array([0.0]* len(n_pixels))
    prob_tot[m] = prob
    conf_cut = NK_GW.find_conf_pixels(NSIDE, nest, clevel, prob_tot) # pixels in 95% confidence region of event
    m_conf = np.arange(hp.nside2npix(NSIDE))[conf_cut]
    ra_pix, dec_pix = NK_GW.pix2ang(m_conf, NSIDE, nest) # pixel centers for a healpix map in radians
    ra_deg = ra_pix*180/np.pi
    dec_deg = dec_pix*180/np.pi
    prob_conf = prob_tot[conf_cut]
    axs.hexbin(cat_ra,cat_dec, cmap = 'viridis')
    axs.hexbin(ra_deg, dec_deg, C = prob_conf, cmap = 'inferno')#, gridsize = 1400)
    axs.set_xlabel('ra')
    axs.set_ylabel('dec')
    axs.set_title('GW Event and Galaxy Catalog Overlap, Percent Overlap: %.3f' % completeness)
    return axs


# For plotting Gaussian approximation of GW luminosity distance posterior
# Returns axs of figure object. Inputs are the luminosity distance for each pixel in the event, probability of each pixel,
# distance error, and distance normalized
def plotGWPrior(axs, prob, gw_distances, distsigma, distnorm ):
    r = np.linspace(50,3000, 200)
    dist_pdf = skymap.distance.marginal_pdf(r,np.array(prob),np.array(gw_distances), np.array(distsigma), np.array(distnorm) )
    #dist_pdf = dist_pdf/sum(dist_pdf)
    mean_dist = r[np.argmax(dist_pdf)]
    axs.plot(r, dist_pdf)
    axs.set_title('GW Luminosity Distance Posterior')
    axs.set_xlabel('Luminosity Distance (Mpc)')
    axs.axvline(mean_dist, color = 'orange', linestyle = '--', label = 'Mean Value = %.1f Mpc' % mean_dist)
    axs.set_ylabel('pdf')
    axs.legend()
    
    return axs


# For plotting the redshift distribution of the galaxy catalog and the number of galaxies used in analysis
# Returns axs object
def plotGalCatZ(axs, cat_z):
    axs.hist(cat_z, bins = 20)
    axs.set_title('Galaxy Catalog Redshift Histogram. Number of Galaxies: ' + str(len(cat_z)))
    axs.set_xlabel('Redshfit')
    axs.set_xlim(0, np.max(cat_z))
    return axs


# Given an axs object, H_0 values, and the directory of the cross correlations and randoms if given, plots for each H_0 value
# the maximum value of the cross-correlation. Returns axs object
def plotMaxCorr(axs, h0_values, directory, rand_directory = None):
    r, xi = plotNKFinalized(directory, h0_values, rand_directory)
    diff_maxi = []
    for i in range(0, len(h0_values)):
        diff_maxi.append(np.max(xi[i]))

    axs.plot(h0_values, diff_maxi)
    axs.scatter(h0_values, diff_maxi)
    axs.axvline( h0_values[np.argmax(diff_maxi)],color = 'r', linestyle = '--')
    axs.set_xlabel('H_0 value')
    axs.set_ylabel('Peak Amplitude')
    axs.set_title('Peak Amplitude from NK cross correlation')
    axs.set_xticks(np.arange(np.min(h0_values), np.max(h0_values) + 2, 10))
    axs.set_xlim(np.min(h0_values)-4, np.max(h0_values)+4)
    axs.set_ylim(0, 16/15*np.max(diff_maxi))
    axs.margins(10.0)
    return axs


# Given H_0 values and directories where the cross-correlations are, and randoms if given, will plot the maximum correlation
# for each H_0 value by finalizing each correlation individually and obtaining an average cross-correlation value. This
# method of combination is not the same as the TreeCorr combination method, as cross-correlations for each event are finalized
# and then combined in an average. It then plots the maximum correlation value for each H_0 value and fits the curve produced
# using a polynomial.
def plotCombineAverage(fig, h0_values, directory, rk_dir = None):
    #nk_list = [treecorr.NKCorrelation(min_sep= 10, max_sep=1000 , nbins=100) for i in range (len(h0_values))]
    nk_list = []
    for r, di, f in os.walk(directory):
        for file in f:
            if '_NKObject' in file:
                pickle_in = open(os.path.join(r, file) ,'rb')
                nk_list.append(pickle.load(pickle_in))
                #nk_combine.__iadd__(nk_corr.copy())
    if (rk_dir is not None):
        rk_list = []
        for r, di, f in os.walk(rk_dir):
            for file in f:
                if '_NKObject_Randoms' in file:
                    pickle_in = open(os.path.join(r, file) ,'rb')
                    rk_list.append(pickle.load(pickle_in))
    r_nk_tot = []
    xi_nk_tot = []
    max_r_tot = []
    max_xi_tot = []
    # use average combination method
    for i in range(len(h0_values)):
        nk_combine = treecorr.NKCorrelation(min_sep= MIN_SEP, max_sep=MAX_SEP , nbins=NBINS)
        xi_combine = np.full(NBINS, 0.0)
        for j in range(len(nk_list)):
            rk_corr = None
            if(rk_dir is not None):
                r_nk, xi_nk, rk_corr, var_xi= NK_GW.nk_finalize(rk_list[j][i], 1)
            r_nk, xi_nk, nk_one, var_xi= NK_GW.nk_finalize(nk_list[j][i], 1, rk_corr)
            xi_combine += (xi_nk)
        xi_combine = xi_combine/len(nk_list)
        max_xi = np.max(xi_combine)
        max_r = r_nk[np.argmax(xi_combine)]
        r_nk_tot.append(r_nk)
        xi_nk_tot.append(xi_combine)
        max_r_tot.append(max_r)
        max_xi_tot.append(max_xi)
    
    # plot maximum correlation for each H_0 and fit curve
    x = h0_values
    y = max_xi_tot
    p = np.poly1d(np.polyfit(x, y, 30))
    xp = np.linspace(np.min(x), np.max(x), 100)
    f_max = np.max( p(xp))
    max_h0 = xp[np.argmax(p(xp))]
    axs= fig.add_subplot(2, 1, 1)
    axs.plot(x, y, 'o', xp, p(xp), '--')
    axs.axvline( h0_values[np.argmax(y)],color = 'r', linestyle = '--', label = 'Max H0 = %.1f' % max_h0)
    axs.set_title('Combined NK Cross-Correlation Peak Using Average')
    axs.set_xlabel('H0 Value')
    axs.set_ylabel('Peak Amplitude Cross Corrrelation')
    axs.legend()
    plt.close(fig)
    return fig

# Given H_0 values and directories where the cross-correlations are, and randoms if given, will plot the maximum correlation
# for each H_0 value by combining all cross-correlations using treecorr and then finalizing. It then plots the maximum correlation
# value for each H_0 value and fits the curve produced using a polynomial.
def plotCombineTreeCorr(fig, h0_values, directory, rand_directory = None):
    r, xi = plotNKFinalized(directory, h0_values, rand_directory) # combination of cross-correlation using TreeCorr
    max_xi_tot = []
    for i in range(0, len(h0_values)):
        max_xi_tot.append(np.max(xi[i]))
    
    # plot maximum correlation for each H_0 and fit curve
    x = h0_values
    y = max_xi_tot
    p = np.poly1d(np.polyfit(x, y, 30))
    xp = np.linspace(np.min(x), np.max(x), 100)
    f_max = np.max( p(xp))
    max_h0 = xp[np.argmax(p(xp))]
    axs= fig.add_subplot(2, 1, 2)
    axs.plot(x, y, 'o', xp, p(xp), '--')
    axs.axvline( h0_values[np.argmax(y)],color = 'r', linestyle = '--', label = 'Max H0 = %.1f' % max_h0)
    axs.set_title('Combined NK Cross-Correlation Peak Using TreeCorr')
    axs.set_xlabel('H0 Value')
    axs.set_ylabel('Peak Amplitude Cross Corrrelation')
    axs.legend()
    plt.close(fig)
    return fig





