# Author: Federico Berlfein
# coding: utf-8
# This python file contains many useful functions and runs the cross-correlations after preprocessing.


# First some imports that we'll use below
from __future__ import print_function
import treecorr
import numpy as np 
import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import contour as Contour
from matplotlib.path import Path
import matplotlib.patches as patches
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import z_at_value
from astropy import units as u
import healpy as hp
import random
import os
from scipy import integrate
from scipy.stats import norm
from joblib import Parallel, delayed, parallel_backend
import pickle



# Global variables from setting file
config_file = 'NKCorr_Settings.yaml'
config = treecorr.read_config(config_file)
MIN_SEP = config['min_sep']
MAX_SEP = config['max_sep']
NBINS = config['nbins']
N_THREADS = config['n_threads']
N_JOBS = config['n_jobs']

os.environ['OMP_THREAD_LIMIT'] = str(N_THREADS) # set the maximum number of threads that can be used

# In[41]:


# function converts pixels from a healpix map into its RA,DEC centers  in radians
def pix2ang(pix, nside, nest = False):
    angle = hp.pix2ang(nside, pix, nest)
    theta = angle[0]
    ra_rad = angle [1]
    dec_rad = (90 - theta*180/np.pi)*np.pi/180
    return ra_rad, dec_rad


# function gets the pixel number for coordinates of ra and dec (in degrees)
def ang2pix (ra, dec, nside, nest = False):
    theta = 0.5*np.pi - np.deg2rad(dec)
    phi = np.deg2rad(ra)
    pix = hp.ang2pix(nside, theta, phi, nest)
    return pix



# Function creates a gaussian distribution of points in ra,dec, and distance around some center
# ra_sigma and dec_sigma is the width of the distirbution in ra, dec, and dist_sigma is the width for the distances
# num_gal is the number of galaxies wanted in the sample.
def rand_gauss(ra_center, dec_center, dist_center, ra_sigma, dec_sigma, dist_sigma, num_gal):   
    # make normal distirbutions of galaxies in ra, dec, and distance
    rand_ra = np.random.normal(ra_center, ra_sigma, num_gal) 
    rand_dec = np.random.normal(dec_center, dec_sigma, num_gal)  
    rand_dis = np.random.normal(dist_center, dist_sigma, num_gal) 
    return rand_ra, rand_dec, rand_dis




# function randomly and uniformly distributes points in ra,dec, distance
# ra_center, dec_center, dist_center represents the center of the distirbution
# a_sigma and dec_sigma is the width of the distirbution in ra, dec, and dist_sigma is the width for the distances
# num_gal is the number of galaxies wanted in the sample.
def rand_uniform(ra_center, dec_center, dist_center, ra_sigma,dec_sigma, dist_sigma, num_gal): 
    # get ranges for ra, dec, distance
    ra_min = np.min((ra_center - ra_sigma)*np.pi/180)
    ra_max = np.max((ra_center + ra_sigma)*np.pi/180)
    dec_min = np.min((dec_center - dec_sigma)*np.pi/180)
    dec_max = np.max((dec_center + dec_sigma)*np.pi/180)
    d_min = np.min(dist_center - dist_sigma)
    d_max = np.max(dist_center + dist_sigma)
    # distribute them uniformly
    rand_dis = np.random.uniform(d_min, d_max, num_gal)  
    rand_ra = np.random.uniform(ra_min, ra_max, num_gal)*180/np.pi 
    rand_sindec = np.random.uniform(np.sin(dec_min), np.sin(dec_max), num_gal) 
    rand_dec = np.arcsin(rand_sindec)*180/np.pi  #convert back to dec 
    return rand_ra, rand_dec, rand_dis



# function randomly and uniformly distributes points in ra,dec, distance
# ra_center, dec_center, dist_center represents the center of the distirbution
# deg_sigma is the width of the distirbution in ra and dec, and dist_sigma is the width of the distance distirbution
def rand_uniform_allsky(max_distance, num_galaxies, ra, dec):
    num_gal = num_galaxies
    ra_min = np.min((ra)*np.pi/180)
    ra_max = np.max((ra)*np.pi/180)
    dec_min = np.min((dec)*np.pi/180)
    dec_max = np.max((dec)*np.pi/180)
    d_min = 0.001
    d_max = np.max(max_distance)
    rand_dis = np.random.uniform(d_min, d_max, num_gal)  
    rand_ra = np.random.uniform(ra_min, ra_max, num_gal)*180/np.pi 
    rand_sindec = np.random.uniform(np.sin(dec_min), np.sin(dec_max), num_gal) 
    rand_dec = np.arcsin(rand_sindec)*180/np.pi  #convert back to dec 
    return rand_ra, rand_dec, rand_dis

# function randomly and uniformly and randomly distributes points in comoving volume. Uses formula for redshift distibution
# in uniform comooving volume given a specific cosmology.
def rand_uniform_comvolume_allsky(min_z, max_z, num_galaxies, cosmo, ra, dec):
    num_gal = num_galaxies
    ra_min = np.min(ra)*np.pi/180
    ra_max = np.max(ra)*np.pi/180
    dec_min = np.min(dec)*np.pi/180
    dec_max = np.max(dec)*np.pi/180
    z_lin = np.linspace(min_z, max_z, 100000000)
    dv_dz = 4*np.pi*cosmo.differential_comoving_volume(z_lin).value
    rand_zpdf = dv_dz/(1+ z_lin)
    rand_zpdf = rand_zpdf/sum(rand_zpdf)
    rand_z = np.random.choice(z_lin, num_galaxies, p = rand_zpdf) 
    rand_ra = np.random.uniform(ra_min, ra_max, num_gal)*180/np.pi 
    rand_sindec = np.random.uniform(np.sin(dec_min), np.sin(dec_max), num_gal) 
    rand_dec = np.arcsin(rand_sindec)*180/np.pi  #convert back to dec 
    return rand_ra, rand_dec, rand_z








# Function caluclates cross correlation with GW map. NK Correlation takes two types of catalogs:
# N Catalog (source catalog): galaxy catalog, usual ra,dec,distance coordinates
# K Catalog (prob_catalog): GW Map, where ra,dec comes from the pixel centers, k is the spatial probability, and all points
# have the same distance, which is the luminosity distance of the GW event.
# Function takles in the ra,dec,distance of the "blob" of galaxies, as well as the min and max separations I want
# to analyze.
def nk_corr(ra_corr, dec_corr, r_corr, dist_gw, prob, nk, ra_deg, dec_deg, weights = None, weights_gal = None):
    if weights_gal is not None:
        source_cat = treecorr.Catalog(ra = ra_corr, dec= dec_corr, r = (r_corr),w = weights_gal ,ra_units='deg', dec_units='deg')
    else:
        source_cat = treecorr.Catalog(ra = ra_corr, dec= dec_corr, r = (r_corr), ra_units='deg', dec_units='deg')
    if weights is not None:
        prob_cat = treecorr.Catalog(ra = ra_deg, dec=dec_deg, r = dist_gw ,k= prob, w = weights, ra_units='deg', dec_units='deg')
    else:
        prob_cat = treecorr.Catalog(ra = ra_deg, dec=dec_deg, r = dist_gw ,k= prob, ra_units='deg', dec_units='deg')
    nk.process_cross(source_cat, prob_cat,  num_threads= N_THREADS) # performs correlation
    vark = treecorr.calculateVarK(prob_cat)
    return nk, vark # returns separation distance (r_nk) and correlation function


# Given an unfinalized cross correlation, and random if given, it will finish the cross-correlation result
# Returns the separation and value of cross-correlation, the NKObject, and the variance
def nk_finalize(nk, vark, rk = None):
    nk_copy = nk.copy() # not to change original file
    nk_copy.finalize(vark)
    r_nk = nk_copy.meanr # The (weighted) mean value of r for the pairs in each bin.
    xi_nk, varxi_nk = nk_copy.calculateXi(rk) # xi_nk is the value of the correlation function
    return r_nk, xi_nk, nk_copy, varxi_nk # returns separation distance (r_nk) and correlation function  



# functions plots correlation function and returns the maximum value of the cross correlation and at what separation it occurs
def plot_corr(r_nk, xi_nk, title):
    plt.scatter (r_nk, xi_nk) 
    plt.ylabel(r'$\xi$ (r)')
    plt.xlabel('r (Mpc)')
    plt.title(title)
    plt.show()
    # finds and returns where the peak in the correlation is.
    max_xi = max(xi_nk)
    max_r = r_nk[np.argmax(xi_nk)]
    print('Maximum correlation at: %.3f Mpc' %max_r)
    return max_r, max_xi



# Given a type of distance (cosmo_distance = cosmo.luminosity_distance for example) and the distances it converts them to redshift
# by interpolating values. Returns redhsift values for given distances
def dist2redshift(cosmo_distance, distances):
    Dvals = distances * u.Mpc
    zmin = z_at_value(cosmo_distance, Dvals.min())
    zmax = z_at_value(cosmo_distance, Dvals.max())
    zgrid = np.logspace(np.log10(zmin), np.log10(zmax), 50)
    Dgrid = cosmo_distance(zgrid)
    return np.interp(Dvals.value, Dgrid.value, zgrid)


# Create linearly spaced distance measurements form gaussian pdf with range of 2-sigmas, and give each distance measurement
# a weights proportional to the value of the pdf at that distance squared
def create_pdf_linear(value, sigma, num_copies):  
    x = np.linspace((value-sigma*2),(value + sigma*2), num_copies)
    pdf = norm.pdf(x, value, sigma)
    weights = pdf**2
    return x, weights


# Calculate the wieght of a galaxy by taking the value of that distance in the pdf corresponding to the pixel in the GW event
# the galaxy belongs to.
def weightGalDist(cat_ra, cat_dec, NSIDE,nest, z, prob, m ,cosmology, gw_distances, distsigma):
    m_tot = np.arange(hp.nside2npix(NSIDE))
    prob_tot = np.array([0.0]* len(m_tot))
    gw_dist_tot = np.array([0.0]* len(m_tot))
    distsigma_tot = np.array([1.0]* len(m_tot))
    prob_tot[m] = prob
    gw_dist_tot[m] = gw_distances
    distsigma_tot[m] = distsigma
    pix_gal = ang2pix (cat_ra, cat_dec, NSIDE, nest)
    lumd_dist = cosmology.luminosity_distance(z).value
    weights = (norm.pdf(lumd_dist, gw_dist_tot[pix_gal], distsigma_tot[pix_gal]))#**2
    return weights 


# Calculates cross-correlation by transofrming redshifts from galaxies and luminosity distances from GW event into comoving 
# distance given a cosmology. It does this by cross-correlating each instance of the GW event and accumulating the counts
# Additionally, it updates the log for that specific cosmology, printing how long it took to calculate the last cross-corr
# how many of the 100 instances of the GW event have been done, and th expected time to finish. 
# Finally, it returns the unfinished NKObject containing the counts.
def cosmo_corr(cosmology, z, lum_distances, weights, prob_map, cat_ra, cat_dec, ra_deg, dec_deg, log_outdir, m, gw_distances, distsigma, NSIDE, nest):
    weights_gal = None
    weights_gal = weightGalDist(cat_ra, cat_dec, NSIDE, nest, z, prob_map, m ,cosmology, gw_distances, distsigma)
    dist_cat = cosmology.comoving_distance(z).value # convert redshifts from galaxies to comoving distance
    nk = treecorr.NKCorrelation(min_sep= MIN_SEP, max_sep=MAX_SEP , nbins=NBINS) # set new NKObject
    z_gw = []
    for lumd in lum_distances: # for each instance of gw event calculate redshifts
        z_gw.append(dist2redshift(cosmology.luminosity_distance, lumd))
    dist_center = cosmology.comoving_distance(np.array(z_gw)).value # convert redfshifts to comoving distance
    vark = []
    tot_time = time.time()
    for j in range(0, len(dist_center)): # loop through each instance of GW event
        start_time = time.time()
        if weights[j] is not None:
            weight = weights[j]
        else:
            weight = None
            
        source_ra = cat_ra
        source_dec = cat_dec
        source_dis = dist_cat
        # calculate cross-corr
        nk, n_vark= nk_corr(source_ra, source_dec, source_dis, dist_center[j], prob_map, nk,ra_deg, dec_deg, weight, weights_gal)
        vark.append(n_vark)
        # update logs
        time_taken = (time.time() - start_time)
        exp_time = (time_taken * (len(dist_center) - (j+1)))
        time_taken = time.strftime('%H:%M:%S', time.gmtime(time_taken))
        exp_time = time.strftime('%H:%M:%S', time.gmtime(exp_time))
        log_file = open(log_outdir + '/log_' + str(cosmology.H(0).value),"w") 
        log_file.write(str(j+1) + '/' + str(len(dist_center)) + ' complete\nExpected time to finish: ' + exp_time + '\nLast Iteration time: ' + time_taken )
        log_file.close()
    # update logs once finished
    log_file = open(log_outdir + '/log_' + str(cosmology.H(0).value),"w")
    tot_end = (time.time() - tot_time)
    tot_end = time.strftime('%H:%M:%S', time.gmtime(tot_end))
    log_file.write('Complete! ' + str(len(dist_center)) + '/' + str(len(dist_center)) + '\nTime taken: ' + tot_end )
    log_file.close()
    return nk

# Calculates the cross-correlation for randoms distributed unifromly and randomly in comoving volume. The random galaxy catalog
# created has the same shape and occupies the same pixels as the galaxy catalog, as well as the same redshift range.
# Besides this, the rest of the calculation is identical to cosmo_corr.
def cosmo_corr_randoms(cosmology, z, lum_distances, weights, prob_map, cat_ra, cat_dec, ra_deg, dec_deg, log_outdir, m, gw_distances, distsigma, NSIDE, nest):
    pix_gal = ang2pix (cat_ra, cat_dec, NSIDE, nest) # the pixels of the galaxy catalog
    pix_area = hp.nside2pixarea(NSIDE, degrees = True)*len(list(set(pix_gal))) # sky area of galaxy catalog
    #pix_frac = pix_area/ hp.nside2pixarea(NSIDE, degrees = True)*hp.nside2npix(NSIDE)
    max_z = np.max(z)
    min_z = np.min(z)
    #comv_volume = cosmology.comoving_volume(max_z).value*pix_frac
    #density = 1.13*10**(-14)
    #num_gal = density* comv_volume
    num_gal = len(cat_ra) # number of randoms is equal to number of galaxies in catalog
    ra_diff = np.absolute(np.max(cat_ra) - np.min(cat_ra))
    dec_diff = np.absolute(np.sin(np.deg2rad(np.max(cat_dec))) - np.sin(np.deg2rad(np.min(cat_dec))))
    # so random catalog has same number of galaxies as catalog after masking, we make more galaxies in square patch
    square_area = (180/np.pi)*ra_diff*dec_diff # calculate the square area of a patch of sky given the galaxy catalog
    num_galaxies = int(num_gal*(square_area)/pix_area) # number of galaxies to randomly distribute in square patch
    rand_ra, rand_dec, rand_z = rand_uniform_comvolume_allsky(min_z, max_z, num_galaxies, cosmology, cat_ra, cat_dec)
    #rand_ra, rand_dec, rand_z = rand_uniform_allsky(max_z, num_galaxies)
    # masking
    pix_rand = ang2pix (rand_ra, rand_dec, NSIDE, nest)
    good_pix = np.in1d(pix_rand, pix_gal)
    cat_ra = rand_ra[good_pix]
    cat_dec = rand_dec[good_pix]
    z = rand_z[good_pix]
    weights_gal = None
    weights_gal = weightGalDist(cat_ra, cat_dec, NSIDE, nest, z, prob_map, m ,cosmology, gw_distances, distsigma)
    dist_cat = cosmology.comoving_distance(z).value # convert redshifts from galaxies to comoving distance
    nk = treecorr.NKCorrelation(min_sep= MIN_SEP, max_sep=MAX_SEP , nbins=NBINS)
    z_gw = []
    for lumd in lum_distances: # for each instance of gw event calculate redshifts
        z_gw.append(dist2redshift(cosmology.luminosity_distance, lumd))
    dist_center = cosmology.comoving_distance(np.array(z_gw)).value
    vark = []
    tot_time = time.time()
    for j in range(0, len(dist_center)): # loop through each instance of GW event
        start_time = time.time()
        if weights[j] is not None:
            weight = weights[j]
        else:
            weight = None
            
        source_ra = cat_ra
        source_dec = cat_dec
        source_dis = dist_cat
        # calculate cross-corr
        nk, n_vark= nk_corr(source_ra, source_dec, source_dis, dist_center[j], prob_map, nk,ra_deg, dec_deg, weight, weights_gal)
        vark.append(n_vark)
        # update logs
        time_taken = (time.time() - start_time)
        exp_time = (time_taken * (len(dist_center) - (j+1)))
        time_taken = time.strftime('%H:%M:%S', time.gmtime(time_taken))
        exp_time = time.strftime('%H:%M:%S', time.gmtime(exp_time))
        log_file = open(log_outdir + '/log_' + str(cosmology.H(0).value),"w") 
        log_file.write(str(j+1) + '/' + str(len(dist_center)) + ' complete\nExpected time to finish: ' + exp_time + '\nLast Iteration time: ' + time_taken )
        log_file.close()
    log_file = open(log_outdir + '/log_' + str(cosmology.H(0).value),"w")
    tot_end = (time.time() - tot_time)
    tot_end = time.strftime('%H:%M:%S', time.gmtime(tot_end))
    log_file.write('Complete! ' + str(len(dist_center)) + '/' + str(len(dist_center)) + '\nTime taken: ' + tot_end )
    log_file.close()
    return nk



def run(h0_values, z_pdf, lumd_pdf, weights_pdf, cat_ra, cat_dec, gw_prob, NSIDE, outdir, outdir_all, GW_ID, gw_distances, distsigma, nest = False, m = None):
    log_outdir = os.path.join(outdir, 'logs')
    if (os.path.isdir(log_outdir) == False):
        os.mkdir(log_outdir)
    if m is None:
        m = np.arange(hp.nside2npix(NSIDE))
    ra_pix, dec_pix = pix2ang (m, NSIDE, nest) # pixel centers for a healpix map in radians
    # ra_deg and dec_deg, the pixel centers, will be associated with the probabilites for each pixel from GW
    # contour maps
    ra_deg = ra_pix*180/np.pi
    dec_deg = dec_pix*180/np.pi
    
    #run correlations
    nk_list = [treecorr.NKCorrelation(min_sep= MIN_SEP, max_sep=MAX_SEP , nbins=NBINS) for i in range (len(h0_values))]
    results = Parallel(n_jobs = N_JOBS)(delayed(cosmo_corr)(FlatLambdaCDM(H0=h0, Om0=0.286, Ob0=0.046), z_pdf, lumd_pdf, weights_pdf, gw_prob, cat_ra, cat_dec, ra_deg, dec_deg, log_outdir, m, gw_distances, distsigma, NSIDE, nest) for h0 in h0_values) 
    for j in range (len(results)):
        nk_list[j].__iadd__(results[j])
        #var_list.append(results[j].estimate_cov('jackknife'))
    pickle_out = open((outdir + GW_ID + '_NKObject' ) ,"wb")
    pickle.dump(nk_list, pickle_out)
    pickle_out.close()
    pickle_out_all = open((outdir_all + 'NKObjects/' + GW_ID + '_NKObject' ) ,"wb")
    pickle.dump(nk_list, pickle_out_all)
    pickle_out_all.close()
    
    
    
def run_randoms(h0_values, z_pdf, lumd_pdf, weights_pdf, cat_ra, cat_dec, gw_prob, NSIDE, outdir, outdir_all, GW_ID, gw_distances, distsigma, nest = False, m = None):
    log_outdir = os.path.join(outdir, 'logs')
    if (os.path.isdir(log_outdir) == False):
        os.mkdir(log_outdir)
    if m is None:
        m = np.arange(hp.nside2npix(NSIDE))
    ra_pix, dec_pix = pix2ang (m, NSIDE, nest) # pixel centers for a healpix map in radians
    # ra_deg and dec_deg, the pixel centers, will be associated with the probabilites for each pixel from GW
    # contour maps
    ra_deg = ra_pix*180/np.pi
    dec_deg = dec_pix*180/np.pi
    
    #run correlations
    nk_list = [treecorr.NKCorrelation(min_sep= MIN_SEP, max_sep=MAX_SEP , nbins=NBINS) for i in range (len(h0_values))]
    results = Parallel(n_jobs = N_JOBS)(delayed(cosmo_corr_randoms)(FlatLambdaCDM(H0=h0, Om0=0.286, Ob0=0.046), z_pdf, lumd_pdf, weights_pdf, gw_prob, cat_ra, cat_dec, ra_deg, dec_deg, log_outdir, m, gw_distances, distsigma, NSIDE, nest) for h0 in h0_values) 
    for j in range (len(results)):
        nk_list[j].__iadd__(results[j])
        #var_list.append(results[j].estimate_cov('jackknife'))
    pickle_out = open((outdir + GW_ID + '_NKObject_Randoms' ) ,"wb")
    pickle.dump(nk_list, pickle_out)
    pickle_out.close()
    pickle_out_all = open((outdir_all + 'NKObjects_Randoms/' + GW_ID + '_NKObject_Randoms' ) ,"wb")
    pickle.dump(nk_list, pickle_out_all)
    pickle_out_all.close()





# Fit a curve to a polynomial
def fit(x,y, num_points = 100):
    p = np.poly1d(np.polyfit(x, y, 10))
    xp = np.linspace(0, np.max(x) - 20, num_points)
    #plt.plot(x, y, '.', xp, p(xp), '--')
    return xp, p(xp)



# plots cross correlation for every H_0 and and the maximum value for every H_0  
# returns the maximum  correlation value of every cosmology
def point_max_plot(r, xi, h0_values, plot_title):
    fig, (ax1, ax2) = plt.subplots(2, 1,figsize = (8,10))
    diff_maxi = []
    for i in range(0, len(h0_values)):
        ax1.scatter(r[i], xi[i], label = 'H_0 = %i' % h0_values[i])
        diff_maxi.append(np.max(xi[i]))

    ax1.set_title('NK Correlation at different H_0 Values, ' + plot_title)
    ax1.set_ylabel(r'$\xi$ (r)')
    ax1.set_xlabel('r (Mpc)')
    #ax1.legend()

    ax2.plot(h0_values, diff_maxi)
    ax2.scatter(h0_values, diff_maxi)
    ax2.axvline( h0_values[np.argmax(diff_maxi)],color = 'r', linestyle = '--')
    ax2.set_xlabel('H_0 value')
    ax2.set_ylabel('Peak Amplitude')
    ax2.set_title('Peak Amplitude from NK cross correlation')
    #ax2.set_yscale('log')
    ax2.set_xticks(np.arange(np.min(h0_values), np.max(h0_values) + 2, 10))
    #fig.show()
    plt.show()
    #fig.close()
    return diff_maxi


# plots cross correlation for every H_0 by fitting the cross-corr and and the maximum value for every H_0  
# returns the maximum  correlation value of every cosmology
def fit_max_plot(r, xi, h0_values, plot_title):
    fig, (ax1, ax2) = plt.subplots(2, 1,figsize = (8,10))
    diff_maxi = []
    for i in range(0, len(h0_values)):
        x, y = fit(r[i],xi[i])
        diff_maxi.append(np.max(y))
        #ax1.scatter(r_nk_tot[i], xi_nk_tot[i]*10000, label = 'H_0 = %i' % h0_values[i])
        ax1.plot(x, y, '--', label = 'H_0 = %i' % h0_values[i])
    ax1.set_title('NK Correlation at different H_0 Values, ' + plot_title)
    ax1.set_ylabel(r'$\xi$ (r)')
    ax1.set_xlabel('r (Mpc)')
    #ax1.legend()

    ax2.plot(h0_values, diff_maxi)
    ax2.scatter(h0_values, diff_maxi)
    ax2.axvline( h0_values[np.argmax(diff_maxi)],color = 'r', linestyle ='--')
    ax2.set_xlabel('H_0 value')
    ax2.set_ylabel('Peak Amplitude')
    ax2.set_title('Peak Amplitude from fitted NK cross correlation')
    #ax2.set_yscale('log')
    ax2.set_xticks(np.arange(np.min(h0_values), np.max(h0_values) + 2, 10))
    #fig.show()
    plt.show()
    #fig.close()
    return diff_maxi


# plots cross correlation for every H_0 and the integral value for every H_0  
def fit_integral_plot(r, xi, h0_values, plot_title):
    fig, (ax1, ax2) = plt.subplots(2, 1,figsize = (8,10))
    diff_maxi = []
    for i in range(0, len(h0_values)):
        x, y = fit(r[i],xi[i])
        #diff_maxi.append(np.trapz(y,x))
        diff_maxi.append(integrate.simps(y,x))
        ax1.plot(x, y, '--', label = 'H_0 = %i' % h0_values[i])
    ax1.set_title('NK Correlation at different H_0 Values, ' + plot_title)
    ax1.set_ylabel(r'$\xi$ (r)')
    ax1.set_xlabel('r (Mpc)')
   #ax1.legend()

    ax2.plot(h0_values, diff_maxi)
    ax2.scatter(h0_values, diff_maxi)
    ax2.axvline( h0_values[np.argmax(diff_maxi)],color = 'r', linestyle = '--')
    ax2.set_xlabel('H_0 value')
    ax2.set_ylabel('Integral of fitted NK cross correlation')
    ax2.set_title('Integral of fitted NK cross correlation')
    #ax2.set_yscale('log')
    #ax2.set_ylim(20,150)
    ax2.set_xticks(np.arange(np.min(h0_values), np.max(h0_values) + 2, 10))
    #fig.show()
    plt.show()
    #fig.close()
    return diff_maxi


 

# # Finding confidence region pixels. This is not my code, code is taken from Fermi GBM Data Tools, specifically 
#   the gbm.data.localization module
#     Authors: William Cleveland (USRA),
#              Adam Goldstein (USRA) and
#              Daniel Kocevski (NASA)

# create the mesh grid in phi and theta
def mesh_grid(NSIDE, num_phi, num_theta, nest):
    theta = np.linspace(np.pi, 0.0, num_theta)
    phi = np.linspace(0.0, 2 * np.pi, num_phi)
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    grid_pix = hp.ang2pix(NSIDE, theta_grid, phi_grid, nest)
    return (grid_pix, phi, theta)

# use greedy algortihgm to find the credible levels
def find_greedy_credible_levels(p):
    p = np.asarray(p)
    pflat = p.ravel()
    i = np.argsort(pflat)[::-1]
    cs = np.cumsum(pflat[i])
    cls = np.empty_like(pflat)
    cls[i] = cs
    return cls.reshape(p.shape)

# find the confidence region path 
def confidence_region_path(NSIDE, clevel, prob, nest, numpts_ra=360, numpts_dec=180):
    # create the grid and integrated probability array
    sig =  1- find_greedy_credible_levels(prob)
    grid_pix, phi, theta = mesh_grid(NSIDE,numpts_ra, numpts_dec, nest)
    sig_arr = 1.0 - sig[grid_pix]
    ra = np.rad2deg(phi)
    dec = np.rad2deg(np.pi / 2.0 - theta)

    # use matplotlib contour to produce a path object
    contour = Contour(ra, dec, sig_arr, [clevel])

    # get the contour path, which is made up of segments
    paths = contour.collections[0].get_paths()

    # extract all the vertices
    pts = [path.vertices for path in paths]

    # unfortunately matplotlib will plot this, so we need to remove
    for c in contour.collections:
        c.remove()
        plt.close()
    plt.close()
    #print(pts)
    return pts

def find_points(NSIDE, nest, pts):
    # Nside for Healpix map
    m = np.arange(hp.nside2npix(NSIDE))
    ra_pix, dec_pix = pix2ang(m, NSIDE, nest)
    ra_deg = ra_pix*180/np.pi
    dec_deg = dec_pix*180/np.pi
    healpy_points = list(zip(ra_deg, dec_deg))
    p = Path(pts) 
    patch = patches.PathPatch(p, facecolor='orange')
    plt.close()
    grid = p.contains_points(healpy_points)
    return grid

# find the pixels that conform the confidence region
def find_conf_pixels(NSIDE, nest, clevel, prob):
    pts = confidence_region_path(NSIDE, clevel, prob, nest)
    pix_cut = np.full(len(prob), False)
    for points in pts:
        is_in_region = find_points(NSIDE, nest, points)
        pix_cut = np.logical_or(pix_cut, is_in_region)
    return pix_cut