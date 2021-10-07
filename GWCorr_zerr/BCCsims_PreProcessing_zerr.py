# Author: Federico Berlfein
# coding: utf-8
# This python file is used for PreProcessing in the 'RunGWEvents_Final.py'. Given just the files for the Settings, GW Event and
# and the galaxy catalog, this code processes and produces all neccesary inputs to run cross-correlations

# First some imports that we'll use below
import treecorr
import numpy as np
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import z_at_value
from astropy import units as u
import healpy as hp
import os
import NK_Correlation_GW_multEvents_Final_zerr as NK_GW
import ligo.skymap as skymap
from ligo.skymap import distance

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# global variable from settings
config_file = 'NKCorr_Settings_zerr.yaml'
config = treecorr.read_config(config_file)
RA_COL = config['ra_col_name']
DEC_COL = config['dec_col_name']
REDSHIFT_COL = config['redshift_col_name']
REDSHIFT_ERR_COL = config['redshift_err_col_name']
PROB_COL = config['prob_col_name']
DISTMU_COL = config['distmu_col_name']
DISTSIGMA_COL = config['distsigma_col_name']
DISTNORM_COL = config['distnorm_col_name']
min_h0 = config['h0_min']
max_h0 = config['h0_max']




# This function selects the pixels from the galaxy catalog that are contained in the confidedence region given (clevel) of the
# GW event. It also returns the "completeness", which is the percentage of the pixels in the GW event's confidence region that
# contain galaxies in the galaxy catalog.
def good_pixels(prob_temp, cat_ra, cat_dec, NSIDE, nest, pixels, clevel):
    pix_cut = NK_GW.find_conf_pixels(NSIDE, nest, clevel, prob_temp) # find pixels in the GW confidence region
    pix_gal = NK_GW.ang2pix (cat_ra, cat_dec, NSIDE, nest)
    m_gal = np.arange(hp.nside2npix(NSIDE))[pix_cut]
    good_pix = np.in1d(pix_gal, m_gal) # what pixels are shared between GW event and galaxy catalog confidence region
    pix_conf = pix_gal[good_pix] # mask those pxiels
    if (len(m_gal) == 0): # if there are no pixels in confidence region, use all pixels from event
        good_pix = np.in1d(pix_gal, pixels)
        pix_conf = pix_gal[good_pix]
        completeness = len(set(pix_conf))/len(set(pixels))
    else:
        completeness = len(set(pix_conf))/len(set(m_gal)) # determine completeness
    return good_pix, completeness



# Estimate the GW Luminosity Distance posterior using ligo's skymap.distance.marginal_pdf to obtain the min and max range 
# of the luminosity distance for the event
def posterior_range(prob, gw_distances, distsigma, distnorm):
    r = np.linspace(5,5000, 1000)
    dist_pdf = distance.marginal_pdf(r, (prob), (gw_distances), (distsigma), (distnorm))
    r = r[dist_pdf >0.001] # this basically filters all non-zero pdf values
    max_range = np.max(r)
    min_range = np.min(r)
    return max_range, min_range


# Main method in running preproccessing. It takes both the gw file and galaxy catalog file.
# It returns the galaxies that are contained in the 95% confidence region of the GW event (in the form of cat_ra, cat_dec, z_gal)
# From the GW event, it returns the pixels (with their respective prob, distance, distance error, distance norm) 
# with non-zero probability and whose distance errors are below 50%. It also returns the NSIDE of the event and its Event ID.
# lumd_pdf and weights_pdf are represent of each pixel's pdf, where each pixel has 100 linearly spaced distancemeasurements
# from its luminosity distance pdf, and the weights are the pdf value at that distance squared. For more details on this
# refer to the README file.
def runPreProcessing(gw_file, cat_file, nest = False):
    # The minimum and maximum cosmology given the H_0 range
    cosmo_max = FlatLambdaCDM(H0=max_h0, Om0=0.286, Ob0=0.046)
    cosmo_min = FlatLambdaCDM(H0=min_h0, Om0=0.286, Ob0=0.046)
    GW_event_ID = gw_file.split('/')[-1].split('.')[0] # the event ID is the name of the file
    # read gw file
    data = Table.read(gw_file)
    prob = np.array(data[PROB_COL])
    gw_distances = np.array(data[DISTMU_COL])
    distsigma = np.array(data[DISTSIGMA_COL])
    distnorm = np.array(data[DISTNORM_COL])
    gw_distances[gw_distances == np.inf] = 0   # taking inf distance to be zero
    gw_distances = np.absolute(gw_distances)
    NSIDE = hp.get_nside(prob)
    # pixel masking
    clevel = 0.95
    pix_cut = prob > 0 # only take pixels with non-zero probability
    #pix_cut = NK_GW.find_conf_pixels(NSIDE, nest, clevel, prob)
    sigma_cut = distsigma/gw_distances < 0.5 # only take pixels with < 50% error
    # applying the cuts to pixels
    cut = np.logical_and(pix_cut, sigma_cut)
    prob = prob[cut]
    gw_distances = gw_distances[cut]
    distsigma = distsigma[cut]
    distnorm = distnorm[cut]
    #print(len(prob))
    m = np.arange(hp.nside2npix(NSIDE))[cut] # the pixels to be used
    # look at luminosity distance posterior range
    max_range, min_range = posterior_range(prob, gw_distances, distsigma, distnorm)
    # obtain z_range using minimum and maximum cosmologies
    min_z = z_at_value(cosmo_min.luminosity_distance,min_range*u.Mpc )
    max_z = z_at_value(cosmo_max.luminosity_distance,max_range*u.Mpc )
    # get lumd_pdf and weights_pdf
    lumd_pdf, weights_pdf = NK_GW.create_pdf_linear(gw_distances,distsigma, 100) # number at the end is # of points in Luminosity distance pdf to chose from
    # look at galaxy catalog
    data_cat = Table.read(cat_file)
    cat_ra = np.array(data_cat[RA_COL])
    cat_dec = np.array(data_cat[DEC_COL])
    z_gal = np.array(data_cat[REDSHIFT_COL])
    z_err = np.array(data_cat[REDSHIFT_ERR_COL])
    # redshift cut from posterior
    max_cut = z_gal < max_z 
    min_cut = z_gal > min_z
    cut = np.logical_and(max_cut, min_cut)
    z_gal = z_gal[cut]
    z_err = z_err[cut]
    cat_ra = cat_ra[cut]
    cat_dec = cat_dec[cut]
    # select galaxies in 95% conf region
    prob_temp = np.array(data[PROB_COL])
    clevel = 0.95
    good_pix, completeness = good_pixels(prob_temp, cat_ra, cat_dec, NSIDE, nest, m, clevel)
    cat_ra = cat_ra[good_pix]
    cat_dec = cat_dec[good_pix]
    z_gal = z_gal[good_pix]
    z_err = z_err[good_pix]
    z_pdf, z_weights_pdf = NK_GW.create_pdf_linear(z_gal,z_err, 25)
    cat_ra = np.tile(cat_ra,25)
    cat_dec = np.tile(cat_dec,25)
    z_pdf = z_pdf.flatten()
    z_weights_pdf = z_weights_pdf.flatten()
    lumd_pdf = np.array([lumd_pdf.flatten()])
    weights_pdf = np.array([weights_pdf.flatten()])
    return [z_pdf, z_weights_pdf, lumd_pdf, weights_pdf, cat_ra, cat_dec, prob, NSIDE, GW_event_ID, m, gw_distances, distsigma,distnorm, completeness]







