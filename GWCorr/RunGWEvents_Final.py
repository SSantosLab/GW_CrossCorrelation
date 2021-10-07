# Author: Federico Berlfein
# coding: utf-8
# This python file is used to run the gravitational wave-galaxy catalog cross correlation pipeline
# It can be run from terminal using 'python RunGWEvents_Final.py'
# Before running, set the appropiate configurations in the NKCorr_Settings.py file



# First some imports that we'll use below
import treecorr
import numpy as np
import os
import matplotlib.pyplot as plt
from ligo.skymap.tool.ligo_skymap_plot import main
import NK_Correlation_GW_multEvents_Final as NK_GW
import BCCsims_PreProcessing as NK_PreProcessing
import Plotting


# Read specific configurations from settings file.

config_file = 'NKCorr_Settings.yaml'
config = treecorr.read_config(config_file)
min_completeness = config['min_completeness']
do_randoms = config['do_randoms']
skip_done = config['skip_done']
nest = config['nest']
h0_min = config['h0_min']
h0_max = config['h0_max']
h0_step = config['h0_step']

h0_values =np.arange(h0_min, h0_max + h0_step, h0_step)

gw_dir = config['gw_dir']
cat_dir = config['cat_dir']
outdir = config['outdir']


# Get the paths to the GW event files and galaxy catalog files. If gw_dir and cat_dir are files, the code will only run
# using those two files. If gw_dir is a directory and cat_dir is a file, it will assume all GW_events in gw_dir will
# use the same galaxy catalog. If both are gw_dir and cat_dir are directories, it will assume a one-to-one relationship
# with the files based on the name of the files by sorting them. Any other desired matching gw_files and cat_files
# can be done by the user here

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
            #num = file.split('_')[2].split('.')[0]
            #num = str(num[0:len(num) -9])
            #cat_files.append(cat_dir + 'Chinchilla-0Y3_v1.6_truth.' + num +'_hpix.fits')
            gw_files.append(os.path.join(r, file))
    gw_files.sort()
    for r, di, f in os.walk(cat_dir):
        for file in f:
            cat_files.append(os.path.join(r, file))
    cat_files.sort()
                
# Create a directory within the outpu dir where all NKObjects from events will go (one subdirectory for noRandoms and Randoms)
# as well as plots for combining all events processed so far.
outdir_all = outdir + 'all_events'
if (os.path.isdir(outdir_all) == False):
    os.mkdir(outdir_all)
if (os.path.isdir(outdir_all + '/NKObjects') == False):
    os.mkdir(outdir_all+ '/NKObjects')
if (os.path.isdir(outdir_all + '/NKObjects_Randoms') == False):
    os.mkdir(outdir_all+ '/NKObjects_Randoms')
outdir_all += '/'  

# Loop through each GW event and galaxy catalog
for gw_file, cat_file in zip(gw_files, cat_files):
    # Run PreProcessing first to obtain all the necessary inputs from both gw and galaxy catalog files
    GW_info = NK_PreProcessing.runPreProcessing(gw_file, cat_file, nest)
    z_gal = GW_info[0]
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
    outdir_event = outdir + GW_ID
    not_skip = True 
    if skip_done: # this will skip the event if it has already been processed and skip_done is set to true in settings
        not_skip = not (os.path.isdir(outdir_event))
        
    if (completeness > min_completeness and not_skip): # additionally, event will be skipped if min_completeness is not reached
       # create all appropiate subdirectories for event
        if (os.path.isdir(outdir_event) == False):
            os.mkdir(outdir_event)
        outdir_event += '/output_noRandoms'
        if (os.path.isdir(outdir_event) == False):
            os.mkdir(outdir_event)
        outdir_event += '/'
        
        outdir_randoms = outdir + GW_ID + '/output_Randoms'
        if (os.path.isdir(outdir_randoms) == False):
            os.mkdir(outdir_randoms)
        outdir_randoms += '/'
        
        
        # run the main cross-correlation code with all inputs
        NK_GW.run(h0_values, z_gal, lumd_pdf, weights_pdf, cat_ra, cat_dec, gw_prob, NSIDE, outdir_event, outdir_all, GW_ID, gw_distances, distsigma, nest,  pixels)
        
        # Generate all plots for analysis with no randoms in output_noRandoms
        
        # Sky localization of GW event
        main([gw_file, '--annotate', '--contour', '50', '90', '-o', outdir_event + 'Sky_Localization.png', '--figure-width', '18', '--figure-height', '16'])
        
        # 4 subplots containing general analysis
        fig, axs = plt.subplots(2, 2, figsize=(18, 12))
        axs[0, 0] = Plotting.plotGW(axs[0, 0], cat_file, gw_prob, pixels, NSIDE, completeness, nest)
        axs[0, 1] = Plotting.plotGWPrior( axs[0, 1], gw_prob, gw_distances, distsigma, distnorm )
        axs[1, 0]  = Plotting.plotGalCatZ(axs[1, 0], z_gal)
        axs[1, 1]  = Plotting.plotMaxCorr(axs[1,1], h0_values, outdir_event)
        fig.savefig(outdir_event + 'Analysis_Plots_noRandoms')
        
        # Plot with the cross-correlations for all H_0 values
        fig_ind = plt.figure(figsize=(30, 60))
        fig_ind = Plotting.plotIndCorr(fig_ind, h0_values, outdir_event)
        fig_ind.savefig(outdir_event + 'All_H0_NKCross_Plots')
        
        # output to all_events directory adding this event to the combination and MaxCorr Plot
        fig_combine = plt.figure(figsize=(10, 10))
        fig_combine = Plotting.plotCombineAverage(fig_combine, h0_values, (outdir_all + 'NKObjects/'))
        fig_combine = Plotting.plotCombineTreeCorr(fig_combine, h0_values, (outdir_all + 'NKObjects/'))
        fig_combine.savefig(outdir_all + 'combine_MaxCorr_Plot_noRandoms')
        
        if do_randoms:
            
            # Run the analysis with the randoms and generate all plots in output_Randoms
            NK_GW.run_randoms(h0_values, z_gal, lumd_pdf, weights_pdf, cat_ra, cat_dec, gw_prob, NSIDE, outdir_randoms, outdir_all, GW_ID, gw_distances, distsigma, nest,  pixels)
            
            # Plot of the cross-correlation of just the randoms for every cosmology
            fig_ind_rand = plt.figure(figsize=(30, 60))
            fig_ind_rand = Plotting.plotIndCorr(fig_ind_rand, h0_values, outdir_event, outdir_randoms)
            fig_ind_rand.savefig(outdir_randoms + 'All_H0_NKCross_Plots_withRandoms')
            
            # Plot of the cross-correlation of with randoms correction for every cosmology
            fig_ind_rand = plt.figure(figsize=(30, 60))
            fig_ind_rand = Plotting.plotIndCorr(fig_ind_rand, h0_values, outdir_randoms)
            fig_ind_rand.savefig(outdir_randoms + 'All_H0_NKCross_Plots_OnlyRandoms')
            
            # Max amplitude of correlation as a function of H_0 for only the randoms
            fig_rand, axs = plt.subplots(1, 1, figsize=(8, 6))
            axs  = Plotting.plotMaxCorr(axs, h0_values, outdir_event, outdir_randoms)
            fig_rand.savefig(outdir_randoms + 'MaxCorr_withRandoms')
            
            # Max amplitude of correlation as a function of H_0 with the randoms correction
            fig_rand, axs = plt.subplots(1, 1, figsize=(8, 6))
            axs  = Plotting.plotMaxCorr(axs, h0_values, outdir_randoms)
            fig_rand.savefig(outdir_randoms + 'MaxCorr_onlyRandoms')
            
            # output to all_events directory adding this event to the combination and MaxCorr Plot using randoms
            fig_combine = plt.figure(figsize=(10, 12))
            fig_combine = Plotting.plotCombineAverage(fig_combine, h0_values, (outdir_all + 'NKObjects/'), (outdir_all + 'NKObjects_Randoms/'))
            fig_combine = Plotting.plotCombineTreeCorr(fig_combine, h0_values, (outdir_all + 'NKObjects/'), (outdir_all + 'NKObjects_Randoms/'))
            fig_combine.savefig(outdir_all + 'combine_MaxCorr_Plot_withRandoms')

        
        
