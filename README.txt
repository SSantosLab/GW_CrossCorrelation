Author: Federico Berlfein

In this file I will try to explain the GW Cross Correlation pipeline. Note that in the repository there are two folders. The one with the "zerr" extensions is
the latest one. I will explain the differences later on.

Main idea:
We have two inputs, a galaxy catalog, with gaalxies speificied by RA, DEC, redshift (z), and Gravitational Wave (GW) Catalog, represented by a probability sky map,
where each area of the sky is pixelated and assigned a probability of hosting the GW events, a given luminosity distance, and uncertainty on the lum distance.
For this purposes we assume the luminosity distance of each pixel is a gaussian, and the values given are the mean and width respectively. From this collection
of data from the GW event, we can also calculate the posterior pdf of the luminosity distance for the event. In order to cross correlate these two catalogs,
we need to have a common coordinate system. We want everything in RA, DEC, comoving distance. Galaxies are already in RA and DEC, but in order to convert redshifts
into comoving distance we need a cosmology. Similarly for the GW event, we can convert the pixels into RA,DEC by taking the coordinate of the center of the pixel.
But to convert from luminosity distance to comoving we also need a cosmology. So what we do is, we take a prior on the cosmology (in this case we fix everything
and only vary H0), and we calculate the cross-correlation given this cosmology between the GW event and the galaxy catalog. We do this for each cosmology,
and then compare (plot) the maximum correlation associated with each value of H0. So we infer H0 based on a maximum correlation analysis. 

This is relatively easy if we did not have uncertainties both in luminosity distance and redshift, but we do. So here is how we deal with them. For the luminosity
distances, the uncertainties can be very big (20-30%) and dominate over the redshift uncertainties. So, what we do is we create 100 different versions of the same
GW event, each one representing a different luminosity distance. To be more clear, each pixel in a GW event has a luminosity distance distribution (a gaussian),
we uniformly choose 100 luminosity distances from this distribution, and assign a weight to each one proportional to the value of that luminosity distance's pdf 
squared. You may ask why we chose to square it? I made this decision in order to amplify the signal coming from values closer to the mean, as using a simple 1 to 1
proportion did not seem to work. So now we have 100 copies of the same GW event in order to represent the luminosity distance distribution of each pixel. For the galaxies
we do the exact same, but with the redshift, and instead of 100 copies we use 25, since the redshift errors are smaller and don't dominate as much as the lum dist errors.
So now that we have 100 versions of the GW catalog, and 25 versions of the galaxy catalog, we cross correlate each one with on another. IMPORTANT: when using Treecorr,
which is the tool we use to calculate cross correlations, there is a "finished" and an "unfinished calculation". The finished one is used when cross-correlating two
catalogs and then dividing by the weights, but since we are doing many correlations for the same catalog, we want to accumulate all cross correlations first and
then finish them. Similarly, if we are doing a correlation over many GW events, we want to do this "unfinished" calculation for all of them, combine them, and then
finish them. This is key. Because we will be saving these unfnished calculations and finishing them later depending on what we want. 

IMPORTANT: Using randoms with this process can be tricky. Usually we want to have a random catalog to do a sort of correction of the NK Correlation. However, we want
the randoms to be the galaxies, which works well when you don't redshift errors. But when you introduce redshift errors, doing the randoms is not as clear. So I
would advice not to use randoms when doing the correlationusing the code with extensions z_err, and using it when you dont'.

WARNING: Be careful of computing resources available. This cross-correlations are VERY computationally expensive, and we are doing a lot of them. There is parallel
processing implemented here to increase computing speed, but still be ware when using large catalogs, you could be using a lot of RAM.


The Pipeline has 5 main files:

1. NKCorr_Settings_zerr.yaml: This yaml file contains the "settings" for running the cross correlation pipeline. In here you specify the path of where the GW 
catalog(s) and Galaxy catalog(s) are. If you pass two files here, it will simply compute the cross correlation between both. If you pass a directory for the
GW catalogs, and a file for the galaxy catalog, it assumes you want to calculate the Cross-corr of every GW Catalog with one galaxy catalog. If you want to pass
both was directories, you need to manually write a function that relates every galaxy catalog in the directory to a galaxy catalog in the other directory. Other 
setting descriptions are in the file themselves.

2. BCCsims_PreProcessing_zerr.py: Ignore the name BCCsims, this is a name I gave when I originally started working with the Blind Cosmology Challenge simulation and
never bothered changing the name. This is really just preprocessing of the data. Given the settings, it performs certain cuts in the GW Catalog and Galaxy Catalog,
and produces all the necesssary inputs for the Cross-Correlation. There are two types of cuts: cuts made to the GW catalog (which is represented in pixels and a 
probability sky map) and galaxy catalog (points in the sky given by RA, DEC, z). The first cut is done to the GW catalog, eliminating pixels with luminosity distance
uncertainty higher than 50%. It also considers only pixels with non-zero probability of hosting the event. GW catalogs have a Luminosity distance distribution,
I only consider the 68% confidence region of the lumunosity distance distribution, so up to 2 sigma in width.
From that distribution we pick 100 luminosity distances uniformly and assign weights to each one proportional to the value of the pdf^2. Cuts on the galaxy catalog:
The first one comes from the luminosit distance posterior, we look at the the minimum and maximum luminosity distance, and combined with the minimum and maximum
value of H0 we want to run our cosmology over, we determine the minimum and maximum redshift we need for the galaxy catalog. So we make a cut in redshift here.
Finally, we only consider galaxies that are within a certain confidence region of the GW event. SO for example, a sensible choice is to say I want to only consider
galaxies in the 95% confidence region of the GW event probability map.

3. NK_Correlation_GW_multEvents_Final_zerr.py: This is where the magic happens. This file contains almost all the functions used in the cross-correlation. 
You can read the specific descriptions of each function in the file itself (warning: some functions became depreciated during the process and I stopped 
using themselves, but I kept just in case).

4. Plotting_zerr.py: This file contains some simple plotting functions that are called later on to make plots. More details in the file itself.

5. RunGWEvents_Final_zerr.py: This is the file you actually run in the terminal. This file reads all the settings, call preprocessing, runs the GW events from
the NK_Correlation file, creates all the relevant directories and plots. To understand the outputs better, please refer to the powerpoint in the github. It is
much clearer to see it from there. But in essence, after calculating the cross-correlation of a given galaxy catalog and GW event, the pipeline
creates a directory in the desired output directory specified in the settings, where the "unfinished" calcuation of the NK cross-corr is saved as a pickled object,
and if you want to calculate the randoms cross corr those are also saved as a separate object. There is also a log where you can see how long the process took,
and some useful plots of the cross-correlation, the specifics of the catalog, etc. 