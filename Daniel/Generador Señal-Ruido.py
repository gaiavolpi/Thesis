
# coding: utf-8

# Following code aim is: 
# - to create from a file_name and path a structure of folders where to save all corresponding images
# - to create signal images from a parameter list (or which are contained in a parameter hyperspace)
# - to create noise images 
# - to keep records of each signal image the parameters used in such image
# - zip the folder


import pycbc.waveform as pycbc_wf
import pycbc.psd as pycbc_psd
import pycbc.noise as pycbc_noise
import pycbc.types as pycbcty
import pycbc.detector as pycbc_det
import pycbc.catalog as pycbc_cat
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
import shutil as sh
import os as os
import astropy.cosmology as astr_cosm

# Turn interactive plotting off
plt.ioff()


# In the following cell we create the paths and folders required to store the different images: signal and noise, and training and test.



# creates paths where to store images and records

file_name = 'CNN_Images_Run24'

root_dir = '/data/danibelt' # root path in Cuda2

base_dir = os.path.join(root_dir, file_name)

training_dir = os.path.join(base_dir, 'Training')
test_dir = os.path.join(base_dir, 'Test')

training_signal_dir = os.path.join(training_dir, 'Signal')
training_noise_dir = os.path.join(training_dir, 'Noise')

test_signal_dir = os.path.join(test_dir, 'Signal')
test_noise_dir = os.path.join(test_dir, 'Noise')

# create folder to store everything
os.mkdir(base_dir)

# create folder to store training, and inside create folders to store signal and noise
os.mkdir(training_dir)
os.mkdir(training_noise_dir)
os.mkdir(training_signal_dir)

# create folder to store test, and inside create folders to store signal and noise
os.mkdir(test_dir)
os.mkdir(test_noise_dir)
os.mkdir(test_signal_dir)


# Here we first define the frecuency cutoff, minimun frecuency to consider. Then we define the parameters corresponding to training signal images and test signal images. In fact, we expand the parameters hyperspace.
# 
# As parameters to generate the GW template we consider $\bf{mass}$, same for both compact objects, $\bf{distance}$ to the source (input to the function is in Mpc) and $\bf{inclination}$, the angle between the angular orbital momentum of the inspiral and the line of sight. All this three are parameters to be considered in the template generator. Later, we 'create' a GW detector where we porject the emitted GW onto it to know what wave would these detector measure based on the detector location in Earth and sky location of the source. For that, we consider another parameter: $\bf{RA}$ and $\bf{DEC}$, which will be considered when projecting the wave.



# low frecuency limit
cut_off = 30

#############################################################################################################
###############################################  TRAIN PARAMETERS  ##########################################
#############################################################################################################

# source paramters
mass_tr = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50]
#distance_tr = [500] # in Mpc


distance_red_tr = [0.001, 0.002, 0.003, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 1.5, 2, 3] # in redshift
distance_tr = np.array(astr_cosm.Planck15.luminosity_distance(np.array(distance_red_tr))) # convertion to Mpc with Planck18 cosmology


# orbit inclination
inclination_tr = [0,0.25, 0.5, 0.75, 1, 1.5]

# sky location of the source (randomly) --> for the moment 3 locations
ra_tr = np.squeeze(np.random.random(size = (1, 4)) * 2 * np.pi)
dec_tr = np.squeeze(np.random.random(size = (1,4)) * np.pi)

#############################################################################################################
#############################################################################################################




############################################################################################################
###############################################  TEST PARAMETERS  ##########################################
############################################################################################################

# source paramters
mass_te = [12, 15, 17, 21, 23, 25, 27.5, 29.4, 31.5, 31.6, 32.8, 33, 34.9, 35.2, 37, 39, 46]
#distance_te = [500] # in Mpc


distance_red_te = [0.0001, 0.005, 0.008, 0.02, 0.04, 0.08, 0.2, 0.7, 1, 2.3, 2.4] # in redshift
distance_te =np.array(astr_cosm.Planck15.luminosity_distance(np.array(distance_red_te))) # convertion to Mpc with Planck18 cosmology

# orbit inclination
inclination_te = [0, 0.5, 1, 1.5]

# sky location of the source (randomly) --> for the moment 3 locations
ra_te = np.squeeze(np.random.random(size=(1,3)) * 2 * np.pi)
dec_te = np.squeeze(np.random.random(size=(1,3)) * np.pi)

#############################################################################################################
#############################################################################################################


# define our detector to project GW onto it
Virgo = pycbc_det.Detector('V1')


# Here we create a function which when called, it will generate a GW of mass "m", emitted from a inspiral with inclination "incl" and emitted from a distance "dist".




def template(m, incl, dist, cut_off):
    
    """
    Builds a template por a GW merger of both masses 'm' using SEOBNRv4_opt approximation method with a 
    sample of 2048 data per second (2048Hz)
    ---------------------------------------------------------------------------------------------------
    
    Arguments:
    m -- masses of both compact black holes
    incl -- angle between angular momentum L and line of sight (from 0 to PI)
    dist -- distance in Mpc to the emitting source
    cut_off -- low frecuency limit
    
    Return:
    hp, hc -- pycbc.timeseries.TimeSeries, for the plus/cross polarization GW     
    
    """
    hp, hc = pycbc_wf.get_td_waveform(approximant = "SEOBNRv4_opt", mass1 = m, mass2 = m, inclination = incl, distance = dist, delta_t = 1.0 / 2048, f_lower = cut_off)

    return hp, hc


# Here we project the previously generated GW form onto our detector to know what waveform would the detector measured based on the relative locations of detector and source. We consider when calling these function the polarizations of the GW wave emitted and sky location of the source.



def Virgo_gw(hp, hc, ra, dec):
    
    """
    Project the GW emitted by a source onto the detector to obtain the wave measured by a particular
    detector depending on coordinates of the detector in the Earth and coordinates of the source in the sky
    ------------------------------------------------------------------------------------------------------
    
    Arguments:
    hp, hc -- pycbc.timeseries-type for the plus/cross polarization GW generated at the source
    ra, dec -- sky location of the source
    pol -- polarization (set to 1)
    
    Return:
    gw -- pycbc.timeseries.TimeSeries, GW as measured by the detector

    """
        
    gw = Virgo.project_wave(hp, hc, ra, dec, 1, method = 'lal')
    
    return gw


# Now we compute the background noise of the detector: first we import a Power Spectral Density (PSD) from a detector (these case Advance Virgo) or particular noise sources (seismic, thermic, quantum ...). Then, after setting parameters such as length of the strain or frecuency of the data, we generate the corresponding gaussian noise from the previous PSD. Each time called, a different noise will be created since random seed is not specified.



# we import NOISE from Virgo detector

# first we import Adv Virgo PSD from LAL Simulations (https://pycbc.org/pycbc/latest/html/pycbc.psd.html)

delta_f = 1.0 / 16  # 16 data samples per second
flength = int(2048 / delta_f)  # 2048 seconds of data with 16 data each second 
low_freq_cutoff = cut_off  # corresponding the cutoff of our GW model

AdvV_PSD = pycbc_psd.analytical.AdvVirgo(flength, delta_f, low_freq_cutoff)

# from our simulated PSD now we obtain our gausian noise (https://pycbc.org/pycbc/latest/html/pycbc.noise.html#module-pycbc.noise.gaussian)

delta_t = 1.0 / 2048
time_length = 50 #  seconds of data
length = int(time_length / delta_t)  # seconds of noise data with 2048 data per second  samples




def noise_gen(length, delta_t, PSD):
    
    """
    From a previous simulated PSD it generates gaussian noise
    ---------------------------------------------------------
    
    Arguments:
    length -- length of the array of noise
    delta_t -- time in sec between two consecutive data
    PSD -- Power Spectral Density of the desired noise
    
    Return:
    strain_noise -- noise generated from PSD
    
    """
    
    strain_noise = pycbc_noise.gaussian.noise_from_psd(length, delta_t, PSD, seed = None) 
    
    return strain_noise


# In this cell we build the injection function: from an input signal waveform measured by the detector and a background noise, the function will inject the signal into the noise, forming what a detector would measure: a strain data formed by noise and a candidate signal to GW. 
# 
# The injection is done randomly, without previously setting the time point when to inject the signal in the noise. However, the injectiong point is saved to keep a record.
# 
# The order of magnitude of the simulated GW signal and the noise are not in real scale. Maybe GW signal is   ~1e-18 while noise is ~1e-21, which is an unreal situation. Therefore the injected GW is rescaled (this can be the weakest point in the hole code).



def injection_by_hand(strain, signal):
    
    """
    This is an injection "by hand", injecting a signal inside a strain of noise
    ---------------------------------------------------------------------------
    
    Arguments:
    strain -- noise or ground data
    signal -- simulated GW signal to inject inside de strain
    
    Returns:
    data -- noise + signal strain   
    sig_loc -- location where the merger is inside de strain
    
    """
    
    if len(strain) < len(signal):
        raise ValueError('Strain data length should be bigger than signal data')
    if strain.delta_t != signal.delta_t:
        raise ValueError('Strain and signal must contain same delta_t')
    
    len_signal = len(signal)
    len_strain = len(strain)
    dif = len_strain - len_signal

    # for a random place inside de strain we will introduce the signal. This place should be such that the signal
    # length can fit inside

    # compute starting time to inject signal by injecting it in a randomly position
    loc = np.random.randint(2 / delta_t, dif - 2 / delta_t)
    
    h = np.copy(strain)
    
    # inject the signal, and multiply by a rescaling factor --> this way the signal isn't too big to be unreal 
    # but not too small to be undetected by the Q transform
    scale = np.mean(abs(np.array(strain))) / np.mean(abs(np.array(signal)))
    
    h[loc:loc+len_signal] = strain[loc:loc+len_signal] + np.array(signal) * scale
    
    sig_loc = (loc + np.argmax(np.array(signal))) * strain.delta_t
    
    data = pycbcty.timeseries.TimeSeries(h, strain.delta_t)
    
    return data, sig_loc


# The following function call for all previous functions to create a bank of images: by specifying the corresponding parameters and its purpose (train or test), the code will run the required functions to generate a signal, a background noise, inject it, save the records of the injection time and parameters used and save in the corresponding folder if train or test.



def bank_generator(mass, inclination, distance, ra, dec, purpose):

    """
    Computes all images for a parameter space
    -----------------------------------------
    
    Arguments:
    mass -- masses of both compact black holes
    inclination -- angle between angular momentum L and line of sight (from 0 to PI)
    distance -- distance in Mpc to the emitting source
    ra dec -- sky location of the source
    purpose -- 'train' or 'test'
    
    Return:
    images in .png format
    record file in .csv format
    
    """
    # starts the records file
    file = ['Ref.Name',  'Mass',  'Distance(Mpc)',  'Inclination (pi units)',  'MergerTime(sec)',  'Ra',  'Dec']
    
    # according to the purpose (train or test) we specificy the path to save the images
    if purpose == 'train':
        dir_name = training_signal_dir
    
    elif purpose == 'test':
        dir_name = test_signal_dir
    
    else:
        raise ValueError('Purpose must be train or test')
    
    for M in tqdm(mass):

        
        it = 1 # keep the record of number of sample inside a collection of images for mass M
        
        for inc in inclination:
            for di in distance:
                for i in range(len(ra)):

                    hp, hc = template(M, inc * np.pi, di, cut_off) # GW wave emitted by source
                    strain = noise_gen(length, delta_t, AdvV_PSD) # noise simulated from detector


                    gw_signal = Virgo_gw(hp, hc, ra[i], dec[i]) # GW projection as measured by detector

                    candidate, sig_loc = injection_by_hand(strain, gw_signal) # injection of the measured GW waveform

                    dic = {'mass':M, 'distance (Mpc)':di, 'inclination':inc, 'merger time (sec)':sig_loc, 'ra':ra[i], 'dec':dec[i]}

                    times, frecs, qplane = candidate.qtransform(.001, logfsteps = 200, qrange = (8, 30), frange = (30, 512))


                    fig = plt.figure(figsize = (10, 5))
                    plt.pcolormesh(times, frecs, qplane**0.5, vmin = 1, vmax = 10, shading = 'auto')
                    plt.ylim(30, 200)
                    plt.xlim(sig_loc-2,sig_loc+2)
                    plt.axis('off')

                    plt.savefig(dir_name + '/mass{}_sample{}.png'.format(M, it), pad_inches=0.0, bbox_inches='tight')
                    plt.close(fig)

                    
                    #####################################################################
                    file_aux = []
                    file_aux.append('mass{}_sample{}'.format(M,it))
                    for j in dic.values():
                        file_aux.append(j)
                    file = np.vstack((file,file_aux))
                    #####################################################################


                    it += 1 
                    
    return file


# The former function was used to create signal images. This one is used to create noise images. Just calls the noise generator and specifies the save folder through the purpose specification.



def just_noise(N, purpose):
    
    """
    This function generates Q transform of noise, creating N images
    ---------------------------------------------------------------
    
    Arguments:
    N -- number of images generated
    purpose -- train or test
    
    Return:
    images -- .png or similar format images
    
    """
    
    # again we specify the path to save images
    if purpose == 'train':
        dir_name = training_noise_dir

    elif purpose == 'test':
        dir_name = test_noise_dir

    else:
        raise ValueError('Purpose must be train or test')

    it = 1 # keeps the count for the number of samples
    for i in tqdm(range(N)):
        strain = noise_gen(length, delta_t, AdvV_PSD) # noise simulated from detector
        times, frecs, qplane = strain.qtransform(.001, logfsteps = 200, qrange = (8, 30), frange = (30, 512))
        fig = plt.figure(figsize = (10, 5))
        plt.pcolormesh(times, frecs, qplane**0.5, vmin=1, vmax=10, shading='auto')
        plt.ylim(30, 200)
        t0 = np.random.randint(2, time_length - 2 )
        plt.xlim(t0-2, t0+2)
        plt.axis('off')


        plt.savefig(dir_name + '/noise_sample{}.png'.format(it), pad_inches=0.0, bbox_inches='tight')
        plt.close(fig)
        
        it += 1
    
    return 


# Following cells calls for all previous functions to create the images and save them in the corresponding folders. First we created the training images by calling bank_generator function to create signal images and then just_noise to create noise images. All this by using the training parameters defined at the very first lines of code. After creating all images, it creates a .csv file containing the records for signal images, including its parameters and injection time.
# 
# Same for test images.
# 
# Before creating them, the code will print a validation statement where prints number of images to create and a required confirmation to go ahead with the process. Both validations for training and test sets are required.



# cell to run code for template bank generator. Need to confirm.
####################################################################################################################
#################################################  TRAIN  ##########################################################
####################################################################################################################


num = len(inclination_tr) * len(distance_tr) * len(ra_tr) * len(mass_tr) # number of images  
print('Se generarán ',num,' TRAINING imágenes de señal GW. También se generará el mismo número de imágenes de ruido.')
#decision = str(input('Quieres seguir con el proceso? [y/n]:'))
decision = 'y' # hacerlo automatico

if decision == 'y':
    rec = bank_generator(mass_tr, inclination_tr, distance_tr, ra_tr, dec_tr, purpose='train')
    df = pd.DataFrame(rec)
    df.columns = ['Ref.Name',  'Mass',  'Distance(Mpc)',  'Inclination (pi units)',  'MergerTime(sec)',  'Ra',  'Dec']
    df.to_csv(training_dir + '/Records_training_signal.csv', sep='\t', float_format='%.4f', header=False, index=False)
    
    just_noise(num, purpose = 'train')
    
    
elif decision == 'n':
    print('Proccess aborted')



####################################################################################################################
#################################################  TEST  ###########################################################
####################################################################################################################

num = len(inclination_te) * len(distance_te) * len(ra_te) * len(mass_te) # number of images  
print('Se generarán ',num,' TEST imágenes de señal GW. También se generará el mismo número de imágenes de ruido.')
#decision = str(input('Quieres seguir con el proceso? [y/n]:'))
decision = 'y' # hacerlo automatico

if decision == 'y':
    rec = bank_generator(mass_te, inclination_te, distance_te, ra_te, dec_te, purpose='test')
    df = pd.DataFrame(rec)
    df.columns = ['Ref.Name',  'Mass',  'Distance(Mpc)',  'Inclination (pi units)',  'MergerTime(sec)',  'Ra',  'Dec']
    df.to_csv(test_dir + '/Records_test_signal.csv', sep='\t', float_format='%.4f', header=False, index=False)
    
    just_noise(num, purpose = 'test')
    
    
elif decision == 'n':
    print('Proccess aborted')


# To end the code, we compress the folder containing all subfolders and images in a .zip file and remove the original folder (not zip) to avoid duplicates.



# creates a zip file containin all images and records, and them removes the directory where images where originally stored, to avoid duplicates
sh.make_archive(base_dir,'zip',root_dir,file_name)
sh.rmtree(base_dir)


# Enlaces de interés:
# 
# https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group___l_a_l_sim_noise_p_s_d__c.html
# 
# http://pycbc.org/pycbc/latest/html/pycbc.psd.html
# 
# http://pycbc.org/pycbc/latest/html/pycbc.noise.html
# 
# http://pycbc.org/pycbc/latest/html/pycbc.html#pycbc.detector.Detector
# 
# http://pycbc.org/pycbc/latest/html/pycbc.waveform.html#pycbc.waveform.waveform.get_td_waveform
# 
# https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group___l_a_l_sim_noise_p_s_d__c.html
# 
