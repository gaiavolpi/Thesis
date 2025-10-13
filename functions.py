import pycbc.waveform as pycbc_wf
import pycbc.psd as pycbc_psd
import pycbc.noise as pycbc_noise
import pycbc.types as pycbcty
import pycbc.detector as pycbc_det
import pycbc.catalog as pycbc_cat
import pycbc.filter as pycbc_fil
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
import shutil as sh
import os as os
import astropy.cosmology as astr_cosm
import warnings

delta_t = 1.0 / 2048
time_length = 3 #  seconds of data, since longest GW signal lasts 2 sec (m1=m2=10) in this sense we reduce computational resources
length = int(time_length / delta_t)  # seconds of noise data with 2048 data per second  samples

# first we import Adv Virgo PSD from LAL Simulations (https://pycbc.org/pycbc/latest/html/pycbc.psd.html)

delta_f = 1.0 / time_length  
flength = int(2048 / delta_f)  
low_freq_cutoff = 30  # corresponding the cutoff of our GW model

# from our simulated PSD now we obtain our gausian noise (https://pycbc.org/pycbc/latest/html/pycbc.noise.html#module-pycbc.noise.gaussian)

Virgo = pycbc_det.Detector('V1')
AdvV_PSD = pycbc_psd.analytical.AdvVirgo(flength, delta_f, low_freq_cutoff)

def chirp(m1,m2):
    
    """
    This function returns the chirp mass of a binarys
    ----------------------------------------------------------------------------------------------
    
    Arguments:
    m1 and m2 -- individual mass

    Return:
    M -- chirp mass

    """
    x = ((m1*m2)**(3/5))/((m1+m2)**(1/5))
    return x


def q(m1,m2):
    
    """
    This function computes the mass ratio between the masses, where this quantity must be less than 1
    -----------------------------------------------------------------------------------------------
   
    Arguments:
    m1 and m2 -- individual mass

    Return:
    q -- mass ratio

    """
    if m1>m2:
        return m2/m1
    elif m1<m2:
        return m1/m2
    
def template(m1, m2, incl, dist, cut_off):

    """
    Builds a template por a GW merger of both masses 'm' using SEOBNRv4_opt approximation method with a
    sample of 2048 data per second (2048Hz)
    ---------------------------------------------------------------------------------------------------

    Arguments:
    m1 -- mass of the first compact black hole
    m2 -- mass of the second compact black hole
    incl -- angle between angular momentum L and line of sight (from 0 to PI)
    dist -- distance in Mpc to the emitting source
    cut_off -- low frecuency limit

    Return:
    hp, hc -- pycbc.timeseries.TimeSeries, for the plus/cross polarization GW (strain)
    """
    hp, hc = pycbc_wf.get_td_waveform(approximant = "SEOBNRv4_opt", mass1 = m1, mass2 = m2, inclination = incl, distance = dist, delta_t = 1.0 / 2048, f_lower = cut_off)

    return hp, hc

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
    
    # strain and noise frec 2048 Hz
    len_signal = len(signal)
    len_strain = len(strain)
    dif = len_strain - len_signal

    # for a random place inside de strain we will introduce the signal. This place should be such that the signal
    # length can fit inside

    # compute starting time to inject signal by injecting it in a randomly position
    
    # since when whitening data we need to crop part of the beginning and end, we make
    # sure we dont inject just at the beginning or end of the strain. Approx crop is 
    # 0.5sec from each point
    loc = np.random.randint(0.5 * 2048, dif - 0.5 * 2048)

    h = np.copy(strain)
    
    # inject the signal, and multiply by a rescaling factor --> this way the signal isn't too big to be unreal 
    # but not too small to be undetected by the Q transform
    
    scale = np.mean(abs(np.array(strain))) / np.mean(abs(np.array(signal))) * (1/7)
    
    h[loc:loc+len_signal] = strain[loc:loc+len_signal] + np.array(signal) * scale
    
    sig_loc = (loc + np.argmax(np.array(signal))) * strain.delta_t
    
    data = pycbcty.timeseries.TimeSeries(h, strain.delta_t)
    
    return data, sig_loc


def SNR(model, noise):
    """
    Computes the Signal-to-Noise Ratio for a given data expressed as sum of a noise and model

    Arguments:
    model -- GW template
    noise -- background noise in the detector


    Return:
    SNR -- Signal to Noise ratio
    """

    candidate, _ = injection_by_hand(noise, model)

    model_copy = model.copy()
    model_copy.resize(len(candidate))

    snr = pycbc_fil.matchedfilter.matched_filter(model_copy, candidate, psd=AdvV_PSD, low_frequency_cutoff=30.0)

    return snr