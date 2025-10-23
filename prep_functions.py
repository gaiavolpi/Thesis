import numpy as np
import os
import pycbc.types 
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def get_paths(mass_range, train=True, signal=True, root_dir='/home/alberto_sinigaglia/gaia'):
        """
        Arguments:
        file_name -- name of the file with extention
        mass_range -- to complete path, must contain 'CNN_low_mass', 'CNN_mid_mass_I', 'CNN_mid_mass_II' or 'CNN_high_mass'
        train -- boolean, True for training set, False for test set
        signal -- boolean, True for signal, False for noise
        root_dir -- root directory path

        Return:
        path -- path to the file
        """

        base_dir = os.path.join(root_dir, mass_range)

        if train:
            path = os.path.join(base_dir, 'Training')
            if signal:
                path = os.path.join(path, 'Signal')
            else:
                path = os.path.join(path, 'Noise')
        else:
            path = os.path.join(base_dir, 'Test')
            if signal:
                path = os.path.join(path, 'Signal')
            else:
                path = os.path.join(path, 'Noise')

        paths = sorted(glob(os.path.join(path, '*')))

        return paths

def load_data(path):
    """
    This function loads a single time series from a given path using PyCBC.
    """
    data = pycbc.types.timeseries.load_timeseries(path) #single timeseries
    
    return data

def process_data(data):
    '''
    This function will process the data, whitening, bandpassing and normalising it.
    '''

    length = len(data)
    segment_duration = data.get_duration()
    sample_rate = data.get_sample_rate()
    max_filter_duration = segment_duration / 4
    frec_low_cutoff = 30
    frec_high_cutoff = 200

    # whiten the data
    data = data.whiten(segment_duration, max_filter_duration, remove_corrupted=True, low_frequency_cutoff=frec_low_cutoff, return_psd=False)

    # bandpass
    data = data.lowpass_fir(frec_high_cutoff, 8, beta=5.0, remove_corrupted=True) # bandpassing: supress data for frec>300
    data = data.highpass_fir(frec_low_cutoff, 8, beta=5.0, remove_corrupted=True) # bandpassing: supress data for frec<30

    # normalising data to 1
    # data = data / max(data)
    data = data / max(np.abs(data))

    # append zeros to beginning and end of the data to keep the input shape unchanged after cropping corrupted segments
    length_cr = len(data)
    dif = length-length_cr
    data.prepend_zeros(int(dif / 2)) # append zeros at beginning
    data.append_zeros(int(dif / 2)) # append zeros at end

    # return an array
    data = np.squeeze(np.array(data)) 

    return data

def get_params(path):
    '''
    This function extracts the chirp mass and mass ratio from the filename.
    '''

    csv_dir = os.path.join('/', *path.split(os.sep)[:6])
    name = os.path.splitext(os.path.basename(path))[0]

    if 'Training' in path:
        csv_path = os.path.join(csv_dir, 'Records_training_signal.csv')
    else:
        csv_path = os.path.join(csv_dir, 'Records_test_signal.csv')

    df = pd.read_csv(csv_path, sep='\t')
    row = df.loc[df['Ref.Name'] == name]
    chirp_mass = row['Chirp mass'].values[0]
    mass_ratio = row['Mass ratio'].values[0]

    return chirp_mass, mass_ratio

def load_dataset(mass_range, train=True, root_dir='/home/alberto_sinigaglia/gaia'):
    '''
    This function loads and processes a dataset given a list of paths.
    '''
    X = []
    y = []

    signal_paths = get_paths(mass_range, train, signal=True, root_dir=root_dir)
    noise_paths = get_paths(mass_range, train, signal=False, root_dir=root_dir)

    for idx in tqdm(range(len(signal_paths))):
        sig = process_data(load_data(signal_paths[idx]))
        X.append(sig)
        chirp_mass, mass_ratio = get_params(signal_paths[idx])
        y.append((1, chirp_mass, mass_ratio))
            
        noise = process_data(load_data(noise_paths[idx]))
        X.append(noise)
        y.append((0, None, None)) # label 0 for noise (classification)

    return np.array(X), np.array(y)