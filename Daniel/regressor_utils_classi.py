import os 
import zipfile
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from fig_config import figure_features
import tqdm

import pycbc.types 

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 


# Define corresponding paths to temporal series files:

file_name = 'CNN_Images_Run35_m1_m2_low_mass'

"""
CNN_Images_Run35_m1_m2_low_mass
CNN_Images_Run35_m1_m2_mid_mass_I
CNN_Images_Run35_m1_m2_mid_mass_II
CNN_Images_Run35_m1_m2_high_mass
"""

root_dir = '/data/danibelt/Images_Series_Chirp_Ratio_1/'
#root_dir = '/run/user/1001/gvfs/sftp:host=pcaecuda2,user=danibelt/data/danibelt/Images_Series_Chirp_Ratio_1/'


base_dir = os.path.join(root_dir, file_name)

# training and test path (2)
training_dir = os.path.join(base_dir, 'Training')
test_dir = os.path.join(base_dir, 'Test')

# series and images paths, two per previous path (4)
image_tr_dir = os.path.join(training_dir, 'Images')
image_te_dir = os.path.join(test_dir, 'Images')
series_tr_dir = os.path.join(training_dir, 'Series')
series_te_dir = os.path.join(test_dir, 'Series')

# signal and noise paths, two per previous path (8)
training_im_signal_dir = os.path.join(image_tr_dir, 'Signal')
training_im_noise_dir = os.path.join(image_tr_dir, 'Noise')

training_se_signal_dir = os.path.join(series_tr_dir, 'Signal')
training_se_noise_dir = os.path.join(series_tr_dir, 'Noise')

test_im_signal_dir = os.path.join(image_te_dir, 'Signal')
test_im_noise_dir = os.path.join(image_te_dir, 'Noise')

test_se_signal_dir = os.path.join(series_te_dir, 'Signal')
test_se_noise_dir = os.path.join(series_te_dir, 'Noise')

"""
# Unzip folder containing files to access them:

local_zip = base_dir + '.zip'  
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall(root_dir)
zip_ref.close()
"""

# See how many images and series of each category we have :

print('total training signal series:', len(os.listdir(training_se_signal_dir)))
print('total training noise series:', len(os.listdir(training_se_noise_dir)))
print('total test signal series:', len(os.listdir(test_se_signal_dir)))
print('total test noise series:', len(os.listdir(test_se_noise_dir)))

# We opne .csv file and append extentions .png and .hdf where needed.

# for SERIES
record_path_train = os.path.join(series_tr_dir,'Records_training_signal.csv')
record_train_series = pd.read_csv(record_path_train, sep='\t')

record_path_test = os.path.join(series_te_dir,'Records_test_signal.csv')
record_test_series = pd.read_csv(record_path_test, sep='\t')

# for IMAGES
#record_path_train = os.path.join(image_tr_dir,'Records_training_signal.csv')
#record_train_images = pd.read_csv(record_path_train, sep='\t')

#record_path_test = os.path.join(image_te_dir,'Records_test_signal.csv')
#record_test_images = pd.read_csv(record_path_test, sep='\t')


def append_ext_png(fn):
    return fn+".png"

def append_ext_hdf(fn):
    return fn+".hdf"

# for SERIES
record_train_series["Ref.Name"]=record_train_series["Ref.Name"].apply(append_ext_hdf)
record_test_series["Ref.Name"]=record_test_series["Ref.Name"].apply(append_ext_hdf)

# for IMAGES
#record_train_images["Ref.Name"]=record_train_images["Ref.Name"].apply(append_ext_png)
#record_test_images["Ref.Name"]=record_test_images["Ref.Name"].apply(append_ext_png)





# Definimos unas funciones para cargar los datos, procesarlos y prepararlos como input a la CNN:

def load_data(name, purpose):

        """
        # Loads the data for a given path, yielding a type pycbc.timeseries.TimeSeries
        # ----------------------------------------------------------------------------

        # Arguments:
        # name -- name of the file with extention
        # purpose -- to complete path, must contain 'train' or 'test' and 'signal' or 'noise'

        # Return:
        # data -- dictionary with data, time duration, and frecuency sample
        
        """

               
        if purpose[0] == 'train':
            if purpose[1] == 'signal':
                aux_path = training_se_signal_dir
            if purpose[1] == 'noise':
                aux_path = training_se_noise_dir

        elif purpose[0] == 'test':
            if purpose[1] == 'signal':
                aux_path = test_se_signal_dir
            if purpose[1] == 'noise':
                aux_path = test_se_noise_dir
       
        data = pycbc.types.timeseries.load_timeseries(aux_path + '/' + name)
        

        return data
    



##############################################   RETURN   ##############################################################################
class GW_data_generator():

    
    # This class will generate an array of temporal series, each row of the array being a GW series, and also will generate its corresponding
    # result (0 for noise, 1 fopr signal), and its chirp mass and mass ratio.
    # ----------------------------------------------------------------------------------------------------------------------------------------
    
    
    def __init__(self, record):
        self.record = record
        


    def generate_indexes(self):
        """
        This function will generate a list of numbers shuffled to identify the series
        """
        idx_list = np.random.permutation(len(self.record)) # we shuffle the training samples

        return idx_list

    
    def process_data(self, name, purpose):

        
        # Process an input data given by returning information about it such as duration and sample rate, 
        # and whitening data, bandpassing, normalising and returning an array
        # -----------------------------------------------------------------------------------------------

        # Arguments:
        # name -- name of the file with extention
        # purpose -- to complete path, must contain 'train' or 'test' and 'signal' or 'noise'

        # Return:
        # dictionary type with duration, sample rate, and array in numpy type and TimeSeries type

        
        data = load_data(name, purpose) 
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
        data = data / max(data)

        # append zeros to beginning and end of the data to keep the input shape unchanged after cropping corrupted segments
        
        length_cr = len(data)
        dif = length-length_cr
        data.prepend_zeros(int(dif / 2)) # append zeros at beginning
        data.append_zeros(int(dif / 2)) # append zeros at end
        

        # return an array
        data = np.squeeze(np.array(data))

        return data

    def generate_series(self, series_idx, purpose, processing):

        
        # Used to generate a batch with series when training/validating/testing our model.
        # -------------------------------------------------------------------------------

        # Arguments:
        # series_idx -- series identification number iloc: if 1050 images, then it takes num in [0,1050]
        # is_training = True
        # purpose -- neccesary to specify directory path
        # batch_size -- len of the batch to b considered

        # Returns:
        # series array containing the temporal series
        # result -- array with corresponding 0 for noise and 1 for signal
        # chirp mass -- array containing corresponding chirp mass of signal series
        # mass ratio -- array containing corresponding mass ratio of signal series

        # Raise:
        # ValueError: Batch size must be multiple of 2 to contain same number of noise than signal series in each batch

        
        # arrays to store our batched data
        series, result = np.empty((1,int(3*2048)),dtype=float), np.array([])
        #chirpM, qratio = [], []
        
        for idx in (series_idx): #tqdm.tqdm

            ######   WE APPEND A SIGNAL SERIES  ########
            aux_purpose = 'signal'
            parameters = self.record.iloc[idx] # returns the row of data corresponding to such series iloc
            file_name = parameters['Ref.Name']    

            # here we load the data and process it according to signal or noise

            # with processing
            if processing == True:    
                series_np = self.process_data(file_name, [purpose, aux_purpose])
            # with no processing
            if processing == False:
                series_np = np.array(load_data(file_name, [purpose, aux_purpose]))

            
            result = np.hstack((result, 1))
            """
            chirp = parameters['Chirp mass']
            qM = parameters['Mass ratio']

            chirpM.append(chirp)
            qratio.append(qM)
            """
            series = np.vstack((series, series_np))


            # once appended a signal value, now we append a noise value

            #######  WE APPEND A NOISE SERIES  ##########
            aux_purpose = 'noise'
            if purpose == 'train':
                aux_path = training_se_noise_dir
            elif purpose == 'test':
                aux_path = 'test_se_noise_dir'
            #it = np.random.randint(1, len(os.listdir(aux_path)))
            file_name = 'noise_sample{}.hdf'.format(idx+1) # csv indexes gows from 0 to ..., but noise samples goes from 1 to ...

            # with processing
            if processing == True:    
                series_np = self.process_data(file_name, [purpose, aux_purpose])
                                
            # with no processing
            if processing == False:
                #input_shape = series_np.shape()
                series_np = np.array(load_data(file_name, [purpose, aux_purpose]))
                

            result = np.hstack((result, 0))
            series = np.vstack((series, series_np))

            
        series = np.delete(series, 0, axis=0) # delete first row of np.empty
        return series, result
 