import matplotlib.pyplot as plt
import numpy as np
import regressor_utils_classi
import tensorflow as tf
from fig_config import figure_features
import os
import pandas as pd

########  TEST OUR MODEL  #########

# Define corresponding paths to temporal series files:

file_name = 'CNN_Images_Run35_m1_m2_high_mass'

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


# We opne .csv file and append extentions .png and .hdf where needed.

# for SERIES
record_path_train = os.path.join(series_tr_dir,'Records_training_signal.csv')
record_train_series = pd.read_csv(record_path_train, sep='\t')

record_path_test = os.path.join(series_te_dir,'Records_test_signal.csv')
record_test_series = pd.read_csv(record_path_test, sep='\t')


def append_ext_png(fn):
    return fn+".png"

def append_ext_hdf(fn):
    return fn+".hdf"

# for SERIES
record_test_series["Ref.Name"]=record_test_series["Ref.Name"].apply(append_ext_hdf)


# test set of series 

data_generator_test = regressor_utils_classi.GW_data_generator(record_test_series)
test_idx = data_generator_test.generate_indexes()
test_X_series, _ = data_generator_test.generate_series(test_idx, purpose='test', processing=True)



dirs = '/data/danibelt/CNN/Classifier/High_mass/Run 10 - 3 ResNet, Last mod/'

classi = ['Classifier_epochs50','Classifier_epochs90','Classifier_epochs120','Classifier_epochs170','Classifier_epochs200','Classifier_epochs250','Classifier_epochs300']
nepochs = [50,90,120,170,200,250,300]

for i in range(len(classi)):

    model = tf.keras.models.load_model(dirs + classi[i])

    test_Y_pred = model.predict(test_X_series)

    # por la forma de haber construido la base de datos, las series impares corresponden a ruido

    # True Positive Test -- plot noise series

    figure_features()
    fig = plt.figure(figsize=(15,10))
    plt.hist(test_Y_pred[0::2],bins=100,histtype='step',color='darkorange', cumulative=-1, label= "Model {} epochs".format(nepochs[i]))
    plt.ylabel('Tests')
    plt.xlabel('Score')
    plt.yscale("log")
    plt.title("True Positive Test")
    plt.legend()
    plt.savefig(dirs + '/Images/M_TPTest_epochs{}.png'.format(nepochs[i]))
    #plt.show()
    plt.close(fig)

