from sklearn.model_selection import train_test_split
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib


def split_train_val(X, y, val_size=0.2):
    '''
    This function splits the dataset into training and validation sets.
    '''
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=42, stratify=y)
    return X_train, X_val, y_train, y_val

def make_dataset(X, y, batch_size, shuffle=True):
    '''
    This function creates a TensorFlow dataset from numpy arrays.
    '''
    X = X.astype('float32')
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    y = y.astype('float32')
    y = np.reshape(y, (y.shape[0], 1))

    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
        
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset

def plot_tp_fp(y_test_pred, title, save_path=None, bins=100, figsize=(10,8)):
    
    fig = plt.figure(figsize=figsize)
    
    # Plot cumulative histograms
    plt.hist(y_test_pred[0::2], bins=bins, histtype='step', color='red', 
             cumulative=-1, linewidth=2.5, label="True Positive Test")
    plt.hist(y_test_pred[1::2], bins=bins, histtype='step', color='blue', 
             cumulative=-1, linewidth=2.5, label="False Positive Test")
    
    plt.ylabel('Cumulative Tests', fontsize=25)
    plt.xlabel('Threshold', fontsize=25)
    plt.yscale("log")
    plt.title(title, fontsize=25, y=1.0)
    plt.grid(linewidth=1, color='black', alpha=0.2)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    # Custom legend
    labels = ['True Positive Test', 'False Positive Test']
    handle1 = matplotlib.lines.Line2D([], [], c='r')
    handle2 = matplotlib.lines.Line2D([], [], c='b')
    leg = plt.legend(handles=[handle1, handle2], labels=labels, 
                     loc='lower center', prop={'size':20})

    for line in leg.get_lines():
        line.set_linewidth(3.0)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=800, pad_inches=0.1, bbox_inches='tight')
    
    plt.show()
    plt.close(fig)

import matplotlib.pyplot as plt
import matplotlib

def plot_signal_noise(y_test_pred, title, save_path=None, bins=100, figsize=(10,8)):

    fig = plt.figure(figsize=figsize)

    # Plot histograms
    plt.hist(y_test_pred[1::2], bins=bins, histtype='step', color='orange',
             linewidth=2.5, label="Noise", hatch='/', facecolor='orange',
             alpha=0.4, fill=True)
    plt.hist(y_test_pred[0::2], bins=bins, histtype='step', color='blue',
             linewidth=2.5, label="Signal", hatch='/', facecolor='blue',
             alpha=0.2, fill=True)

    # Labels and styling
    plt.ylabel('Tests', fontsize=25)
    plt.xlabel('Score', fontsize=25)
    plt.yscale("log")
    plt.title(title, fontsize=25, y=1.0)
    plt.grid(linewidth=1, color='black', alpha=0.2)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Custom legend
    labels = ['Noise', 'Signal']
    handle1 = matplotlib.lines.Line2D([], [], c='orange')
    handle2 = matplotlib.lines.Line2D([], [], c='blue')
    leg = plt.legend(handles=[handle1, handle2], labels=labels, 
                     loc='upper center', prop={'size':20})

    for line in leg.get_lines():
        line.set_linewidth(3.0)

    plt.tight_layout()

    # Save if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=800, pad_inches=0.1, bbox_inches='tight')

    plt.show()
    plt.close(fig)
