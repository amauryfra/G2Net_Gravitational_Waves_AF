# Imports
import os
import shutil
import csv
import h5py
import numpy as np
import pickle
import subprocess
import zipfile
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy import stats
from os.path import isfile, join
from os import listdir
import copy
from os import devnull
import sys


def read_hdf5(filepath) :
    """
    This function reads the training and testing data hdf5 files as numpy arrays.
    Credits : CHAZZER - 'How to read the hdf5 files' notebook
    """
    
    data = []
    with h5py.File(filepath, "r") as f :

        for file_key in f.keys():
            group = f[file_key]

            if isinstance(group, h5py._hl.dataset.Dataset) :
                data.append(np.array(group))
                continue

            for group_key in group.keys() :
                group2 = group[group_key]

                if isinstance(group2, h5py._hl.dataset.Dataset) :
                    data.append(np.array(group2))
                    continue

                for group_key2 in group2.keys() :
                    group3 = group2[group_key2]

                    if isinstance(group3, h5py._hl.dataset.Dataset) :
                        data.append(np.array(group3))
                        continue
    return data



def save_GLN(model, filename) :
    """
    Saves the Gated Linear Network model to disk.
    """
    
    with open(filename, 'wb') as output :
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)



def load_GLN(filename) :
    """
    Loads the Gated Linear Network model from disk.
    """
    
    with open(filename, 'rb') as inpt :
        model = pickle.load(inpt)
    return model



def argmax(lst) :
    """
    Finds the argmax of python list. Works with datetime objects.
    """
    
    return max(enumerate(lst), key = lambda x: x[1])[0]



def load_provided_SFTs(stop_at = None, from_test = False) :
    """
    Loads the provided SFT files and their corresponding labels. Use stop_at
    argument to load a subset of provided SFTs.
    """
    
    # HDF5 files
    if not from_test : SFT_path = 'data.nosync/provided_SFTs'
    else : SFT_path = 'data.nosync/test_SFTs'
    hdf5files = [SFT_path + '/' + f for f in os.listdir(SFT_path) if \
                 os.path.isfile(os.path.join(SFT_path, f))]
    try :
        hdf5files.remove(SFT_path + '/.DS_Store')
        hdf5files.remove(SFT_path + '/.gitignore')
    except :
        pass

    data_list = []
    i = 0
    if stop_at is None : mx = len(hdf5files)
    else : mx = stop_at

    print('')
    print('--Loading provided SFTs--')
    for file in hdf5files :
        i += 1
        print('\r' + format(i/mx * 100, '.2f'), '% loaded', end = '')
        lst = read_hdf5(file)
        data_list += [lst]
        if i == stop_at :
            break
    print('\n--Data loaded--')
    print('')

    # Labels
    if not from_test :
        label_path = 'data.nosync/provided_SFTs_labels.csv'
        label_file = open(label_path)
        csvreader = csv.reader(label_file)
        labels = []
        for index, row in enumerate(csvreader) :
            if index == 0 :
                continue
            labels.append(int(row[1]))
            if index == stop_at :
                break
        labels = np.array(labels)
    else :
        labels = -1
    
    return np.array(data_list, dtype = object), labels



def clean_dir(folder = 'train_SFT.nosync/noise') :
    """
    This function deletes all files inside the specified directory.
    Credits : Nick Stinemates - Stack Overflow
    """
    
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if (os.path.isfile(file_path) or os.path.islink(file_path)) and filename != '.gitignore' :
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason : %s' % (file_path, e))
            
                

def download_test_samples(filename = None, number = None, quiet = False, 
                          path_to_test = 'data.nosync/test_SFTs/') :
    """
    Downloads test samples using the Kaggle API.
    """
    
    testing_files = np.load('data.nosync/test_files_list.npy')
    
    command = ''
    command += 'kaggle competitions download ' 
    command += '-c g2net-detecting-continuous-gravitational-waves ' 
    command += '-p ' + path_to_test + ' '
    if quiet : command += '-q '
    
    
    if filename is not None : 
        print('Downloading test file : ' + filename)
        command += '-f  test/' + filename
        
        process = subprocess.Popen(command.split(), stdout = subprocess.PIPE)
        output, error = process.communicate()
        
        with zipfile.ZipFile(path_to_test + filename + '.zip', 'r') as zip_ref :
            zip_ref.extractall(path_to_test)
        os.remove(path_to_test + filename + '.zip')
        print('')
        return output, error
    
    elif number is not None :
        testing_files = np.load('data.nosync/test_files_list.npy')
        already_there = np.array(['test/' + f for f in listdir('data.nosync/test_SFTs') if \
                       isfile(join('data.nosync/test_SFTs', f))])
        testing_files = np.array([file for file in testing_files if file not in already_there])
        selected_files = np.random.choice(testing_files, size = number, replace = False)
        print('')
        print('Downloading ' + str(number) + ' selected files...')
        print('')
        for filename in selected_files :
            _, _ = download_test_samples(filename = filename[5:], number = None, quiet = quiet)
            
        return True
    
    else : 
        print('')
        print('Provide a filename or a number of files to randomly select.')
        print('')
        return False
    
    
    
def test_record_view(MA_SFT):
    """
    This function builds relevant plots for denoised SFT data.
    Credits : KONSTANTIN DMITRIEV  - 'G2Net: Exporing test & train datasets' notebook
    """
    
    freqs = MA_SFT[4]
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(18,10))
    ax_names = ('Interval No', 'Frequency, Hz')
    for col, detector in enumerate(['L1', 'H1']):
        if detector == 'H1': indx = 0
        else : indx = 2
        axs[0, col].set_title(f'Detector={detector}')
        axs[0, col].set_xlabel(ax_names[0])
        axs[0, col].set_ylabel(ax_names[1])
        axs[0, col].pcolormesh(range(MA_SFT[indx].shape[1]), freqs, MA_SFT[indx])
        for row in range(2):
            x = np.mean(MA_SFT[indx], axis=row)
            k2, p = stats.normaltest(x)
            axs[row+1, col].set_title(f'Detector={detector}; p-value={p:.3}')
            axs[row+1, col].set_xlabel(ax_names[row])
            if row==0:
                axs[row+1, col].plot(x)
            else:
                axs[row+1, col].plot(freqs, x)         
    fig.tight_layout(pad=1.5)
    plt.show()
    


def get_MA_SFT(SFTs, MA = 12) :
    """
    This function performs denoising of the SFT data using a moving average process.
    Credits : KONSTANTIN DMITRIEV  - 'G2Net: Exporing test & train datasets' notebook
    """
    
    MA_SFT = copy.deepcopy(SFTs)
    for detector in [0, 2] :
        sft = SFTs[detector]
        power = np.power(np.abs(sft), 2)
        power_MA_cumsum = np.cumsum(power, axis=1)
        power_MA_cumsum_0 = np.concatenate((np.zeros((power.shape[0],1)), power_MA_cumsum), axis=1)[:,::MA]
        power_MA = np.diff(power_MA_cumsum_0, axis=1)[:,:-1]/MA
        MA_SFT[detector] = power_MA
    return MA_SFT



def denoise_SFT(SFT, MA = 12) :
    """
    This function performs denoising of the SFT data using a moving average process.
    Credits : KONSTANTIN DMITRIEV  - 'G2Net: Exporing test & train datasets' notebook
    Adapted to unique SFT.
    """
    
    power = SFT.real**2 + SFT.imag**2
    power_MA_cumsum = np.cumsum(power, axis = 1)
    power_MA_cumsum_0 = np.concatenate((np.zeros((power.shape[0],1)), power_MA_cumsum), axis=1)[:,::MA]
    power_MA = np.diff(power_MA_cumsum_0, axis=1)[:,:-1]/MA
    return power_MA



def to_2_channels(SFT) :
    """
    This function transforms a complex SFT into a 2-channels image.
    """
    
    img = np.array([np.real(SFT), np.imag(SFT)])
    return img.reshape(img.shape[1], img.shape[2], img.shape[0])



def to_2_channels_vec(SFT_bank) :
    """
    This function transforms a set of complex SFTs into a set of 2-channels 
    images.
    """
    
    img_bank = np.array([np.array([np.real(SFT), np.imag(SFT)]) for SFT in SFT_bank])
    return img_bank.reshape(img_bank.shape[0], 
                          img_bank.shape[2], 
                          img_bank.shape[3], 
                          img_bank.shape[1])



def to_complex(img) :
    """
    This function transforms a 2-channels image into a complex SFT.
    """
    
    return img[:, :, 0] + 1j * img[:, :, 1]



def to_complex_vec(img_bank) :
    """
    This function transforms a set of 2-channels images into a set of complex SFTs.
    """
    
    SFT_bank = np.array([img[:, :, 0] + 1j * img[:, :, 1] for img in img_bank])
    return SFT_bank



def blockPrint() :
    """
    This functions blocks prints.
    Credits : Brigand - Stack Overflow
    """
    
    sys.stdout = open(devnull, 'w')



def enablePrint() :
    """
    This functions enables prints.
    Credits : Brigand - Stack Overflow
    """
    
    sys.stdout = sys.__stdout__
    
    
    
def files_in_dir(input_dir, remove_labels = True, sort = True) :
    """
    This function lists all files in the input directory.
    Credits : OpenAI API.
    """
    
    # List to store the files in the input directory 
    files_list = [] 
      
    # Traverse root directory, and list directories as dirs and files as files 
    for root, dirs, files in os.walk(input_dir): 
        for filename in files: 
            # Add file to list 
            files_list.append(os.path.join(root, filename)) 

    try :
        files_list.remove(input_dir + '/.DS_Store')
    except :
        pass
        
    try :
        files_list.remove(input_dir + '/.gitignore')
    except :
        pass
    
    try :
        files_list.remove(input_dir + '/labels/.gitignore')
    except :
        pass
    
    if remove_labels :
        try :
            files_list.remove(input_dir + '/labels/labels.npy')
        except :
            pass

    if sort : return sorted(files_list, key = lambda x: int(x.split('_')[0].split('/')[-1]))
    
    return files_list
    
    
    
###############################################################################
# The following plotting functions have been copied from 
# 'PyFstat Tutorial adapted to kaggle'tutorial from GEORGE CHIRITA


def plot_real_imag_spectrograms(timestamps, frequency, fourier_data) :
    fig, axs = plt.subplots(1, 2, figsize=(16, 10))

    for ax in axs:
        ax.set(xlabel="SFT index", ylabel="Frequency [Hz]")

    time_in_days = (timestamps - timestamps[0]) / 1800

    axs[0].set_title("SFT Real part")
    c = axs[0].pcolormesh(
        time_in_days,
        frequency,
        fourier_data.real,
        norm=colors.CenteredNorm(),
    )
    
    fig.colorbar(c, ax=axs[0], orientation="horizontal", label="Fourier Amplitude")

    axs[1].set_title("SFT Imaginary part")
    c = axs[1].pcolormesh(
        time_in_days,
        frequency,
        fourier_data.imag,
        norm=colors.CenteredNorm(),
    )

    fig.colorbar(c, ax=axs[1], orientation="horizontal", label="Fourier Amplitude")
    
    return fig, axs



def plot_real_imag_spectrograms_with_gaps(timestamps, frequency, fourier_data, Tsft) :

    # Fill up gaps with Nans
    gap_length = timestamps[1:] - (timestamps[:-1] + Tsft)

    gap_data = [fourier_data[:, 0]]
    gap_timestamps = [timestamps[0]]

    for ind, gap in enumerate(gap_length):
        if gap > 0:
            gap_data.append(np.full_like(fourier_data[:, ind], np.nan + 1j * np.nan))
            gap_timestamps.append(timestamps[ind] + Tsft)

        gap_data.append(fourier_data[:, ind + 1])
        gap_timestamps.append(timestamps[ind + 1])

    return plot_real_imag_spectrograms(
        np.hstack(gap_timestamps), frequency, np.vstack(gap_data).T
    )



def plot_real_imag_histogram(fourier_data, theoretical_stdev = None) :

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set(xlabel="SFT value", ylabel="PDF", yscale="log")

    ax.hist(
        fourier_data.real.ravel(),
        density=True,
        bins="auto",
        histtype="step",
        lw=2,
        label="Real part",
    )
    ax.hist(
        fourier_data.imag.ravel(),
        density=True,
        bins="auto",
        histtype="step",
        lw=2,
        label="Imaginary part",
    )

    if theoretical_stdev is not None:
        x = np.linspace(-4 * theoretical_stdev, 4 * theoretical_stdev, 1000)
        y = stats.norm(scale=theoretical_stdev).pdf(x)
        ax.plot(x, y, color="black", ls="--", label="Gaussian distribution")

    ax.legend()

    return fig, ax



def plot_amplitude_phase_spectrograms(timestamps, frequency, fourier_data) :
    fig, axs = plt.subplots(1, 2, figsize=(16, 10))

    for ax in axs:
        ax.set(xlabel="SFT index", ylabel="Frequency [Hz]")

    time_in_days = (timestamps - timestamps[0]) / 1800

    axs[0].set_title("SFT absolute value")
    c = axs[0].pcolorfast(
        time_in_days, frequency, np.absolute(fourier_data), norm=colors.Normalize()
    )
    fig.colorbar(c, ax=axs[0], orientation="horizontal", label="Value")

    axs[1].set_title("SFT phase")
    c = axs[1].pcolorfast(
        time_in_days, frequency, np.angle(fourier_data), norm=colors.CenteredNorm()
    )

    fig.colorbar(c, ax=axs[1], orientation="horizontal", label="Value")

    return fig, axs
    
    
    
    