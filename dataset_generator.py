# Imports
import os
from generator import Generator
import numpy as np
from progress.bar import FillingCirclesBar
from utils import to_2_channels
from skimage.transform import resize
import h5py
import shortuuid


def generate_dataset_avg(number, path_to_datasets = 'CNN/datasets.nosync/', 
                     number_samples = 500, average_size = 4585, std_size = 80,
                     save_to_disk = True) :
    """
    This function generates a medium size dataset for the training of the CNN.
    The images are resized to the test set average size by interpolation.
    """
    
    print('')
    print('')
    bar = FillingCirclesBar(' Generating dataset number ' + str(number) + ' ',\
                            max = int(number_samples / 4), \
                            suffix = '%(percent).1f%% - %(eta)ds')
    # Creating directory
    path_to_this_dataset = path_to_datasets + 'dataset_' + str(number) 
    if not os.path.exists(path_to_this_dataset) :
        os.makedirs(path_to_this_dataset)
    
    # Memory allocation
    dataset_samples = np.zeros((number_samples, 360, average_size, 2))
    dataset_labels = np.zeros(number_samples, dtype = int)
    
    for sample in range(0, number_samples, 2) :
        # Generating sample without prints
        diff = np.inf
        while diff > 4 * std_size :
            SFTs, with_signal = Generator(enable_prints = False).generate_sample()
            obtained_size = np.array([SFTs[0].shape[1], SFTs[2].shape[1]])
            diff = np.max(np.abs(obtained_size - average_size))
        
        # Adding resized samples as 2-channels images
        dataset_samples[sample] = resize(to_2_channels(SFTs[0]), 
                                         (360, average_size, 2), anti_aliasing = True)
        dataset_labels[sample] = int(with_signal)
        dataset_samples[sample+1] = resize(to_2_channels(SFTs[2]), 
                                         (360, average_size, 2), anti_aliasing = True)
        dataset_labels[sample+1] = int(with_signal)
        bar.next()
        
    bar.finish()
    
    if save_to_disk :
        h5f = h5py.File(path_to_this_dataset + '/SFTs_' + str(number) + '.h5', 'w')
        h5f.create_dataset('dataset_' + str(number), data = dataset_samples)
        h5f.close()
        np.save(path_to_this_dataset + '/labels_' + str(number) + '.npy', dataset_labels)
        print('')
        print('Dataset saved to disk')
        print('')
        
        return True, True
    
    else : return dataset_samples, dataset_labels



def crop_center(arr, minimum_size) :
    """
    This helper function crops the SFT array to the minimum size while maintaining 
    it centered.
    """
    
    diff = arr.shape[1] - minimum_size
    if diff%2 == 0 :
        start = diff//2
        end = diff//2
    else :
        start = diff//2 + 1
        end = diff//2
    return arr[:, start:-end]
    


def generate_dataset_min(purpose = 'test', path_to_datasets = 'CNN/datasets.nosync/', 
                     number_samples = 500, minimum_size = 4281) :
    """
    This function generates a medium size dataset for the training of the CNN.
    The images are cropped to the minimum size of the test set.
    """
    
    print('')
    print('')
    bar = FillingCirclesBar(' Generating CNN ' + purpose + ' dataset' ,\
                            max = int(number_samples / 2), \
                            suffix = '%(percent).1f%% - %(eta)ds')
        

    # Creating directory
    path_to_this_dataset = path_to_datasets + purpose
    if not os.path.exists(path_to_this_dataset) :
        os.makedirs(path_to_this_dataset)
    path_to_this_dataset_labels = path_to_datasets + purpose + '/labels'
    if not os.path.exists(path_to_this_dataset_labels) :
        os.makedirs(path_to_this_dataset_labels)
    
    # Memory allocation
    dataset_labels = np.zeros(number_samples, dtype = object)
    
    with_signal = False
    for sample in range(0, number_samples, 2) :
        # Alternating 1/2 signal injection
        with_signal = not with_signal
        # Generating sample without prints
        min_shape = minimum_size - 1
        while min_shape < minimum_size :
            SFTs, with_signal = \
                Generator(enable_prints = False).generate_sample(with_signal = with_signal)
            obtained_sizes = np.array([SFTs[0].shape[1], SFTs[2].shape[1]])
            min_shape = np.min(obtained_sizes)
        
        # Adding resized samples as 2-channels images
        uuid_1 = shortuuid.uuid()
        filename_1 = str(sample) + '_' + str(int(with_signal)) + '_' + uuid_1 + '.npy'
        np.save(path_to_this_dataset + '/' + filename_1, 
                to_2_channels(crop_center(SFTs[0], minimum_size)))
        dataset_labels[sample] = (int(with_signal), uuid_1)
        
        uuid_2 = shortuuid.uuid()
        filename_2 = str(sample+1) + '_' + str(int(with_signal)) + '_' + uuid_2 + '.npy'
        np.save(path_to_this_dataset + '/' + filename_2, 
                to_2_channels(crop_center(SFTs[2], minimum_size)))
        dataset_labels[sample+1] = (int(with_signal), uuid_2)
        bar.next()
        
    bar.finish()
    

    np.save(path_to_this_dataset + '/labels/labels' + '.npy', dataset_labels)
    print('')
    print('Dataset saved to disk')
    print('')
        
    return True
    
  
    
if __name__ == '__main__' :
    generate_dataset_min()


