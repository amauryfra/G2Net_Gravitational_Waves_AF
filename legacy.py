# Imports
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from sklearn.model_selection import train_test_split
from utils import to_complex
from utils import denoise_SFT
from utils import clean_dir
import h5py
import gc
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2M
import os
import pickle
from dataset_generator import generate_dataset_min
from train_CNN import ClearingCallback
from generator import Generator
from dataset_generator import crop_center
from utils import to_2_channels
from progress.bar import FillingCirclesBar
from detector import Detector 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from utils import files_in_dir


def _generate_dataset_min(number = 1, path_to_datasets = 'CNN/datasets.nosync/', 
                     number_samples = 1000, minimum_size = 4281,
                     save_to_disk = True) :
    """
    This function generates a medium size dataset for the training of the CNN.
    The images are cropped to the minimum size of the test set. Not used anymore.
    """
    
    print('')
    print('')
    bar = FillingCirclesBar(' Generating dataset number ' + str(number) + ' ',\
                            max = int(number_samples / 2), \
                            suffix = '%(percent).1f%% - %(eta)ds')
        
    if save_to_disk :
        # Creating directory
        path_to_this_dataset = path_to_datasets + 'dataset_' + str(number) 
        if not os.path.exists(path_to_this_dataset) :
            os.makedirs(path_to_this_dataset)
    
    # Memory allocation
    dataset_samples = np.zeros((number_samples, 360, minimum_size, 2))
    dataset_labels = np.zeros(number_samples, dtype = int)
    
    with_signal = False
    for sample in range(0, number_samples, 2) :
        # Alternating 1/2 signal injection
        with_signal = not with_signal
        # Generating sample without prints
        min_shape = minimum_size - 1
        while min_shape < minimum_size :
            SFTs, with_signal = Generator(enable_prints = False).generate_sample(with_signal = with_signal)
            obtained_sizes = np.array([SFTs[0].shape[1], SFTs[2].shape[1]])
            min_shape = np.min(obtained_sizes)
        
        # Adding resized samples as 2-channels images
        dataset_samples[sample] = to_2_channels(crop_center(SFTs[0], minimum_size))
        dataset_labels[sample] = int(with_signal)
        dataset_samples[sample+1] = to_2_channels(crop_center(SFTs[2], minimum_size))
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
    
    
    
def _load_dataset(dataset_id, path_to_datasets = 'CNN/datasets.nosync/',
                 enable_prints = True) :
    """
    This function loads the dataset in memory for the training of the CNN.
    """
    
    # SFTs
    if enable_prints : print('')
    if enable_prints : print('Loading dataset ' + str(dataset_id) + '...')
    h5f = h5py.File('CNN/datasets.nosync/dataset_' + str(dataset_id) + '/SFTs_' + str(dataset_id) + '.h5', 'r')
    SFT_dataset = h5f['dataset_' + str(dataset_id)][:]
    h5f.close()
    # Labels
    labels = np.load('CNN/datasets.nosync/dataset_' + str(dataset_id) + '/labels_' + str(dataset_id) + '.npy')
    if enable_prints : print('Dataset loaded in memory')
    if enable_prints : print('')
    
    return SFT_dataset, labels



def _process_dataset(SFT_dataset, type_, 
                    enable_prints = True, del_previous = False) :
    """
    This function applies the moving average denoising procedure to the SFTs 
    in dataset.
    """
    
    if enable_prints :
        mx = SFT_dataset.shape[0]
        print('Processing loaded SFTs')
        
    if type_ == 2 :
        processed_dataset = np.zeros((SFT_dataset.shape[0], SFT_dataset.shape[1], 
                                      denoise_SFT(to_complex(SFT_dataset[0])).shape[1]))
        for indx, img in enumerate(SFT_dataset) :
            fourier_data = to_complex(img)
            processed_dataset[indx] = denoise_SFT(fourier_data)
            if enable_prints : print('\r' + format((indx+1)/mx * 100, '.2f'), '% processed', end = '')
            
    if type_ == 3 :
        processed_dataset = np.zeros((SFT_dataset.shape[0], SFT_dataset.shape[1], SFT_dataset.shape[2]))
        for indx, img in enumerate(SFT_dataset) :
            fourier_data = to_complex(img)
            processed_dataset[indx] = np.square(np.abs(fourier_data))
            if enable_prints : print('\r' + format((indx+1)/mx * 100, '.2f'), '% processed', end = '')
        
    if enable_prints :
        print('')
        print('SFTs processed')
        print('')
        
    if del_previous : del SFT_dataset
    return processed_dataset



def _train(dataset_id, type_, batch_size = 32, number_epochs = 3,
          learning_rate = 1e-4, SFT_dataset = None, labels = None) :
    """
    This function creates a CNN, loads existing weights when applicable and 
    performs training. Not used anymore.
    """
    
    # Loading data into memory
    if SFT_dataset is None :
        SFT_dataset, labels = _load_dataset(dataset_id)


    if type_ > 1 :
        SFT_dataset = _process_dataset(SFT_dataset, type_)
    # Split train and test sets
    X_train, X_test, y_train, y_test = train_test_split(SFT_dataset, labels, 
                                                        test_size = 0.20)
    
    if type_ > 1 :
        X_train = np.expand_dims(X_train, -1)
        X_test = np.expand_dims(X_test, -1)
    y_train = np.asarray(y_train).astype('float64').reshape((-1,1))
    y_test = np.asarray(y_test).astype('float64').reshape((-1,1))
    
    gc.collect()
    
    # Input shape
    img_inputs = Input(shape = X_train.shape[1:], dtype = X_train.dtype)
    
    print('')
    print('Starting training')
    print('')
    
    # Saving model weights along the way
    file_path = './CNN/model.nosync/weights_type=' + str(type_) + '.hdf5'
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = file_path,
                                                     monitor = 'val_loss',
                                                     save_best_only = True,
                                                     save_weights_only = True,
                                                     verbose = 1)
    
    # Perform early stopping
    es_callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 3)
    
    if os.path.isfile(file_path) : 
        weights = file_path
        print('Loading weights...')
    else : weights = None
    
    # Clearing callback
    clearing_callback = ClearingCallback()
    
    # Using EfficientNetV2M
    model = EfficientNetV2M(input_tensor = img_inputs, weights = weights, classes = 1)
    if weights is not None : 
        print('Weights loaded')
        print('')
    
    
    # Managing learning rate
    def get_lr_metric(optimizer) :
        def lr(y_true, y_pred) :
            return optimizer._decayed_lr(tf.float32) 
    return lr
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate, 
                                                                 decay_steps = 10000, 
                                                                 decay_rate = 0.96)
    
    # Params
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)
    loss = tf.keras.losses.BinaryCrossentropy()
    lr_metric = get_lr_metric(optimizer)
    metrics = [tf.keras.losses.BinaryCrossentropy(), tf.keras.metrics.FalseNegatives(),
               tf.keras.metrics.FalsePositives(), tf.keras.metrics.TruePositives(),
               tf.keras.metrics.TrueNegatives(), lr_metric]
    
    # Compiling
    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)   
    
    with tf.device('/GPU:0') :
        # Training
        metrics_dict = model.fit(x = X_train, #x = train_dataset
                  y = y_train,
                  batch_size = batch_size, 
                  epochs = number_epochs, 
                  verbose = 1,
                  validation_data = (X_test, y_test), #validation_data = val_dataset
                  shuffle = True, 
                  callbacks = [es_callback, cp_callback, clearing_callback])
    
    
    print('')
    print('')
    print('Training complete')
    print('')
    
    return metrics_dict
    


def _test_types(_number_samples = 1000, _batch_size = 1, _number_epochs = 20, _learning_rate = 1e-4,
               generate = False, which_dataset = 1) :
    """
    This function generates a dataset and trains the CNN model using the 3 types 
    of preprocessing for comparison.
    """
    
    # Storing metrics 
    metrics = {}
    print('')
    print('--------- Starting CNN preprocessing types comparison ---------')
    
    if generate :
        print('')
        print('Generating ' + str(_number_samples) + ' datapoints')
        print('')
        _SFT_dataset, _labels = generate_dataset_min(number = -1000, 
                             number_samples = _number_samples, 
                             minimum_size = 4281,
                             save_to_disk = False)  
        print('Dataset generated')
        print('')
        print('')
        dataset_id = -1000
    else :
        dataset_id = which_dataset
        _SFT_dataset, _labels = _load_dataset(dataset_id = dataset_id)
        
        
    for _type in range(3) :
        _type += 1
        print('--- TYPE = ' + str(_type) + ' ---')
        m = _train(dataset_id = dataset_id, type_ = _type, batch_size = _batch_size, 
                  number_epochs = _number_epochs, learning_rate = _learning_rate, 
                  SFT_dataset = _SFT_dataset, labels = _labels)
        metrics[str(_type)] = m.history
        print('')
        print('')
        print('')
    
    print('')
    print('Saving results to disk...')
    with open('3_types_metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    print('Results saved')
    print('')
    print('Cleaning models directory...')
    clean_dir(folder = 'CNN/model.nosync') 
    print('Directory cleaned')
    print('')
    
    print('----- All training completed ------')
    print('')
    
    
    
def _generate_and_train(number_iterations = 20, _number_samples = 500, 
                       _type = 3, _batch_size = 16, 
                       _number_epochs = 20, _learning_rate = 1e-4) :
    
    for nb in range(number_iterations) :
        nb += 1
        print('')
        print('')
        print('')
        print('')
        print('-----------------------------------------------------------------------------------')
        print('-----------------------------------------------------------------------------------')
        print('GENERATE AND TRAIN ITERATION ', nb)
        print('-----------------------------------------------------------------------------------')
        print('-----------------------------------------------------------------------------------')
        print('')
        print('')
        # Generate data
        _SFT_dataset, _labels = generate_dataset_min(number = nb, 
                                                     path_to_datasets = 'CNN/datasets.nosync/', 
                                                     number_samples = _number_samples, 
                                                     minimum_size = 4281,
                                                     save_to_disk = False)
        # Train model
        _ = _train(dataset_id = nb, type_ = _type, batch_size = _batch_size, 
                  number_epochs = _number_epochs, learning_rate = _learning_rate, 
                  SFT_dataset = _SFT_dataset, labels = _labels)
        
    print('--------------------------------------------------------------------------------------------------')
    print('--------------------------------------------------------------------------------------------------')
    print('--------------------------------------------------------------------------------------------------')
    print('--------------------------------------------------------------------------------------------------')
    print('--------------------------------------------------------------------------------------------------')
    print('LAST ITERATION IS COMPLETED')
    print('')
    print('')
    


def _test_model(d, true_labels = None, test_SFTs = None) :
    """
    This function performs testing on the pre-trained GLN model to assess the selected
    hyperparameters.
    """
    
    if true_labels is None :
        true_labels = np.load('data.nosync/generated_SFTs/test/test_labels.npy')
        test_SFTs = np.load('data.nosync/generated_SFTs/test/test_SFTs.npy',
                            allow_pickle = True)
        test_SFTs, true_labels = shuffle(test_SFTs, true_labels)
    
    nb_samples = test_SFTs.shape[0]
    predicted_probs = np.zeros(nb_samples)
    
    for i in range(nb_samples) :
        predicted_probs[i] = d.predict(sfts_arr = test_SFTs[i])
        print('\r' + format((i+1)/nb_samples * 100, '.2f'), '% tested', end = '')
    
    print('')
    # Computing metrics
    #print('')
    #print(predicted_probs)
    #print('')
    matrix = confusion_matrix(true_labels.astype(int), (predicted_probs>0.5).astype(int))
    accuracy = accuracy_score(true_labels.astype(int), (predicted_probs>0.5).astype(int))
    results = {}
    results['confusion'] = matrix
    results['accuracy'] = accuracy
    
    return results



def _pretrain_and_test(depth, nb_neurons, context_map_size,
                      train_labels = None, train_SFTs = None,
                      test_labels = None, test_SFTs = None) :
    """
    This function performs pre-training of the GLN model using the set of 
    hyperparameters passed as input.
    """
    
    # Creating model
    layer_sizes = [nb_neurons for n in range(depth)]
    layer_sizes += [1]
    d = Detector(layer_sizes = layer_sizes, context_map_size = context_map_size)
    d.initialize()
    
    # Loading data
    if train_labels is None :
        true_labels = np.load('data.nosync/generated_SFTs/pretrain/pretrain_labels.npy')
        train_SFTs = np.load('data.nosync/generated_SFTs/pretrain/pretrain_SFTs.npy',
                            allow_pickle = True)
        train_SFTs, true_labels = shuffle(train_SFTs, true_labels)
    
    nb_samples = train_SFTs.shape[0]
    
    # Pre-training model
    for i in range(nb_samples) :
        d.train(label = bool(train_labels[i]), sfts_arr = train_SFTs[i])
        print('\r' + format((i+1)/nb_samples * 100, '.2f'), '% of pre-training completed', end = '')
        
    print('')
    
    # Testing
    res = _test_model(d = d, true_labels = test_labels, test_SFTs = test_SFTs)
    
    return res



def _data_pipeline(dataset_purpose = 'tryouts', nb_train_samples = 15, nb_test_samples = 5) :
    """
    This function prepares the data set on which the hyperparameter search will be 
    conducted.
    """
    
    # Data input
    dataset_files =  files_in_dir('CNN/datasets.nosync/' + dataset_purpose)
    labels = np.load('CNN/datasets.nosync/' + dataset_purpose + '/labels/labels.npy',
                     allow_pickle = True)
    labels = np.array([label[0] for label in labels])
    
    
    train_files = dataset_files[:nb_train_samples-1]
    train_SFTs = [[to_complex(np.load(train_files[i])), '-1000', to_complex(np.load(train_files[i+1])),
     '-1000'] for i in range(0, len(train_files), 2)]
    train_SFTs = np.array(train_SFTs, dtype = object)
    train_labels = labels[:nb_train_samples-1]
    
    
    test_files = dataset_files[nb_train_samples-1:nb_train_samples+nb_test_samples]
    test_SFTs = [[to_complex(np.load(test_files[i])), '-1000', to_complex(np.load(test_files[i+1])),
     '-1000'] for i in range(0, len(test_files), 2)]
    test_SFTs = np.array(test_SFTs, dtype = object)
    test_labels = labels[nb_train_samples-1:nb_train_samples+nb_test_samples]
    
    return train_labels, train_SFTs, test_labels, test_SFTs
    
    

def _hyperparameter_search(data_from_CNN_set = True) :
    """
    This function performs a grid-search for finding the optimal hyperparameters 
    to apply to the GLN model.
    """
    
    print('')
    print('---- Starting hyperparameter search ----')
    print('')
    
    # Data input 
    train_labels, train_SFTs, test_labels, test_SFTs = _data_pipeline()
    
    # Building hyperparameters list
    depths = [3, 7, 10] # 15
    depths = [3]
    nb_neurons = [8, 32, 128] # 128
    nb_neurons = [4]
    context_map_size = [4, 7, 10] # 15
    context_map_size = [4]
    if data_from_CNN_set :
        hyperparams = [(x, y, z, train_labels, train_SFTs, test_labels, test_SFTs) \
                       for x in depths for y in nb_neurons for z in context_map_size] 
    else :
        hyperparams = [(x, y, z, None, None, None, None) \
                       for x in depths for y in nb_neurons for z in context_map_size] 
    nb_hyperparams = len(hyperparams)
    
    # Preparing results
    results_dict = {}
    
    i = 0
    for params in hyperparams :
        i += 1
        print('-----------------------------------------------------------------------')
        print('Hyperparameters ' + str(i) + '/' + str(nb_hyperparams) + ' | ' 
              + 'Values : ' + str(params))
        res = _pretrain_and_test(*params) 
        print('Accuracy : ', res['accuracy'])
        results_dict[str(params)] = res
        print('')
        
    print('')
    np.save('data.nosync/hyperparameters_results.npy', results_dict)
    print('Results saved to disk')
    print('---- End of hyperparameter search ----')
    print('')
    
    # Optimal hyperparameters
    max_accuracy = 0
    max_key = None
    for key, value in results_dict.items() :  
        if value['accuracy'] > max_accuracy :   
            max_accuracy = value['accuracy']  
            max_key = key
    print('Optimal hyperparameters : ' + max_key)
        
    
    