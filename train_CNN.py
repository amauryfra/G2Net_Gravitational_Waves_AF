# Imports
import os
import numpy as np
import math
import tensorflow as tf
from utils import to_complex
from utils import denoise_SFT
from utils import clean_dir
from utils import files_in_dir
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras import Input
import os.path
import gc
import pickle


def scale(num) :
    """
    Scaling float to [-1,1] range.
    """
    
    return ((num - np.min(num)) / (np.max(num) - np.min(num))) * 2. - 1.



def process_datasample(img, type_) :
    """
    This function applies the moving average denoising procedure to the SFT 
    datasample provided.
    """
        
    if type_ == 2 :
        fourier_data = to_complex(img)
        processed_datasample = denoise_SFT(1e22 * fourier_data)  
        processed_datasample = scale(processed_datasample)
            
    if type_ == 3 :
        fourier_data = to_complex(img)
        processed_datasample = np.square(np.abs(fourier_data))
        
    return processed_datasample


        
class SFT_Sequence(tf.keras.utils.Sequence) :
    """
    tf.keras Sequence as data pipeline for CNN model.
    """

    def __init__(self, x_set, y_set, batch_size, type_):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.type_ = type_
        
    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        
        target = tf.keras.utils.to_categorical(y = batch_y, num_classes = 2, dtype = 'float64')

        if self.type_ > 1 :
            return np.expand_dims(np.array([ 
                process_datasample(np.load(file_name), self.type_)
                   for file_name in batch_x]), -1), target
        
        return np.array([np.load(file_name)
               for file_name in batch_x]), target
        
    
    
def get_datasets_pipeline(type_, dataset_purpose, batch_size) :
    """
    This functions gets the training and testing datasets as tf.keras Sequence
    for use in the CNN model.
    """
    
    # Preparing data pipeline
    dataset_files =  files_in_dir('CNN/datasets.nosync/' + dataset_purpose)
    labels = np.load('CNN/datasets.nosync/' + dataset_purpose + '/labels/labels.npy',
                     allow_pickle = True)
    labels = np.array([label[0] for label in labels])
    
    X_train, X_test, y_train, y_test = train_test_split(dataset_files, labels, 
                                                        test_size = 0.20, shuffle = True)
    
    train_dataset = SFT_Sequence(X_train, y_train, batch_size, type_)
    test_dataset = SFT_Sequence(X_test, y_test, batch_size, type_)
    
    return train_dataset, test_dataset
    
    
    
class ClearingCallback(tf.keras.callbacks.Callback) :
    """
    Callback for solving memory issue through memory clearing.
    """
    
    def on_epoch_end(self, epoch, logs = None) :
        gc.collect()
        tf.keras.backend.clear_session()



def train(type_ = 2, dataset_purpose = 'test', batch_size = 32, number_epochs = 40,
          learning_rate = 1e-2) :
    """
    This function creates a CNN, loads existing weights when applicable and 
    performs training.
    """
    
    # Preparing data 
    train_dataset, test_dataset = get_datasets_pipeline(type_, dataset_purpose, batch_size)
    shape_ = train_dataset.__getitem__(0)[0].shape[1:]
    dtype_ = train_dataset.__getitem__(0)[0].dtype
    
    # Input shape
    img_inputs = Input(shape = shape_, dtype = dtype_)
    
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
    model = InceptionResNetV2(input_tensor = img_inputs, weights = weights, classes = 2)
    if weights is not None : 
        print('Weights loaded')
        print('')
        
        
    # Params
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate, 
                                                                 decay_steps = 5, 
                                                                 decay_rate = 0.96)
    optimizer = tf.keras.optimizers.SGD(learning_rate = lr_schedule) 
    # Managing learning rate
    def lr_metric(y_true, y_pred, optimizer = optimizer) :
        return optimizer.learning_rate
    loss = tf.keras.losses.CategoricalCrossentropy()
    metrics = [tf.keras.losses.CategoricalCrossentropy(), tf.keras.losses.BinaryCrossentropy(), tf.keras.losses.CategoricalHinge(), 
               tf.keras.metrics.CategoricalAccuracy(), lr_metric]
    
    
    # Compiling
    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)   
    
    
    with tf.device('/GPU:0') :
        # Training
        metrics_dict = model.fit(x = train_dataset,
                  epochs = number_epochs, 
                  verbose = 1,
                  validation_data = test_dataset, 
                  shuffle = True, 
                  callbacks = [es_callback, cp_callback, clearing_callback])
    
    
    print('')
    print('')
    print('Training complete')
    print('')
    
    return metrics_dict



def test_types(batch_size = 8, number_epochs = 15, learning_rate = 1e-4,
               dataset_purpose = 'train') :
    """
    This function generates a dataset and trains the CNN model using the 3 types 
    of preprocessing for comparison.
    """
    
    # Storing metrics 
    metrics = {}
    print('')
    print('--------- Starting CNN preprocessing types comparison ---------')
    print('')
    print('')
    print('')
        
    for _type in range(3) :
        _type += 1
        print('--- TYPE = ' + str(_type) + ' ---')
        m = train(type_ = _type, dataset_purpose = dataset_purpose, 
                  batch_size = batch_size, number_epochs = number_epochs,
                  learning_rate = learning_rate) 
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
    
    
    
if __name__ == '__main__' :
    train()
    
    
    