# Imports
import numpy as np
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras import Input
from train_CNN import process_datasample


def initialize_extractor(type_ = 2, path_to_weights = 'CNN/model.nosync/') :
    """
    This function intializes the pre-trained CNN for building the predictor.
    """
    
    print('Initializing feature extractor...')
    if type_ == 1 : shape = (360, 4281, 2)
    elif type_ == 2 : shape = (360, 355, 1)
    else : shape = (360, 4281, 1)
    
    img_inputs = Input(shape = shape, dtype = 'float64')
    weights = path_to_weights + 'weights_type=' + str(type_) + '.hdf5'
    model = InceptionResNetV2(include_top = False,
                            input_tensor = img_inputs,
                            weights = None, classes = 2,
                            pooling = 'max')
    print('Loading weights...')
    model.load_weights(weights, by_name = True, skip_mismatch = True) 
    print('Weights loaded.')
    
    return model



def extract_features(img, model, type_ = 2) :
    """
    This function uses the CNN model to extract high-level features to be fed 
    to the GLN model. 
    """
    
    img = process_datasample(img, type_ = type_)
    img = np.expand_dims(img, -1)
    res = model(np.array([img]))
    return res.numpy().flatten()
    


if __name__ == '__main__' :
    model = initialize_extractor(type_ = 2, path_to_weights = 'CNN/model.nosync/')
    res = extract_features(np.load('CNN/datasets.nosync/test/0_1_fzNGNL9f9VBDscvnLcwQV7.npy'), model)
    
    
    