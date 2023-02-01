# Imports
from os import listdir
from os.path import isfile, join
from datetime import datetime
import numpy as np
from utils import argmax
from utils import load_GLN
from utils import read_hdf5
from utils import save_GLN
from utils import to_2_channels
from dataset_generator import crop_center
from feature_extractor import initialize_extractor
from feature_extractor import extract_features
import pygln
import warnings
warnings.filterwarnings('ignore')


class Detector() :
    """
    This class implements the detector used to classify the input signals as
    holding a continuous gravitational-wave or not.
    """

    def __init__(self,
                 type_ : int = 3,
                 layer_sizes: list = [32, 32, 32, 32, 32, 1],
                 context_map_size: int = 10, 
                 learning_rate: float = 1e-4,
                 reset_learning_rate: bool = True,
                 activate_decay: bool = False,
                 pred_clipping: float = 1e-3,
                 weight_clipping: float = 5.0,
                 base_predictor = None) :
        
        self.type_= type_
        self.loaded_model = False
        self.initialized = False
        self.training_iterations = 0
        self.current_training_iterations = 0
        self.model_date = datetime.now()
        self.last_save_date = None

        self.layer_sizes = layer_sizes
        self.context_map_size = context_map_size
        self.learning_rate = learning_rate
        self.reset_learning_rate = reset_learning_rate
        self.activate_decay = activate_decay
        self.pred_clipping = pred_clipping
        self.weight_clipping = weight_clipping
        self.base_predictor = base_predictor
        
        if self.type_ == 1 : self.input_size = 721
        elif self.type_ == 2 : self.input_size = 1441
        elif self.type_ == 3 : self.input_size = 1536
        
        if self.type_ == 3 :
            self.CNN_model = initialize_extractor()


        # Initializing a new Gated Linear Network (GLN) or loading the most recent model
        # available in models directory.
        
        path = 'models/' + 'type_' + str(self.type_) 
        if not len(listdir(path)) == 0 and not (len(listdir(path)) == 1
                                                    and listdir(path)[0] == '.DS_Store') :

            models_list = [f for f in listdir(path) if \
                           isfile(join(path, f))]
            try :
                models_list.remove('.DS_Store')
                models_list.remove('.gitignore')
            except :
                pass
            
            models_date = [datetime.fromtimestamp(float(model[:-4].split('_')[1].replace('-','.'))) \
                           for model in models_list]
            latest_model_index = argmax(models_date)
            latest_model = models_list[latest_model_index]
            self.model = load_GLN(path + '/' + latest_model)

            self.loaded_model = True
            self.model_date = datetime.fromtimestamp(float(latest_model[:-4].split('_')[1].replace('-','.'))) 
            self.training_iterations = int(models_list[latest_model_index][:-4].split('_')[2])
            
            if not self.reset_learning_rate :
                self.learning_rate = 1. * self.model.learning_rate.constant_value
                self.current_training_iterations = 1 * self.training_iterations
            else : 
                self.model.learning_rate.constant_value = 1. * self.learning_rate
                
        else :
            self.model = pygln.GLN(backend = 'pytorch', layer_sizes = self.layer_sizes,
                        input_size = self.input_size,
                        context_map_size = self.context_map_size,
                        base_predictor = self.base_predictor,
                        learning_rate = self.learning_rate,
                        pred_clipping = self.pred_clipping,
                        weight_clipping = self.weight_clipping)

        self.initialized = True



    def save_model(self) :
        """
        Saving the GLN to disk.
        """
        self.last_save_date = datetime.now()
        self.learning_rate = self.model.learning_rate.constant_value
        save_GLN(self.model, 'models/type_' + str(self.type_) + '/model_' +
                 str(datetime.timestamp(self.last_save_date)).replace('.','-') +
                 '_' + str(self.training_iterations) + '_type=' + str(self.type_) + '.pkl')



    def build_predictor(self, sfts_arr = None, sft_hdf5file = None) :
        """
        This function builds the predictor passed to the GLN model for prediction,
        based on a short Fourier transform as input.
        """

        # Reading hdf5 file
        if sfts_arr is None :
            sfts_arr = read_hdf5(sft_hdf5file)
        
        if self.type_ == 1 :
            
            # First predictor -> mean of frequencies
            frequency_median = np.median(sfts_arr[4])
            
            # Following predictors :
            # SFT is first reduced by computing mean of values upon each frequency
            # Then real and imaginary parts are stacked side by side along with the frequency median
            squished_arr_H = np.mean(sfts_arr[0], axis = 1)
            predictor_H = np.concatenate((np.array([frequency_median]), np.real(squished_arr_H), \
                            np.imag(squished_arr_H)))
            # Scaling
            predictor_H[1:] *= 1e25
    
            squished_arr_L = np.mean(sfts_arr[2], axis = 1)
            predictor_L = np.concatenate((np.array([frequency_median]), np.real(squished_arr_L), \
                                np.imag(squished_arr_L)))
            # Scaling
            predictor_L[1:] *= 1e25
            
        elif self.type_ == 2 :
            
            # First predictor -> mean of frequencies
            frequency_median = np.median(sfts_arr[4])
            
            # Following predictors :
            # Real and imaginary parts of SFT are separated and stacked side by side
            # Then the obtained array is reduced by computing mean of values upon each frequency
            # as well as standard deviation of values upon each frequency
            # Mean array and STD array are stacked side by side along with the frequency median
            extend_arr_H = np.vstack((np.real(sfts_arr[0]), np.imag(sfts_arr[0])))
            mean_extend_arr_H = np.mean(extend_arr_H, axis = 1)
            std_extend_arr_H = np.std(extend_arr_H, axis = 1)
            predictor_H = np.concatenate((np.array([frequency_median]), mean_extend_arr_H, \
                            std_extend_arr_H))
            # Scaling
            predictor_H[1:] *= 1e25
            
            extend_arr_L = np.vstack((np.real(sfts_arr[2]), np.imag(sfts_arr[2])))
            mean_extend_arr_L = np.mean(extend_arr_L, axis = 1)
            std_extend_arr_L = np.std(extend_arr_L, axis = 1)
            predictor_L = np.concatenate((np.array([frequency_median]), mean_extend_arr_L, \
                            std_extend_arr_L))
                
            # Scaling
            predictor_L[1:] *= 1e25
        
        elif self.type_ == 3 :
            
            # Using last flattened layer of pre-trained CNN as predictor
            img1 = to_2_channels(crop_center(sfts_arr[0], minimum_size = 4281))
            predictor_H = extract_features(img1, self.CNN_model)
            
            img2 = to_2_channels(crop_center(sfts_arr[2], minimum_size = 4281))
            predictor_L = extract_features(img2, self.CNN_model)
            
        return predictor_H.reshape((1,self.input_size)), predictor_L.reshape((1,self.input_size))



    def predict(self, sfts_arr = None, sft_hdf5file = None) :
        """
        Performs a prediction based on a short Fourier transform as input.
        """

        # Building predictor
        predictor_H, predictor_L = self.build_predictor(sfts_arr = sfts_arr, 
                                                        sft_hdf5file = sft_hdf5file)
        
        # Averaging predictions made on each interferometers
        prediction_H = self.model.predict(predictor_H, return_probs = True)
        prediction_L = self.model.predict(predictor_L, return_probs = True)
        return 0.5 * (prediction_H + prediction_L)



    def train(self, label, sfts_arr = None, sft_hdf5file = None, return_probs = False) :
        """
        Train the current model via an online update. A prediction is made based
        on a short Fourier transform as input and is
        compared with the target class to perform the update.
        """

        # Building predictor
        predictor_H, predictor_L = self.build_predictor(sft_hdf5file = sft_hdf5file, 
                                                        sfts_arr = sfts_arr)

        # Training on both interferometers data
        label_arr = np.array([label])
        prob1 = self.model.predict(predictor_H, target = label_arr, return_probs = return_probs)
        prob2 = self.model.predict(predictor_L, target = label_arr, return_probs = return_probs)
        
        self.training_iterations += 1
        self.current_training_iterations += 1
        # Learning rate decay 
        if self.activate_decay :
            self.model.learning_rate.constant_value = 1.0 * ((1.0 * self.learning_rate) / \
                (1 + 5e-3 * self.current_training_iterations))
            
        if return_probs :
            return prob1, prob2
            
        

if __name__ == '__main__' :
    d = Detector()
    res = d.predict(sft_hdf5file = 'data.nosync/test_SFTs/0abf30d11.hdf5') 
    d.train(sft_hdf5file = 'data.nosync/test_SFTs/0abf30d11.hdf5', label = 0.0)
    
    
    