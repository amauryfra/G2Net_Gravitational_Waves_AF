# Imports
print('')
import numpy as np
from generator import Generator
from detector import Detector 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import hinge_loss
from sklearn.utils import shuffle


def generate_SFT_set(nb_samples, name, 
                           path_to_dataset = 'data.nosync/generated_SFTs/test') :
    """
    This function creates a dataset of generated data computed by the 
    Generator module.
    """
    
    data_list = []
    labels = []

    print('')
    print('--Generating SFTs--')
    with_signal = False
    
    for i in range(nb_samples) :
        with_signal = not with_signal
        SFTs, _ = \
            Generator(enable_prints = False).generate_sample(with_signal = with_signal)
        labels += [int(with_signal)]
        data_list += [SFTs]
        i += 1
        print('\r' + format(i/nb_samples * 100, '.2f'), '% created', end = '')
    data_arr = np.array(data_list, dtype = object)
    labels = np.array(labels)
    print('\n--Data generated--')
    np.save(path_to_dataset + '/' + name + '_SFTs.npy', data_arr)
    np.save(path_to_dataset + '/' + name + '_labels.npy', labels)
    print('')
    
    
    
class Tuner() :
    """
    This class implements a set of methods for performing a hyperparameter search
    for the Gated Linear Network model.
    """
    
    def __init__(self, hyperparams = None) :
    
        # Loading data 
        print('')
        print('Loading data...')
        self.train_labels = np.concatenate((np.load('data.nosync/generated_SFTs/pretrain/pretrain1_labels.npy'),
                                      np.load('data.nosync/generated_SFTs/pretrain/pretrain2_labels.npy')))
        self.train_SFTs = np.concatenate((np.load('data.nosync/generated_SFTs/pretrain/pretrain1_SFTs.npy',
                            allow_pickle = True),
                                     np.load('data.nosync/generated_SFTs/pretrain/pretrain2_SFTs.npy',
                                                         allow_pickle = True)))
        self.test_labels = np.load('data.nosync/generated_SFTs/test/test_labels.npy')
        self.test_SFTs = np.load('data.nosync/generated_SFTs/test/test_SFTs.npy',
                            allow_pickle = True)
        print('Data loaded')
        print('')
        
        
        # Building hyperparameters list
        self.depths = [3, 5, 10] 
        self.nb_neurons = [8, 16, 32, 128]
        self.context_map_size = [4, 6, 8, 10, 15]
        self.hyperparams = hyperparams
        if self.hyperparams == None :
            self.hyperparams = [(x, y, z) \
                           for x in self.depths for y in self.nb_neurons for z in self.context_map_size] 
        self.nb_hyperparams = len(self.hyperparams)
        
    
    
    def test_model(self, d) :
        """
        This function performs testing on the pre-trained GLN model to assess the selected
        hyperparameters.
        """
        
        SFTs, labels = shuffle(self.test_SFTs, self.test_labels)
        
        nb_samples = self.test_labels.shape[0]
        predicted_probs = np.zeros(nb_samples)
        
        for i in range(nb_samples) :
            predicted_probs[i] = d.predict(sfts_arr = list(self.test_SFTs[i]))
            print('\r' + format((i+1)/nb_samples * 100, '.2f'), '% tested', end = '')
        
        print('')
        # Computing metrics
        matrix = confusion_matrix(self.test_labels.astype(int), (predicted_probs>0.5).astype(int))
        accuracy = accuracy_score(self.test_labels.astype(int), (predicted_probs>0.5).astype(int))
        loss = hinge_loss(self.test_labels.astype(float), predicted_probs)
        results = {}
        results['confusion'] = matrix
        results['accuracy'] = accuracy
        results['loss'] = loss
        results['probabilities'] = predicted_probs

        return results
    
    
    
    def pretrain_model(self, depth, nb_neurons, context_map_size) :
        """
        This function performs pre-training of the GLN model using the set of 
        hyperparameters passed as input.
        """
        
        # Creating model
        layer_sizes = [nb_neurons for n in range(depth)]
        layer_sizes += [1]
        d = Detector(layer_sizes = layer_sizes, context_map_size = context_map_size)
        
        
        SFTs, labels = shuffle(self.train_SFTs, self.train_labels)
        nb_samples = labels.shape[0]
        
        # Pre-training model
        for i in range(nb_samples) :
            d.train(label = self.train_labels[i], sfts_arr = list(self.train_SFTs[i]))
            print('\r' + format((i+1)/nb_samples * 100, '.2f'), '% of pre-training completed', end = '')
        print('')
        
        return d
        
    
    
    def hyperparameter_search(self) :
        """
        This function performs a grid-search for finding the optimal hyperparameters 
        to apply to the GLN model.
        """
        
        print('')
        print('---- Starting hyperparameter search ----')
        print('')
        
        # Preparing results 
        results_dict = {}
        
        i = 0
        for params in self.hyperparams :
            i += 1
            print('-----------------------------------------------------------------------')
            print('Hyperparameters ' + str(i) + '/' + str(self.nb_hyperparams) + ' | ' 
                  + 'Values : ' + str(params))
            d = self.pretrain_model(*params) 
            res = self.test_model(d = d)
            print('Accuracy : ', res['accuracy'])
            print('Confusion matrix : ', res['confusion'])
            print('Loss : ', res['loss'])                               
            print('Sample probabilities : ', np.random.choice(res['probabilities'], size = 6, replace = False))
            results_dict[str(params[:3])] = res
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
        print('')
            


if __name__ == '__main__' :
    t = Tuner() 
    t.hyperparameter_search()
    
    
    