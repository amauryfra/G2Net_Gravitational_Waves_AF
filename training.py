# Imports
print('')
import numpy as np
from detector import Detector
from generator import Generator
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import hinge_loss
from sklearn.metrics import roc_auc_score


class Trainer() :
    """
    This class contains all methods necessary to perfrom the online training of 
    the Gated Linear Network.
    """
    
    def __init__(self, nb_iterations = 18500, test_and_save_steps = 250) :
        
        print('')
        print('------- Preparing trainer module -------')
        # Number of iterations for online learning
        self.nb_iterations = nb_iterations
        
        # Managing detector
        print('Initializing detector...')
        self.detector = Detector()
        print('Detector initialized.')
        
        if self.detector.loaded_model :
            print('Running with latest model saved on : ', 
                  self.detector.model_date.strftime('%d/%m/%Y, %H:%M:%S'))
            print('Training iterations already performed : ', self.detector.training_iterations)
            print('Current learning rate : ', self.detector.learning_rate)
        else :
            print('Created a new model. Starting from scratch.')
            
        
        # Managing generator
        self.generator = Generator(enable_prints = False)
        
        # Managing data
        print('Loading test data...')
        self.dataset_labels = np.load('data.nosync/generated_SFTs/pretrain/pretrain1_labels.npy') 
        self.dataset_SFTs = np.load('data.nosync/generated_SFTs/pretrain/pretrain1_SFTs.npy',
                            allow_pickle = True)
        print('Data loaded.')
        print('')
        
        # Testing and saving 
        self.test_and_save_steps = test_and_save_steps
    
    
    
    def test_session(self, nb) :
        """
        This method implements a testing session to evaluate the model on-the-go.
        """
        
        print('---- Test session ' + str(nb) + ' ----')
        # Selecting some testing data
        indx = np.random.choice(self.dataset_labels.shape[0], size = 250, replace = False)
        labels = self.dataset_labels[indx]
        SFTs = self.dataset_SFTs[indx]
        
        nb_samples = labels.shape[0]
        predicted_probs = np.zeros(nb_samples)
        
        for i in range(nb_samples) :
            predicted_probs[i] = self.detector.predict(sfts_arr = SFTs[i])
            print('\r' + format((i+1)/nb_samples * 100, '.2f'), '% tested', end = '')
        
        print('')
        # Computing metrics
        auc = roc_auc_score(labels.astype(int), (predicted_probs>0.5).astype(int))
        matrix = confusion_matrix(labels.astype(int), (predicted_probs>0.5).astype(int))
        accuracy = accuracy_score(labels.astype(int), (predicted_probs>0.5).astype(int))
        loss = hinge_loss(labels.astype(float), predicted_probs)
        
        print('Results : ')
        print('AUC score = ', auc) 
        print('Accuracy = ', accuracy)
        print('Confusion matrix = ', matrix)
        print('Hinge loss = ', loss)
        print('Sample probabilities -> ', np.random.choice(predicted_probs, size = 6, replace = False))
        print('---- Test session completed ----')
        print('')
        
        if type(nb) != str : 
            return nb + 1
        
    
    
    def training_iteration(self, nb) :
        """
        This method implements a single training iteration for the GLN. It performs
        both the data generation and the model's update.
        """
        
        print('Iteration ' + str(nb) + '/' + str(self.nb_iterations))
        with_signal = bool(np.random.randint(2))
        print('Generating signal...')
        SFTs, _, D_storage = self.generator.generate_sample(with_signal = with_signal)
        
        print('Online update...')
        prob1, prob2 = self.detector.train(label = int(with_signal), 
                                           sfts_arr = SFTs, return_probs = True)
        prob_average = 0.5 * (prob1[0] + prob2[0])
        if with_signal : 
            print('Target = ' + str(int(with_signal)) + ' | Prediction = ' + str(prob_average))
            print('P1 = ' + str(prob1[0]) + ' | P2 = ' + str(prob2[0]) + ' | ' + 'D = ' + str(D_storage))
        else :
            print('Target = ' + str(int(with_signal)) + ' | Prediction = ' + str(prob_average))
            print('P1 = ' + str(prob1[0]) + ' | P2 = ' + str(prob2[0]))
            
        return True
        
    
    
    def train_session(self) :
        """
        Master method. This method performs the full training session of the
        GLN model.
        """
        
        print('----------------------- Training Session -----------------------')
        print('Number of iterations to be performed : ', self.nb_iterations)
        print('')
        
        test_session_nb = 1
        for it in range(self.nb_iterations) :
            
            # Testing 
            if it == 0 or (it+1)%self.test_and_save_steps == 0 : 
                test_session_nb = self.test_session(test_session_nb)
                
            # Saving
            if it > 0 and (it+1)%self.test_and_save_steps == 0 : 
                print('--->| Saving model...')
                self.detector.save_model()
                print('--->| Model saved to disk.')
                print('')
                print('Current learning rate : ', self.detector.model.learning_rate.constant_value)
                print('')
            
            # Training
            _ = self.training_iteration(it + 1)
            if (it+1)%20 == 0 and it > 0 :
                print('Current learning rate : ', self.detector.model.learning_rate.constant_value)
            print('')
        
        
        # Last test session
        _ = self.test_session('Final')
        # Saving final model
        print('--->| Saving model...')
        self.detector.save_model()
        print('--->| Model saved to disk.')
        print('')
        print('')
        print('----------------------- Training Completed -----------------------')
        print('')
        print('')
            
    
    
if __name__ == '__main__' :
    t = Trainer()
    t.train_session()

        
        