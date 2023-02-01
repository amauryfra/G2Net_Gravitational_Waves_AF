# Imports
import numpy as np
from utils import load_provided_SFTs
from generator import Generator
from skimage.metrics import structural_similarity as ssim
from sklearn.neighbors import NearestNeighbors
from progress.bar import FillingCirclesBar
import tensorflow as tf
from train_CNN import SFT_Sequence
from utils import files_in_dir
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras import Input
from detector import Detector
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import hinge_loss
from sklearn.metrics import roc_auc_score



### Testing Generator ###

def test_timestamps_generation(data_SFTs = None, labels = None) :
    """
    This function implements a test of the random timestamps generation process 
    and verifies whether the randomly generated timestamps have the same 
    characteristics as those of the provided data.
    """
    
    print('')
    print('------------------------------------------------')
    print('Testing the timestamps random generation process')
    print('')
    # Loading provided data 
    if data_SFTs is None :
        data_SFTs, labels = load_provided_SFTs(stop_at = None)
    # Signal length
    real_signals_lengths = [data_SFTs[i,0].shape[1] for i in range(data_SFTs.shape[0])]
    real_signals_lengths += [data_SFTs[i,2].shape[1] for i in range(data_SFTs.shape[0])]
    real_signals_lengths = np.array(real_signals_lengths)
    # Time median
    real_signals_time_med = np.array([np.median(i) for i in data_SFTs[:,1]])
    real_signals_time_med = np.concatenate((real_signals_time_med, 
                                             np.array([np.median(i) for i in data_SFTs[:,3]])))
    # Time spacing
    real_signals_diffs = np.array([np.diff(i) for i in data_SFTs[:,1]] +
                                  [np.diff(i) for i in data_SFTs[:,3]] , dtype = object)
    real_signals_percentage_not1800 = np.array([np.count_nonzero(np.where(i != 1800.00))\
                                                / i.shape[0] for i in real_signals_diffs])
    real_signals_max_times = np.array([np.max(i/60) for i in real_signals_diffs])
    real_signals_min_times = np.array([np.min(i/60) for i in real_signals_diffs])
    # Time range 
    real_signals_first_day_arr = np.array([i[0] for i in data_SFTs[:,1]] + 
                                          [i[0] for i in data_SFTs[:,3]])
    real_signals_last_day_arr = np.array([i[-1] for i in data_SFTs[:,1]] + 
                                          [i[-1] for i in data_SFTs[:,3]])
    real_signals_timerange = np.array([i[-1] - i[0] for i in data_SFTs[:,1]] + 
                                      [i[-1] - i[0] for i in data_SFTs[:,3]])/(60*60*24)
    real_signals_timerange = real_signals_timerange[real_signals_timerange > 50]
    
    # Generating random timestamps
    fake_signals = []
    for _ in range(data_SFTs.shape[0]) :
        timestamps = Generator().get_timestamps()
        fake_signals += [timestamps['H1']]
        fake_signals += [timestamps['L1']]
    fake_signals = np.array(fake_signals, dtype = object)
    
    # Signal length
    fake_signals_lengths = [fake_signals[i].shape[0] for i in range(fake_signals.shape[0])]
    # Time median
    fake_signals_time_med = np.array([np.median(i) for i in fake_signals])
    # Time spacing
    fake_signals_diffs = np.array([np.diff(i) for i in fake_signals], dtype = object)
    fake_signals_percentage_not1800 = np.array([np.count_nonzero(np.where(i != 1800.00))\
                                                / i.shape[0] for i in fake_signals_diffs])
    fake_signals_max_times = np.array([np.max(i/60) for i in fake_signals_diffs])
    fake_signals_min_times = np.array([np.min(i/60) for i in fake_signals_diffs])
    # Time range
    fake_signals_first_day_arr = np.array([i[0] for i in fake_signals]) 
    fake_signals_last_day_arr = np.array([i[-1] for i in fake_signals]) 
    fake_signals_timerange = np.array([i[-1] - i[0] for i in fake_signals])/(60*60*24)
    print('----- Mean signal length -----')
    print('Real = ', np.mean(real_signals_lengths))
    print('Fake = ', np.mean(fake_signals_lengths))
    print('----- Mean time median -----')
    print('Real = ', np.mean(real_signals_time_med))
    print('Fake = ', np.mean(fake_signals_time_med))
    print('Difference in hours = ', np.abs(np.median(real_signals_time_med) -
                                            np.median(fake_signals_time_med))\
          /(60*60))
    print('----- Mean time first differences -----')
    real_signals_diffs_m = np.array([np.mean(i) for i in real_signals_diffs])
    fake_signals_diffs_m = np.array([np.mean(i) for i in fake_signals_diffs])
    print('Real = ', np.mean(real_signals_diffs_m))
    print('Fake = ', np.mean(fake_signals_diffs_m))
    print('----- STD time first differences -----')
    print('Real = ', np.std(real_signals_diffs_m))
    print('Fake = ', np.std(fake_signals_diffs_m))
    print('----- Mean of offline slots ratio -----')
    print('Real = ', np.mean(real_signals_percentage_not1800))
    print('Fake = ', np.mean(fake_signals_percentage_not1800))
    print('----- Mean of max offline times -----')
    print('Real = ', np.mean(real_signals_max_times))
    print('Fake = ', np.mean(fake_signals_max_times))
    print('----- Mean of min offline times -----')
    print('Real = ', np.mean(real_signals_min_times))
    print('Fake = ', np.mean(fake_signals_min_times))
    print('----- Mean of time range -----')
    print('Real = ', np.mean(real_signals_timerange))
    print('Fake = ', np.mean(fake_signals_timerange))
    print('----- STD of time range -----')
    print('Real = ', np.std(real_signals_timerange))
    print('Fake = ', np.std(fake_signals_timerange))
    print('----- Median of first day -----')
    print('Real = ', np.median(real_signals_first_day_arr))
    print('Fake = ', np.median(fake_signals_first_day_arr))
    print('Difference in hours = ', np.abs(np.median(real_signals_first_day_arr) - \
          np.median(fake_signals_first_day_arr))/(60*60))
    print('----- Median of last day -----')
    print('Real = ', np.median(real_signals_last_day_arr))
    print('Fake = ', np.median(fake_signals_last_day_arr))
    print('Difference in hours = ', np.abs(np.median(real_signals_last_day_arr) - \
          np.median(fake_signals_last_day_arr))/(60*60))
        
    print('')
    print('End of test')
    print('------------------------------------------------')
    print('')
    
    return data_SFTs, labels



def test_frequency_generation(data_SFTs = None, labels = None) :
    """
    This function implements a test of the random frequency generation process 
    and verifies whether the randomly generated frequencies have the same 
    characteristics as those of the provided data.
    """
        
    print('')
    print('------------------------------------------------')
    print('Testing the frequencies random generation process')
    print('')
    # Loading provided data 
    if data_SFTs is None :
        data_SFTs, labels = load_provided_SFTs(stop_at = None)
    
    real_frequency_medians = np.array([np.median(f) for f in data_SFTs[:,4]])
    real_frequency_medians_with_signal = real_frequency_medians[labels == 1]
    real_frequency_medians_without_signal = real_frequency_medians[labels == 0]
    
    # Generating random timestamps
    fake_frequency_medians = []
    for _ in range(2000) : #data_SFTs.shape[0]
        fake_frequency_median = Generator().get_frequency()
        fake_frequency_medians += [fake_frequency_median]
    fake_frequency_medians = np.array(fake_frequency_medians)
    
    print('----- Mean of frequency medians -----')
    print('Real = ', np.mean(real_frequency_medians))
    print('Real with signal = ', np.mean(real_frequency_medians_with_signal))
    print('Real without signal = ', np.mean(real_frequency_medians_without_signal))
    print('Fake = ', np.mean(fake_frequency_medians))
    
    print('----- STD of frequency medians -----')
    print('Real = ', np.std(real_frequency_medians))
    print('Real with signal = ', np.std(real_frequency_medians_with_signal))
    print('Real without signal = ', np.std(real_frequency_medians_without_signal))
    print('Fake = ', np.std(fake_frequency_medians))
    
    print('----- Max of frequency medians -----')
    print('Real = ', np.max(real_frequency_medians))
    print('Real with signal = ', np.max(real_frequency_medians_with_signal))
    print('Real without signal = ', np.max(real_frequency_medians_without_signal))
    print('Fake = ', np.max(fake_frequency_medians))
    
    print('----- Min of frequency medians -----')
    print('Real = ', np.min(real_frequency_medians))
    print('Real with signal = ', np.min(real_frequency_medians_with_signal))
    print('Real without signal = ', np.min(real_frequency_medians_without_signal))
    print('Fake = ', np.min(fake_frequency_medians))
    
    print('')
    print('End of test')
    print('------------------------------------------------')
    print('')
    
    return data_SFTs, labels
    


def generate_small_testset(nb_with_signal = None, nb_without_signal = None) :
    """
    This function generates a small test set of simulated data computed by the 
    Generator module.
    """
    
    data_list = []
    labels = []

    print('')
    print('--Generating fake SFTs--')
    for i in range(77) :
        if i == nb_without_signal :
            break
        with_signal = False
        SFTs, _ = Generator().generate_sample(with_signal)
        labels += [0]
        data_list += [SFTs]
    for j in range(33) :
        if j == nb_with_signal :
            break
        with_signal = True
        SFTs, _ = Generator().generate_sample(with_signal)
        labels += [1]
        data_list += [SFTs]
    
    data_arr = np.array(data_list, dtype = object)
    labels = np.array(labels)
    print('\n--Data generated--')
    np.save('data.nosync/fake_samples_SFTs', data_arr)
    np.save('data.nosync/fake_samples_labels', labels)
    print('')
    
        
        
def test_generated_data_distribution(fake_data_SFTs = None, fake_labels = None,
                                     data_SFTs = None, labels = None,
                                     from_test = True, do_ssim = False) :
    """
    This function performs several data tests to assess how well the generated 
    data matches the characteristics of the provided training samples.
    """
    
    print('')
    print('------------------------------------------------')
    print('Testing the distribution of generated data')
    print('')
    
    
    ### Loading data ###
    # Loading real data 
    if data_SFTs is None :
        data_SFTs, labels = load_provided_SFTs(from_test = from_test) 
    # Loading fake data
    if fake_data_SFTs is None :
        fake_data_SFTs = np.load('data.nosync/fake_samples_SFTs.npy', allow_pickle = True)
        fake_labels = np.load('data.nosync/fake_samples_labels.npy', allow_pickle = True)
    # Cleaning
    if not from_test :
        data_SFTs = data_SFTs[labels != -1]
        labels = labels[labels != -1]
        assert labels.shape[0] == data_SFTs.shape[0]
    assert fake_labels.shape[0] == fake_data_SFTs.shape[0]


    
    
    ### Building SFT data bank with cropped SFTs of same shape ###
    # Finding minimum value on time axis #
    time_vals_true = np.concatenate((np.array([data[0].shape[1] \
                              for data in data_SFTs]),np.array([data[2].shape[1] \
                                                                for data in data_SFTs])))
    time_vals_fake = np.concatenate((np.array([data[0].shape[1] \
                              for data in fake_data_SFTs]),np.array([data[2].shape[1] \
                                                                     for data in fake_data_SFTs])))
    time_vals = np.concatenate((time_vals_true, time_vals_fake))
    time_vals = time_vals[time_vals > 4000]
    nb_SFTs_fake = time_vals_fake[time_vals_fake > 4000].shape[0]
    nb_SFTs_true = time_vals_true[time_vals_true > 4000].shape[0]
    min_time_vals = np.min(time_vals) # Smallest value on time axis that is > 4000

    del time_vals
    del time_vals_fake
    del time_vals_true
    # Building SFTs bank #
    # Fake #
    fake_SFT_bank = np.zeros((nb_SFTs_fake, 360, min_time_vals), dtype = np.complex64)
    i = -1
    for fake_SFTs in fake_data_SFTs :
        if fake_SFTs[0].shape[1] > 4000 : # Cleaining unrelevantly small SFTs
            i += 1
            fake_SFT_bank[i, :, :] = fake_SFTs[0][:, :min_time_vals]
        if fake_SFTs[2].shape[1] > 4000 : 
            i += 1
            fake_SFT_bank[i, :, :] = fake_SFTs[2][:, :min_time_vals]
        print('\r' + 'Building set of cropped SFTs for generated data... ' +
                  format(i/(nb_SFTs_fake-1) * 100, '.2f'), '%', end = '')
    assert nb_SFTs_fake - 1 == i 
    del fake_data_SFTs
    # True #
    true_SFT_bank = np.zeros((nb_SFTs_true, 360, min_time_vals), dtype = np.complex64)
    i = -1
    print('')
    for SFTs in data_SFTs :
        if SFTs[0].shape[1] > 4000 :
            i += 1
            true_SFT_bank[i, :, :] = SFTs[0][:, : min_time_vals]
        if SFTs[2].shape[1] > 4000 :
            i += 1 
            true_SFT_bank[i, :, :] = SFTs[2][:, : min_time_vals]
        print('\r' + 'Building set of cropped SFTs for provided data... ' +
                      format(i/(nb_SFTs_true-1) * 100, '.2f'), '%', end = '')
    assert nb_SFTs_true - 1 == i 
    del data_SFTs
    print('')
    
    print('')
    print('Processing arrays...')
    fake_SFT_bank = np.array([np.array([np.real(SFT), np.imag(SFT)]) for SFT in fake_SFT_bank])
    fake_SFT_bank = fake_SFT_bank.reshape(fake_SFT_bank.shape[0], 
                          fake_SFT_bank.shape[2], 
                          fake_SFT_bank.shape[3], 
                          fake_SFT_bank.shape[1])
    
    true_SFT_bank = np.array([np.array([np.real(SFT), np.imag(SFT)]) for SFT in true_SFT_bank])
    true_SFT_bank = true_SFT_bank.reshape(true_SFT_bank.shape[0], 
                          true_SFT_bank.shape[2], 
                          true_SFT_bank.shape[3], 
                          true_SFT_bank.shape[1])
    

    ### From here dataset are prepared ###
    # Usable data structures : (fake_SFT_bank,true_SFT_bank) #
    ### Nearest neighbors Test ###
    
    # Using structural similarity 
    
    def metric(a, b) :
        return (1 / (1 + ssim(1e21 * a, 1e21 * b, channel_axis = 2, win_size = 107))) - 0.5
    
    print('')
    print('------Performing nearest neighbors test------')
    print('Pre-processing arrays...')
    
    nb_fake = fake_SFT_bank.shape[0]
    nb_true = true_SFT_bank.shape[0]
    stacked_SFTs = np.vstack((fake_SFT_bank,true_SFT_bank))
    del true_SFT_bank
    del fake_SFT_bank
    print('Arrays pre-processed')
    
    if do_ssim :
        
        print('')
        print('---Fitting kNN model with SSIM---')
        print('')
        mx = (stacked_SFTs.shape[0] * (stacked_SFTs.shape[0] + 1) / 2) - stacked_SFTs.shape[0]
        bar = FillingCirclesBar('Generating distances matrix',\
                                max = mx, \
                                suffix = '%(percent).1f%% - %(eta)ds')
        D = np.inf * np.ones((stacked_SFTs.shape[0], stacked_SFTs.shape[0]))
        for i in range(stacked_SFTs.shape[0]) :
            for j in range(i+1, stacked_SFTs.shape[0]) :
                D[i,j] = metric(stacked_SFTs[i], stacked_SFTs[j])
                D[j,i] = D[i,j]
                bar.next()
        bar.finish()
        print('')
        bar = FillingCirclesBar('Computing nearest neighbors',\
                                max = stacked_SFTs.shape[0], \
                                suffix = '%(percent).1f%% - %(eta)ds')
        indices_ssim = np.column_stack((np.arange(stacked_SFTs.shape[0]), np.arange(stacked_SFTs.shape[0])))
        for k in range(stacked_SFTs.shape[0]) :
            indices_ssim[k][1] = np.argmin(D[k,:])
            bar.next()
        bar.finish()
        print('')
        
        print('Indexes obtained')
        print('Building results...')
        print('')
        true_neighbors = 0
        fake_neighbors = 0
        for i in range(nb_fake) :
            neighbor = indices_ssim[i, 1]
            if neighbor > nb_fake - 1 : true_neighbors+=1
            else : fake_neighbors+=1
        print('Fake SFTs have : ' + format(true_neighbors/nb_fake * 100, '.2f') \
              + '% True neighbors')
        print('Fake SFTs have : ' + format(fake_neighbors/nb_fake * 100, '.2f') \
              + '% Fake neighbors')
        print('')
        true_neighbors = 0
        fake_neighbors = 0
        for i in range(nb_true) :
            neighbor = indices_ssim[i + nb_fake, 1]
            if neighbor > nb_fake - 1 : true_neighbors+=1
            else : fake_neighbors+=1
        print('True SFTs have : ' + format(true_neighbors/nb_true * 100, '.2f') \
              + '% True neighbors')
        print('True SFTs have : ' + format(fake_neighbors/nb_true * 100, '.2f') \
              + '% Fake neighbors')
        
            
        print('')
        print('')
        
    
    print('')
    print('---Fitting kNN model with Minkowski---')
    print('Pre-processing arrays...')
    stacked_SFTs = stacked_SFTs.reshape(-1, stacked_SFTs.shape[0]).T
    print('Arrays pre-processed')
    nbrs = NearestNeighbors(n_neighbors = 2,
                            algorithm = 'auto').fit(stacked_SFTs)

    print('Model fitted')
    print('Retreiving neighbors indexes...')
    _, indices_mink = nbrs.kneighbors(stacked_SFTs)
    print('Indexes obtained')
    print('Building results...')
    print('')
    true_neighbors = 0
    fake_neighbors = 0
    for i in range(nb_fake) :
        neighbor = indices_mink[i, 1]
        if neighbor > nb_fake - 1 : true_neighbors+=1
        else : fake_neighbors+=1
    print('Fake SFTs have : ' + format(true_neighbors/nb_fake * 100, '.2f') \
          + '% True neighbors')
    print('Fake SFTs have : ' + format(fake_neighbors/nb_fake * 100, '.2f') \
          + '% Fake neighbors')
    print('')
    true_neighbors = 0
    fake_neighbors = 0
    for i in range(nb_true) :
        neighbor = indices_mink[i + nb_fake, 1]
        if neighbor > nb_fake - 1 : true_neighbors+=1
        else : fake_neighbors+=1
    print('True SFTs have : ' + format(true_neighbors/nb_true * 100, '.2f') \
          + '% True neighbors')
    print('True SFTs have : ' + format(fake_neighbors/nb_true * 100, '.2f') \
          + '% Fake neighbors')
        
    if do_ssim : return D, indices_ssim, indices_mink
    else : return indices_mink
    

    
### Testing CNN ###


def test_CNN() :
    """
    This function peforms testing on the pre-trained CNN.
    """
    
    # Preparing data pipeline
    dataset_files =  files_in_dir('CNN/datasets.nosync/test')
    labels = np.load('CNN/datasets.nosync/' + 'test' + '/labels/labels.npy',
                     allow_pickle = True)
    labels = np.array([label[0] for label in labels])
    
    val_dataset = SFT_Sequence(dataset_files, labels, batch_size = 8, type_ = 2)
    shape_ = val_dataset.__getitem__(0)[0].shape[1:]
    dtype_ = val_dataset.__getitem__(0)[0].dtype
    img_inputs = Input(shape = shape_, dtype = dtype_)
    
    weights = 'CNN/model.nosync/weights_type=2.hdf5'
    model = InceptionResNetV2(input_tensor = img_inputs, weights = weights, classes = 2)
    
    # Params
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4)
    loss = tf.keras.losses.CategoricalCrossentropy() 
    metrics = [tf.keras.losses.CategoricalCrossentropy(), tf.keras.metrics.CategoricalAccuracy(),
               tf.keras.losses.CategoricalHinge()]
    
    
    # Compiling
    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)   
    
    
    with tf.device('/GPU:0') :
        # Evaluating
        results = model.evaluate(val_dataset)
        
    
    return results
    
    
    
### Testing GLN ###


def test_GLN(verbose = 1, from_test = False) :
    """
    This function implements testing to evaluate the Gated Linear Network model.
    """
    
    print('')
    print('GLN test session ')
    
    # Loading provided SFTs
    SFTs, labels = load_provided_SFTs(stop_at = 100, from_test = from_test)
    
    # Processing SFTs
    if not from_test :
        mask1 = labels != -1  
        mask2 = np.array([SFT[0].shape[1] > 4280  and SFT[2].shape[1] > 4280 for SFT in SFTs])
        mask = np.array([(m[0] and m[1]) for m in zip(mask1, mask2)])
        print('Number of samples discarded : ', mask1.shape[0] - list(mask).count(True))
        SFTs = SFTs[mask]
        labels = labels[mask]
    
    nb_samples = SFTs.shape[0]
    predicted_probs = np.zeros(nb_samples)
    
    # Initializing detector
    detector = Detector()
    print('Running with latest model saved on : ', 
          detector.model_date.strftime('%d/%m/%Y, %H:%M:%S'))
    print('Training iterations already performed : ', detector.training_iterations)
    
    for i in range(nb_samples) :
        predicted_probs[i] = detector.predict(sfts_arr = SFTs[i])
        if verbose == 0 :
            print('\r' + format((i+1)/nb_samples * 100, '.2f'), '% tested', end = '')
        elif verbose == 1 and not from_test :
            print(f'Sample {i + 1}/{nb_samples} | Target = {labels[i]} | Prediction = {predicted_probs[i]}')
        else :
            print(f'Sample {i + 1}/{nb_samples} | Prediction = {predicted_probs[i]}')
    
    print('')
    
    if not from_test :
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
        
    
    
if __name__ == '__main__' :
    test_GLN()
    
    
    