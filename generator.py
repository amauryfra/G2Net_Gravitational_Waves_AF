# Imports
import numpy as np
import scipy
import scipy.stats as stats
import pyfstat
logger = pyfstat.set_up_logger(log_level = 'CRITICAL')
from pyfstat.utils import get_sft_as_arrays
from utils import clean_dir
import copy
from math import pi
from progress.bar import FillingCirclesBar
import pickle


class Generator() :
    """
    This class implements the generator used to produce samples similar to those 
    in the test set.  It produces SFTs for both detectors along with the 
    corresponding timestamps and frequency range.  The user controls whether a 
    signal is injected or not.
    """
    
    def __init__(self, outdir = 'train_SFT.nosync/', 
                 noisedir = 'train_SFT.nosync/noise/',
                 linedir = 'train_SFT.nosync/line/',
                 linenoisedir = 'train_SFT.nosync/line_noise',
                 signaldir = 'train_SFT.nosync/signal/',
                 oversample_artifacts = False,
                 enable_prints = True) :
        # Writer parameters
        self.outdir = outdir 
        self.noisedir = noisedir
        self.linedir = linedir
        self.linenoisedir = linenoisedir
        self.signaldir = signaldir
        self.oversample_artifacts = oversample_artifacts
        self.enable_prints = enable_prints
        self.detectors = 'H1,L1'
        
        with open('./data.nosync/generator_params.pkl', 'rb') as f :
            generator_params = pickle.load(f)
            
        # Inferred from data exploration 
        self.first_days = generator_params['first_days']
        self.offline_percentage = generator_params['offline_percentage']
        self.offline_duration_params = generator_params['offline_duration_params']
        self.total_duration_params = generator_params['total_duration_params']
        self.frequency_median_params = generator_params['frequency_median_params']
        self.band = generator_params['band']
        self.noise_profile_params = generator_params['noise_profile_params']
        if self.oversample_artifacts : self.noise_profile_params['line'][0] = 0.40
        self.artifact_h0_arr = generator_params['artifact_h0_arr']
        self.NSN_profile_params = generator_params['NSN_profile_params']
        # Hyperparameters
        self.noise_magnitude_params = generator_params['noise_magnitude_params']
        self.D_params = generator_params['D_params']
        self.F1_params = generator_params['F1_params']
        
    
    
    def coin_toss(self, tricked_offline = False, tricked_artifact = False,
                  tricked_noise = False) :
        """
        This function performs the random draws needed for the SFTs generation.
        
        By default : randomly draws a boolean with probability 1/2. 
        
        If tricked_offline is set to True : randomly draws a boolean based on 
        the offline_percentage attribute -> used to randomize if the interferometer 
        is online or offline.
        
        If tricked_artifact is set to True : randomly selects if a narrow instrumental
        artifact is present or not in the signal of 'L1', 'H1' or 'both' detectors.
        
        If tricked_noise is set to True : randomly selects if non stationary noise
        is present or not in the signal of 'L1', 'H1' or 'both' detectors.
        """
        
        if tricked_offline :
            return bool(np.random.choice(np.array([0, 1]), 
                                     p = np.array([1-self.offline_percentage, 
                                                   self.offline_percentage])))
        if tricked_artifact :
            prob1 = self.noise_profile_params['line'][0]
            toss1 = bool(np.random.choice(np.array([0, 1]), p = np.array([1-prob1, prob1])))
            if toss1 :
                prob2 = self.noise_profile_params['line'][1]
                toss2 = bool(np.random.choice(np.array([0, 1]), p = np.array([1-prob2, prob2])))
                if toss2 :
                    return 'both'
                else :
                    return np.random.choice(np.array(['H1', 'L1']))
            else :
                return 'none'
                    
        if tricked_noise :
            prob1 = self.noise_profile_params['NSN'][0]
            toss1 = bool(np.random.choice(np.array([0, 1]), p = np.array([1-prob1, prob1])))
            if toss1 :
                prob2 = self.noise_profile_params['NSN'][1]
                toss2 = bool(np.random.choice(np.array([0, 1]), p = np.array([1-prob2, prob2])))
                if toss2 :
                    return 'both'
                else :
                    return np.random.choice(np.array(['H1', 'L1']))
            else :
                return 'none'
                    
        return bool(np.random.randint(2))
    
    
    
    def get_offline_duration(self) :
        """
        Draw a typical duration for which the interferometer is offline. The 
        distribution was inferred from the provided dataset.
        """
        
        return scipy.stats.gamma.rvs(*self.offline_duration_params)

    
    
    def get_timestamps(self) :
        """
        Draw typical timestamps at which measurements are made. The distribution 
        was inferred from the provided dataset.
        """
        
        timestamps = {}
        
        for interferometer in ['H1', 'L1'] :
            
            # Draw the total duration interferometer is used to generate SFT 
            # (approx 120 days)
            total_duration = scipy.stats.laplace_asymmetric.rvs(*self.total_duration_params)
            total_duration *= 86400 # Going from days to seconds
            
            # Filling time intervals between stamps
            # The following builds the equivalent of np.diff(timestamps)
            stamps_separation = [] # Intervals between stamps
            
            filling = 0 # Intermediate duration of measurements
            while filling < total_duration :
                # Draw if the interferometer is offline 
                if  self.coin_toss(tricked_offline = True) :
                    duration = self.get_offline_duration() # Draw offline duration
                    filling += duration
                    stamps_separation += [duration]
                    
                else :
                    filling += 1800.00 # 30 minutes measurements by default
                    stamps_separation += [1800.00]
                    
            # Builds stamps using first day stamp + successive intervals between stamps
            timestamps[interferometer] = \
                np.concatenate(([np.random.choice(self.first_days, size = 1)[0]], \
                                                          stamps_separation)).cumsum().astype(int)
                          
        return timestamps
        
    
        
    def get_frequency(self) :
        """
        Draw typical frequency medians at which measurements are made. The distribution 
        was inferred from the provided dataset.
        """
        
        return scipy.stats.uniform.rvs(*self.frequency_median_params)
        
    
    
    def get_NSN_profile(self, total_duration) :
        """
        This method gives the non-stationary noise segments durations, as well as
        the relative difference in noise amplitude. The distribution was inferred 
        from the provided dataset.
        """
        
        segments_duration = []
        noise_relative_magnitudes = []
        
        current_duration = 0
        i = 0 
        # Alternate peaks of higher magnitude noises with off-peaks of
        # lower magnitude noises
        while current_duration < total_duration :
            if i%2 == 0 : # off-peak
                duration = np.abs(stats.gamma.rvs(*self.NSN_profile_params['off_peak_length']))
                current_duration += duration
                segments_duration += [duration]
                noise_relative_magnitudes += \
                [stats.exponnorm.rvs(*self.NSN_profile_params['off_peak_mag'])]
                i += 1
            else : # peak
                duration = np.abs(stats.johnsonsu.rvs(*self.NSN_profile_params['peak_length']))
                current_duration += duration
                segments_duration += [duration]
                noise_relative_magnitudes += [stats.moyal.rvs(*self.NSN_profile_params['peak_mag'])]
                i += 1
                
        segments_duration = np.array(segments_duration)
        noise_relative_magnitudes = np.array(noise_relative_magnitudes)
        assert segments_duration.shape[0] == noise_relative_magnitudes.shape[0]
        
        return segments_duration, noise_relative_magnitudes
    
    
    
    def split_timestamps(self, ts) :
        """
        This function splits the time frame over which the measurements are made 
        and provides a slicing of the timestamps array so as to apply non-stationary 
        noise to the data afterwards.
        """
        
        total_duration = ts[-1] - ts[0]
        assert total_duration > 0
        segments_duration, noise_relative_magnitudes = self.get_NSN_profile(total_duration)
        
        durations = np.diff(ts) # Time intervals 
        
        index = [0] # Will hold indexes for numpy.split
        nxt = 0 # Index for next split
        for slice_size in segments_duration :
            if len(durations) == 0 :
                break
            current_size = 0 # Temporary variable 
            while current_size < slice_size  :
                if len(durations) != 0 :
                    current_size += 1.0 * durations[0]
                    durations = copy.deepcopy(durations[1:]) # Reducing remaining durations array
                    nxt += 1
                else : 
                    break
            index += [nxt]
            
        index = index[1:-1] # numpy.split doesn't take the first and last index in its input
        
        # Uneven splitting of timestamps based on built index array
        split_timestamps = np.split(ts, np.array(index))
        noise_relative_magnitudes = noise_relative_magnitudes[:len(split_timestamps)]

        assert len(split_timestamps) == noise_relative_magnitudes.shape[0]
        return split_timestamps, noise_relative_magnitudes 
    
    
    
    def get_line_parameters(self, frequencies, mean_sqrtSX) :
        """
        Draw typical parameters for which a narrow instrumental artifact is present. 
        The distribution was inferred from the provided dataset.
        """
        
        F_line = np.random.choice(frequencies)
        h0 = mean_sqrtSX * np.random.uniform(*self.artifact_h0_arr)
        phi = np.random.uniform(0, 2*pi)
        return F_line, h0, phi
    
    
    
    def  get_noise_profile(self) :
        """
        This method draws the noise profile of the SFT generation on both detectors.
        It determines if the SFT has non stationary noise or if a narrow instrumental 
        artifact is present. The distribution was inferred from the provided dataset.
        """
        
        noise_profile = {'H1': {}, 'L1': {}}
        
        NSN_result = self.coin_toss(tricked_noise = True)
        noise_profile['H1']['NSN'] = False
        noise_profile['L1']['NSN'] = False
        if NSN_result == 'both' :
            noise_profile['H1']['NSN'] = True
            noise_profile['L1']['NSN'] = True
        elif NSN_result != 'none' :
            noise_profile[NSN_result]['NSN'] = True
        
        line_result = self.coin_toss(tricked_artifact = True)
        noise_profile['H1']['line'] = False
        noise_profile['L1']['line'] = False
        if line_result == 'both' :
            noise_profile['H1']['line'] = True
            noise_profile['L1']['line'] = True
        elif line_result != 'none' :
            noise_profile[line_result]['line'] = True

        return noise_profile
    
           

    def generate_noise_stationary(self, interferometer, F0, timestamps) :
        """
        This method uses the PyFstat library to generate a stationary noise SFT.
        """
        
        # Getting noise magnitudes 
        sqrtSX = stats.laplace_asymmetric.rvs(*self.noise_magnitude_params)
        sqrtSX *= 0.0070710678118654745 # Scaling inferred from dataset
        
        # Setup Writer
        writer_kwargs = {
            'label': 'stationary',
            'outdir': self.noisedir,
            'timestamps': timestamps,
            'detectors': interferometer,
            'F0': F0,  
            'Band': self.band,  
            'sqrtSX': sqrtSX,  
            'Tsft': 1800,  
            'SFTWindowType': 'tukey',  
            'SFTWindowBeta': 0.01,  
        }
        writer = pyfstat.Writer(**writer_kwargs)
        # Create SFT
        if self.enable_prints : print('Generating stationary noise SFT...')
        writer.make_data()
        sftfilepath = writer.sftfilepath
        
        frequencies, timestamps1, fourier_data = get_sft_as_arrays(sftfilepath)
        fourier_data = fourier_data[interferometer]
        
        assert ((timestamps==timestamps1[interferometer]).all())
        return fourier_data, frequencies, sqrtSX
    
    
    
    def generate_noise_non_stationary(self, interferometer, F0, timestamps) :
        """
        This method uses the PyFstat library to generate a non stationary noise 
        SFT.
        """

        segmented_timestamps, noise_relative_magnitudes = self.split_timestamps(timestamps)
        
        
        # Getting noise magnitudes 
        base_magnitude = stats.laplace_asymmetric.rvs(*self.noise_magnitude_params)
        base_magnitude = 0.0070710678118654745 * base_magnitude # Scaling inferred from dataset
        segment_sqrtSX = base_magnitude * noise_relative_magnitudes

        sft_path = []
        # Setup Writer
        writer_kwargs = {
            'outdir': self.noisedir,  
            'F0': F0,  
            'Band': self.band,  
            'Tsft': 1800, 
            'SFTWindowType': 'tukey',
            'SFTWindowBeta': 0.01,
            'detectors': interferometer
        }
        
        # Progress bar
        mx = len(segmented_timestamps)
        if self.enable_prints :
            bar = FillingCirclesBar('Generating non stationary noise SFT', max = mx, \
                                    suffix = '%(percent).1f%% - %(eta)ds')
        
        # Non-stationary noise implementation
        for segment in range(mx) :
            writer_kwargs['label'] = f'segment_{segment}'
            writer_kwargs['timestamps'] = segmented_timestamps[segment]
            writer_kwargs['sqrtSX'] = segment_sqrtSX[segment]
    
        
            writer = pyfstat.Writer(**writer_kwargs)
            writer.make_data()
        
            sftfilepath = writer.sftfilepath
            sft_path.append(sftfilepath)
            
            if self.enable_prints : bar.next()
        
        sftfilepath = ";".join(sft_path)  # Concatenate different files using ;
        if self.enable_prints : bar.finish()
        
        frequencies, timestamps1, fourier_data = get_sft_as_arrays(sftfilepath)
        fourier_data = fourier_data[interferometer]
        
        # Adding top-up noise
        general_noise = np.random.normal(loc = np.mean(np.real(fourier_data)), 
                                         scale = 1.10 * np.std(np.real(fourier_data)), 
                                         size = fourier_data.shape).astype(np.complex64) + \
            1j * np.random.normal(loc = np.mean(np.imag(fourier_data)), 
                                             scale = 1.10 * np.std(np.imag(fourier_data)), 
                                             size = fourier_data.shape).astype(np.complex64)
            
        assert fourier_data.dtype == general_noise.dtype
        fourier_data += general_noise
        
        assert ((timestamps==timestamps1[interferometer]).all())
        return fourier_data, frequencies, np.mean(segment_sqrtSX)
    
    
    
    def generate_line(self, frequencies, mean_sqrtSX, interferometer, timestamps) :
        """
        This method uses the PyFstat library to generate a narrow instrumental 
        artifact SFT.
        """
        
        F_line, h0, phi = self.get_line_parameters(frequencies, mean_sqrtSX)
        
        writer_kwargs = {
            'label' : 'line_noise',
            'outdir': self.linenoisedir,
            'F0': F_line, 
            'timestamps': timestamps,
            'detectors': interferometer,
            'sqrtSX': 0,
            'Band': 720 * (frequencies[1] - frequencies[0]), # Extended band
            'Tsft': 1800,  
            'SFTWindowType': 'tukey',
            'SFTWindowBeta': 0.01
        }
        writer = pyfstat.Writer(**writer_kwargs)
        if self.enable_prints : print('Pre-processing narrow instrumental artifact...')
        writer.make_data()
        noisepath = writer.sftfilepath
        
        writer_kwargs = {
            'label' : 'line',
            'outdir': self.linedir,
            'F0': F_line,  
            'phi': phi,
            'h0': h0,
            'detectors': interferometer,
            'sqrtSX': 0,
            'Band': 720 * (frequencies[1] - frequencies[0]), # Extended band
            'Tsft': 1800,  
            'SFTWindowType': 'tukey',
            'SFTWindowBeta': 0.01,
            'noiseSFTs': noisepath
        }
        
        writer = pyfstat.LineWriter(**writer_kwargs)
        if self.enable_prints : print('Generating narrow instrumental artifact...')
        writer.make_data()
        sftfilepath = writer.sftfilepath
        
        # Extended data -> slicing back to initial frequency range
        frequencies_ext, timestamps_ext, fourier_data_ext = get_sft_as_arrays(sftfilepath)
        fourier_data_ext = fourier_data_ext[interferometer]
        
        F_middle = frequencies[179]
        try :
            indx_middle_ext = np.where(frequencies_ext == F_middle)[0][0]
        except :
            indx_middle_ext = np.argmin(np.abs(frequencies_ext - F_middle))
        
        
        fourier_data_ext = fourier_data_ext[indx_middle_ext-179:indx_middle_ext+181]
        frequencies_ext = frequencies_ext[indx_middle_ext-179:indx_middle_ext+181]
        
            
        assert ((np.abs(frequencies_ext-frequencies)<1e-12).all())
        assert fourier_data_ext.shape[0] == 360
        assert ((timestamps_ext[interferometer]==timestamps).all())
        
        return fourier_data_ext    
        
    
    
    def generate_noise_distri(self, interferometer, F0, timestamps, with_line, with_NSN) :
        """
        This helper method triggers the correct noise generation functions, based 
        on the noise profile drawn in the generate_noise_sample method.
        """
        
        if with_NSN :
            fourier_data, frequencies, mean_sqrtSX = self.generate_noise_non_stationary(interferometer, 
                                                                                        F0, timestamps)
        else :
            fourier_data, frequencies, mean_sqrtSX = self.generate_noise_stationary(interferometer, 
                                                                                    F0, timestamps)
        if with_line :
            fourier_data += self.generate_line(frequencies, mean_sqrtSX, interferometer, timestamps)
            
        return fourier_data, frequencies, mean_sqrtSX



    def generate_noise_sample(self) :
        """
        This method generates noise SFT samples for the detectors H1 and L1. It
        returns a list of arrays organized in the same way as what is returned by
        the hdf5 reading utility function.
        """
        
        # Getting frequency and timestamps
        timestamps = self.get_timestamps() 
        F0 = self.get_frequency()
        # Getting noise profile 
        noise_profile = self.get_noise_profile()
        
        # Returning a list of arrays matching hdf5 reading format
        SFTs = ['Fourier H1', 'Timestamps H1', 'Fourier L1', 'Timestamps L1', 'Frequencies']
        SFTs[1] = timestamps['H1']
        SFTs[3] = timestamps['L1']
        
        # Sanity check
        frequencies_storage = []
        # For SFTs afterwards
        fourier_storage = []
        # For signal generation if present
        sqrtSX_storage = []
        
        for interferometer in ['H1', 'L1'] :
            
            if self.enable_prints :
                print('')
                print('Generating noise for ' + interferometer + '...')
                print('Cleaning previous noise data...')
            clean_dir(self.noisedir) 
            if self.enable_prints : print('Cleaning previous line data...')
            clean_dir(self.linedir) 
            clean_dir(self.linenoisedir)
            
            with_line =  noise_profile[interferometer]['line']
            with_NSN = noise_profile[interferometer]['NSN']
            if self.enable_prints :
                print('--- Non stationary noise : ' + str(with_NSN) +\
                      ' | Line artifact : ' + str(with_line) + ' ---')
            
            int_timestamps = timestamps[interferometer]
            fourier_data, frequencies, mean_sqrtSX = self.generate_noise_distri(interferometer,  
                                                                   F0, int_timestamps, 
                                                                   with_line, with_NSN)
            frequencies_storage += [copy.deepcopy(frequencies)]
            fourier_storage += [fourier_data]
            sqrtSX_storage += [mean_sqrtSX]
        
        
        assert (frequencies_storage[0]==frequencies_storage[1]).all()
        SFTs[0] = fourier_storage[0]
        SFTs[2] = fourier_storage[1]
        SFTs[4] = frequencies_storage[0]
        return SFTs, sqrtSX_storage
    
    
    
    def gaussian_choice(self, frequencies) :
        """
        This method performs a random choice based on a discrete distribution 
        having a Gaussian 'shape'. It chooses F0 closer to the center of noise.
        Credits : ayhan - Stack Overflow
        """

        x = np.arange(-180, 180)
        xU, xL = x + 0.5, x - 0.5 
        prob = stats.norm.cdf(xU, scale = 75) - stats.norm.cdf(xL, scale = 75)
        prob = prob / prob.sum() 
        
        F0 = np.random.choice(frequencies, p = prob)
        
        return F0
    
    
    
    def get_CW_parameters(self, frequencies, sqrtSX) :
        """
        Draw typical parameters for which a signal is present. 
        """
        
        # Higher probability of having signal close to the center
        F_CW = self.gaussian_choice(frequencies) 
        F1 = -1. * np.random.uniform(*self.F1_params)
        return F_CW, F1
        
    
        
    def generate_signal(self, interferometer, frequencies, timestamps, sqrtSX,
                        F_CW, F1, CW_params) :
        """
        This method generates a simulated continuous gravitational-wave signal (CW) 
        SFT.
        """
        
        # h0 = noise magnitude / D
        D = np.random.uniform(*self.D_params)
        
        # Writer params
        writer_kwargs = {
            'outdir': self.signaldir,
            'Band' : 720 * (frequencies[1] - frequencies[0]), # Extended band,
            'timestamps': timestamps,
            'detectors': interferometer,
            'h0': sqrtSX / D,
            'tref': timestamps[0],
            'sqrtSX': 0., # Noise will be added afterwards
            'Tsft': 1800,
            'SFTWindowType': "tukey",
            'SFTWindowBeta': 0.01,
        }
        
        writer = pyfstat.Writer(**writer_kwargs, **CW_params)
        writer.make_data()
        sftfilepath = writer.sftfilepath
        
        frequencies_ext, timestamps_ext, fourier_data_ext = get_sft_as_arrays(sftfilepath)
        fourier_data_ext = fourier_data_ext[interferometer]
        
        # Extended data -> slicing back to initial frequency range
        F_middle = frequencies[179]
        try :
            indx_middle_ext = np.where(frequencies_ext == F_middle)[0][0]
        except :
            indx_middle_ext = np.argmin(np.abs(frequencies_ext - F_middle))
        
        
        fourier_data_ext = fourier_data_ext[indx_middle_ext-179:indx_middle_ext+181]
        frequencies_ext = frequencies_ext[indx_middle_ext-179:indx_middle_ext+181]
        
            
        assert ((np.abs(frequencies_ext-frequencies)<1e-12).all())
        assert fourier_data_ext.shape[0] == 360
        assert ((timestamps_ext[interferometer]==timestamps).all())
        
        return fourier_data_ext, D
        
        
        
    def add_signal(self, SFTs, sqrtSX) :
        """
        This method adds a simulated continuous gravitational-wave signal (CW) to
        the provided noise data. 
        """
        
        if self.enable_prints :
            print('')
            print('Adding continuous gravitational-wave signal')
            print('Cleaning previous signal data...')
        clean_dir(self.signaldir) 
        
        frequencies =  SFTs[4]
        
        # Randomly selecting parameters
        F_CW, F1 = self.get_CW_parameters(frequencies, sqrtSX)
        signal_parameters_generator = pyfstat.AllSkyInjectionParametersGenerator(
            priors={
                'F0': F_CW,
                'F1': F1,
                'F2': 0.,
                **pyfstat.injection_parameters.isotropic_amplitude_priors
            },
        )
        CW_params = signal_parameters_generator.draw()
        
        D_storage = ['H1', 'L1']
        
        for indx, interferometer, sqrtSX_indx in zip([0, 2],['H1', 'L1'], [0,1]) :
            if self.enable_prints : print('Injecting CW signal for ' + interferometer + '...')
            int_timestamps = SFTs[indx+1]
            signal, D = self.generate_signal(interferometer, frequencies, 
                                          int_timestamps, 
                                          sqrtSX[sqrtSX_indx], F_CW, F1, CW_params)
            SFTs[indx] += signal
            D_storage[sqrtSX_indx] = D
            if self.enable_prints : print('CW added with D = ' + '{:.3f}'.format(D))
            
        return SFTs, D_storage
    
    
    
    def generate_sample(self, with_signal = None) :
        """
        Master method. This method generates fake SFTs suitable for training. 
        If with_signal is set to True, a continuous gravitational wave signal 
        will be present in the supplied SFTs. If with_signal is None, the signal
        will be injected with probability 0.5.
        """
        
        if with_signal is None :
            with_signal = self.coin_toss()
            
        D_storage = None
        
        if self.enable_prints :
            print('')
            print('-------------- ' + 
                  'Generating sample | With signal : ' + str(with_signal) 
                  + ' --------------')
        
        success = False
        attemps = 0
        while not success :
            attemps += 1
            if attemps > 10 : raise Exception('Unable to generate noise')
            try : 
                SFTs, sqrtSX_storage = self.generate_noise_sample()
                success = True
            except :
                if self.enable_prints :
                    print('')
                    print('-------------------------------------')
                    print('Failed to generate noise - Restarting')
                    print('-------------------------------------')
                    print('')
        
        if with_signal :
            success = False
            attemps = 0
            while not success :
                attemps += 1
                if attemps > 10 : raise Exception('Unable to generate signal')
                try : 
                    SFTs, D_storage = self.add_signal(SFTs, sqrtSX_storage)
                    success = True
                except : 
                    if self.enable_prints :
                        print('')
                        print('-------------------------------------')
                        print('Failed to add signal - Restarting')
                        print('-------------------------------------')
                        print('')
                        
        if self.enable_prints :
            print('')
            print('--------------------------- Sample generated ---------------------------')
            print('')
            print('')
            print('')
            
        return SFTs, with_signal, D_storage
        
        
  
if __name__ == '__main__' :
    SFTs, with_signal = Generator().generate_sample()


