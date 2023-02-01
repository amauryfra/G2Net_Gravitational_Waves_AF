
# G2Net Gravitational Wave Detection - Kaggle Competition
This project contains a solution proposed for solving the [_G2Net Detecting Continuous Gravitational Waves_](https://www.kaggle.com/competitions/g2net-detecting-continuous-gravitational-waves/overview) Kaggle competition. The challenge is to perform a classification task on data provided by the __Laser Interferometer Gravitational-Wave Observatory__ (LIGO). In particular, the goal is to find [continuous gravitational-wave signals](https://www.ligo.org/science/GW-Continuous.php) hidden in noisy data. Said continuous gravitational waves are between 1 and 2 orders of magnitude smaller than the amplitude of the noise.

Only a small set of training data is provided, the goal for competitors being to generate their own labeled dataset using the physics-informed `PyFstat` library. These data samples are in the form of __Short Fourier Transforms__ (SFTs). Said Fourier transforms are given by a 2-dimensional complex `NumPy` array. A set of unlabeled test samples is provided, the goal is to maximize the area under the ROC curve, between predicted probabilities of signal presence and the ground truth target (unknown from competitors).

## Methodology
A `PyFstat`-based SFT generator is first built, inferring the input parameters distribution from a thorough exploration of the unlabeled test set. Subsequently, this data generator is used to produce a medium size dataset on which a CNN classifier is trained (_InceptionResNetV2_ [[1]](#1)). By excluding the top layers and max-pooling, said CNN is then used to extract a vector of meaningful features from the SFTs. Finally, a __continual learning__ approach is used : SFTs are continuously generated and their CNN-extracted features are fed to a __Gated Linear Network__ (GLN) [[2]](#2). The GLN is updated online for each generated sample that is passed as _side information_. The obtained CNN + GLN chain of models is consequently used to perform predictions on the test set.  

#### SFT Generator

The `PyFstat` library allows to generate artificial noisy signals similar to those produced by the LIGO interferometers. The noise and signal generation requires a set of problem-specific input parameters. As the objective is to supply our models with data comparable to those of the test set, we performed data exploration, inferring the distribution of said parameters from the provided SFTs using the `fitter` library. All the useful parameters are stored in the `data.nosync/generator_params.pkl` dictionary.

#### CNN Feature Extractor

The InceptionResNetV2 model is used via its `TensorFlow` implementation. The generated SFTs are pre-processed using a moving-average denoising procedure. The 2 classes are balanced in the training set. Binary cross-entropy is minimized, using an exponential learning rate decay schedule and performing early stopping. Subsequently, a feature vector is obtained as the top layers are discarded and max-pooling is applied.

#### GLN Classifier    

The GLN model is implemented via the `pygln` library, using a `PyTorch` backend. The source code has been modified to enable saving the model to disk using object serialization through `pickle`. A hyperparameter grid search has been performed to determine the optimal GLN architecture in this context. The CNN feature vector output is used as side information for the GLN, which has no base model.

## Project structure
<ins>Data Generation</ins> :
* The full data generator is implemented in the `generator.py` file. The `Generator` class contains all methods necessary to generate SFT samples similar to those of the test set. In particular the `Generator.generate_sample` method produces SFTs, timestamps, and a frequency range for both LIGO interferometers.
* The `dataset_generator.py` file contains a set of functions for generating full-size datasets, directly usable for training the CNN model.

<ins>CNN Training</ins> :
* The full training process of the InceptionResNetV2 model is implemented in the `train_CNN.py` script.
* The feature vector is computed by the functions available in the `feature_extractor.py` file, which loads weights and performs forward passes through the pre-trained CNN.

<ins>GLN Continual Training</ins> :
* The GLN classifier model is wrapped up in the `Detector` class defined in `detector.py`. The `Detector.train` method is used to trigger an online update by providing an SFT and its target class.  
* The `Tuner` class implements a hyperparameter grid-search in `tuner.py`. The `Tuner.hyperparameter_search` method saves the obtained results in the `data.nosync/hyperparameters_results.npy` dictionary.
* The `training.py` file provides the `Trainer` class from which continual learning can be performed through the `Trainer.train_session` method.

<ins>Utilities</ins> :
* General utilities are available in `utils.py`.
* `test.py` is a testing script to assess the generated data samples quality, the pre-trained CNN model's accuracy, and the pre-trained GLN model's accuracy.
* The `legacy.py` file contains all deprecated functions used throughout the project.


## Usage

To start working on the project first create a virtual environment, then clone the repository and install all dependencies by running :

```
$ git clone https://github.com/amauryfra/G2Net_Gravitational_Waves_AF.git
$ cd G2Net_Gravitational_Waves_AF
$ python3.8 pip install -r requirements.txt
```

To initiate continual learning run :

```
$ python3.8 training.py
```

## Credits

This project has leveraged the useful GLN implementation provided by the `aiwabdn/pygln` repository to develop the code.

## References
<a id="1">[1]</a>
Szegedy, C., Ioffe, S., Vanhoucke, V., & Alemi, A. A. (2017, February). Inception-v4, inception-resnet and the impact of residual connections on learning. In Thirty-first AAAI conference on artificial intelligence.

<a id="2">[2]</a>
Veness, J., Lattimore, T., Budden, D., Bhoopchand, A., Mattern, C., Grabska-Barwinska, A., ... & Hutter, M. (2021, May). Gated linear networks. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 35, No. 11, pp. 10015-10023).
