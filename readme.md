This repository contains all the code to reproduce the results:
* main.py --- The basic file to train the neural networks, including Resnet18, VGG16-BN, and Densenet121.
* create_data_subset.py --- The helper file to create data subsets with equal number of examples for different categories and corrupted labels if necessary.
* data.py --- The dataloader file with corrupted features if necessary.
* csr.py --- The approach of LASS and FASS to estimate the critical sample ratio.
* custom_hurst.py --- Hurst exponent estimator that is slightly adapted from the Python package hurst.
* id_measures.py --- Various methods to estimate the intrinsic dimension of the hidden representation, including MIND, MLE, GeoMLE, ED and more.
* intrinsic_dimension.py --- Including the intrinsic dimension estimator, TwoNN.
* multi-nlid.py --- Evaluating the intrinsic dimension of all ReLU layers of neural networks from the first epoch to the last epoch.
* toy.py --- The two-layer neural network to classify FashionMNIST and KMNIST. The metrics such as sign consistency, correlation level, and Fisher information matrix are evaluated therein.
* config.yml --- Python package requirements necessary to reproduce all these results.
* run.sh --- Shell file to run with different configuration of model architecture, optimizer, data set, and noisy feature (label).
* grid-search.sh --- Shell file to run with different choice of learning rate and batch size to exhibit the transition from compression to expansion.

It is noteworthy that different from the standard Pytorch implementations of Resnet18, VGG16-BN, and Densenet121, we made a slight modification by changing the structure of Conv-BN-ReLU-Conv to Conv-ReLU-BN-Conv.