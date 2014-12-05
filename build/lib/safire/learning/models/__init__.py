#!/usr/bin/env python

"""
This package directly contains implementations of various nnet safire models.
"""

# Bringing the models into the namespace, gensim-stype, to save some typing.
#from .base_model import BaseModel
#from .base_unsupervised_model import BaseUnsupervisedModel
#from .base_supervised_model import BaseSupervisedModel
#from .base_pretrained_supervised_model import BasePretrainedSupervisedModel

from .logistic_regression import LogisticRegression
from .logistic_regression_classifier import LogisticRegressionClassifier
from .hidden_layer import HiddenLayer
from .multilayer_perceptron import MultilayerPerceptron
from .autoencoder import Autoencoder
from .denoising_autoencoder import DenoisingAutoencoder
from .sparse_denoising_autoencoder import SparseDenoisingAutoencoder
from .restricted_boltzmann_machine import RestrictedBoltzmannMachine
from .replicated_softmax import ReplicatedSoftmax


def check_model_dataset_compatibility(dataset, model_class):
    from safire.learning.models.base_supervised_model import BaseSupervisedModel
    if issubclass(model_class, BaseSupervisedModel):
        if dataset.mode == 0:
            raise ValueError('Cannot run supervised training with unsupervised'+
                             ' dataset mode. (Model %s, dataset mode %s' %
                             (str(model_class), str(dataset.mode)))