import os
import pickle
import warnings

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import keras.backend as K
from keras.callbacks import Callback
from keras.models import Model, load_model
from keras.layers import LSTM, Input, Lambda, Dense, TimeDistributed
from keras.layers.merge import Concatenate
from keras.engine.training import _standardize_sample_weights
from keras.optimizers import Nadam

import metrics
from data_utils import pad_sequence_into_array, to_categorical


class BlstmCCE(object):
    """
    Implements BLSTM RNN with Categorical Crossentropy loss function in the user-friendly way.
    """

    def __init__(self, 
                 nb_feat=34, 
                 nb_class=4, 
                 max_timesteps=78, 
                 accuracies=True, 
                 optimizer="Nadam", 
                 modelname=None, 
                 modelpath="."):
        """
        Constructor

        Args:
            nb_feat(int): dimension of feature space
            nb_class(int): number of classes
            max_timesteps(int): length of the input sequences of features in time dimension
            accuracies(bool): whether to trace accuracies during the training phase or not
            optimizer(str or keras.optimizer): optimizer to use during training
            modelname(str): if this argument is passed then model will be loaded from the files
            modelpath(str): path where to find model in case of loading from file
            
        Return:
            None
        """

        if modelname is not None:
            if not os.path.exists(modelpath):
                raise NameError("No such path!")

            model_file = os.path.join(modelpath, modelname + ".model")

            if not os.path.exists(model_file):
                raise NameError("No model file!")

            self._model = load_model(filepath=model_file, 
                                     custom_objects={"weighted_accuracy_ohe_tf": metrics.weighted_accuracy_ohe_tf, 
                                                     "unweighted_accuracy_ohe_tf": metrics.unweighted_accuracy_ohe_tf})
            
            # Nadam default optimizer
            self.optimizer = self._model.optimizer

            # reconstruction of class parameters from loaded network
            self.nb_feat = self._model.get_layer("data").input_shape[2]
            self.nb_class = self._model.get_layer("softmax").output_shape[1]
            self.max_timesteps = self._model.get_layer("data").input_shape[1]
            self.accuracies = len(self._model.metrics) > 0
            
            if self.accuracies:
                self._model.metrics_names = ["loss", "wa", "ua"]

        else:

            self.nb_feat = nb_feat
            self.nb_class = nb_class
            self.max_timesteps = max_timesteps
            self.accuracies = accuracies
            self.optimizer = optimizer

            # building model from scratch
            self._model = _build_cce_model(nb_feat=self.nb_feat, 
                                           nb_class=self.nb_class, 
                                           max_timesteps=self.max_timesteps, 
                                           accuracies=self.accuracies, 
                                           optimizer=self.optimizer)


    def fit(self, X, y, 
            batch_size=64, epochs=10, verbose=1, callbacks=[], shuffle=True, 
            validation_split=0., validation_data=None, validation_random_state=None, 
            sample_weight=None):
        """
        Fit already built CTC model

        Args:
            X(ndarray): data; array of lists (or n-dim array) which will be truncated/padded to the
                max_timesteps length and casted into one n-dim array
            y(ndarray): labels; array of lists (or n-dim array) which will be truncated/padded to the
                max_label_sequence_length length and casted into one n-dim array
            batch_size(int): size of a mini-batch for training
            epochs(int): numberof epochs to train
            verbose(int): level of output delatization during training
            callbacks(list): list of Keras callbacks to apply
            shuffle(bool): whether to shuffle samples during training
            validation_split(float): percentage of all data that will be separated into validation set
            validation_data(tuple): validation data and labels
            validation_random_state(int): random seed to use for train validation split
            sample_weight(ndarray): array of the same length as x, containing
                weights to apply to the model's loss for each sample.
        
        Return:
            hist(keras.callbacks.History): history of a training
        """

        # pad/truncate initial data and generate masks for them to take into account only really existing points
        X_train, _ = pad_sequence_into_array(X, maxlen=self.max_timesteps, padding="pre")
        y_train = to_categorical(y, classes=range(self.nb_class))
        sample_weight_train = [sample_weight]

        if validation_data:
            # if validation data is provided then just take them and transform to the input format of CTC network
            X_val, y_val = validation_data
            # pad/truncate validation data and generate masks for them to take into account only really existing points
            X_val, _ = pad_sequence_into_array(X_val, maxlen=self.max_timesteps)
            y_val = to_categorical(y_val, classes=range(self.nb_class))

            validation_data = (X_val, y_val)
        elif validation_split and 0. < validation_split < 1.:
            # if validation data is not provided but we still want to do a validation we divide data randomly
            # divide indexes
            idxs_train, idxs_val = train_test_split(np.arange(X_train.shape[0]), 
                                                    test_size=validation_split, 
                                                    random_state=validation_random_state)
            # divide data correspondingly to indexes
            X_train, X_val = X_train[idxs_train], X_train[idxs_val]
            y_train, y_val = y_train[idxs_train], y_train[idxs_val]

            if sample_weight_train[0] is not None:
                sample_weight_train[0] = sample_weight_train[0][idxs_train]
            
            validation_data = (X_val, y_val)
        else:
            # do no validation
            validation_data = None

        # recalculate metrics that are datasetwise (e.g. unweughted accuracy)
        if validation_data is not None:
            val_metrics_update = ValidationMetricsCorrectionCallback()
            callbacks_new = callbacks + [val_metrics_update]
        else:
            callbacks_new = callbacks

        # start fitting
        return self._model.fit(X_train, y_train, 
                               batch_size=batch_size, epochs=epochs, verbose=verbose, 
                               callbacks=callbacks_new, 
                               validation_split=0., validation_data=validation_data, 
                               shuffle=shuffle, sample_weight=sample_weight_train)

    def predict(self, X, batch_size=64, verbose=0, return_probas=False):
        """
        Do a prediction for X

        Args:
            X(ndarray):  data to predict; array of lists (or n-dim array) which will be truncated/padded to the
                max_timesteps length and casted into one n-dim array
            batch_size(int): mini-batch size for prediction (currently not supported)
            verbose(int): level of output delatization during prediction process
        Return:
            y(ndarray): predictions
        """

        # pad/truncate initial data and generate masks for them to account only for really existing points
        X, _ = pad_sequence_into_array(X, maxlen=self.max_timesteps)

        y_pred = self._model.predict(X, batch_size=batch_size, verbose=verbose)

        if return_probas:
            return np.argmax(y_pred, axis=1), y_pred
        else:
            return np.argmax(y_pred, axis=1)
    
    def evaluate(self, X, y, verbose=0):
        """
        """

        X, _ = pad_sequence_into_array(X, maxlen=self.max_timesteps)
        y = to_categorical(y, classes=range(self.nb_class))

        results = self._model.evaluate(X, y, batch_size=X.shape[0], verbose=verbose)
        
        return results

    def summary(self):
        """
        Print the summary of a model
        
        Args:
            None
        
        Return:
            None
        """
        self._model.summary()
    
    def save(self, modelname, path="."):
        """ Save model architecture, weights and optimizer parameters

        Args:
            modelname(str): name of the model
            modelpath(str): path to save
        """
        if not os.path.exists(path):
            raise NameError("No such path!")
        model_file = os.path.join(path, modelname + ".model")

        self._model.save(model_file, overwrite=True)


class ValidationMetricsCorrectionCallback(Callback):
    
    def __init__(self, **kwargs):
        super(ValidationMetricsCorrectionCallback, self).__init__(**kwargs)
        
    def on_train_begin(self, logs={}):
        self.accuracies = len(self.model.metrics) > 0
    
    def on_epoch_end(self, epoch, logs={}):
        if self.accuracies and self.validation_data is not None:
            losses = self.model.evaluate(self.validation_data[0], self.validation_data[1], 
                                         batch_size=self.validation_data[0].shape[0], 
                                         verbose=0)
            losses = dict(zip(["loss", "wa", "ua"], losses))
            for l in losses.items():
                logs["val_" + l[0]] = np.mean(l[1], dtype=type(logs["val_" + l[0]]))


def _build_cce_model(nb_feat, 
                     nb_class, 
                     max_timesteps, 
                     accuracies=True, 
                     optimizer="Nadam"):
    """
    Function to build ctc model. At the moment Keras does not support parametric loss functions and CTC need masks.
    Thus CTC objective is implemented through the Lambda layers. Also some accuracy computations available through
    the same technique. Model should have "input", "softmax" layers to be able to work.

    Args:
        nb_feat(int): feature space dimension
        nb_class(int): number of classes
        max_timesteps(int): maximum number of timesteps considered
        max_label_sequence_length(int): maximum length of the label sequence
        optimizer(function ort string): optimizer to use during training
        accuracies(bool): whether to return accuracies

    Return:
        model(Model): compiled model
    """

    # input layer
    data = Input(name="data", shape=(max_timesteps, nb_feat))

    # BLSTM block 1
    forward_lstm1 = LSTM(units=64,
                         return_sequences=True
                         )(data)
    backward_lstm1 = LSTM(units=64,
                          return_sequences=True, 
                          go_backwards=True
                          )(data)
    blstm_output1 = Concatenate()([forward_lstm1, backward_lstm1])

    # BLSTM block 2
    forward_lstm2 = LSTM(units=64,
                         return_sequences=False
                         )(blstm_output1)
    backward_lstm2 = LSTM(units=64,
                          return_sequences=False,
                          go_backwards=True
                          )(blstm_output1)
    blstm_output2 = Concatenate(axis=-1)([forward_lstm2, backward_lstm2])

    # Dense classification layers
    hidden = Dense(512, activation="relu")(blstm_output2)

    # Softmax layer which accounts fot NULL class introduced in CTC
    softmax = Dense(nb_class, activation="softmax", name="softmax")(hidden)

    # Construct different outputs with and without accuracies
    if accuracies:
        model_metrics = [metrics.weighted_accuracy_ohe_tf, metrics.unweighted_accuracy_ohe_tf]
    else:
        model_metrics = []

    # Build final model and compile it
    model = Model(inputs=[data], outputs=[softmax])
    model.compile(loss="categorical_crossentropy", 
                  optimizer=optimizer, 
                  metrics=model_metrics)
    model.metrics_names = ["loss", "wa", "ua"]
    return model
