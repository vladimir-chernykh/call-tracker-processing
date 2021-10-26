import os
import pickle
import warnings

import numpy as np
from sklearn.model_selection import train_test_split

import keras.backend as K
from keras.callbacks import Callback
from keras.models import Model, load_model
from keras.layers import LSTM, Input, Lambda, Dense, TimeDistributed
from keras.layers.merge import Concatenate
from keras.optimizers import Nadam

from . import metrics
from .data_utils import pad_sequence_into_array


class BlstmCTC(object):
    """
    Implements BLSTM RNN with CTC loss function in the user-friendly way.
    """

    def __init__(self,
                 nb_feat=34,
                 nb_class=4,
                 max_timesteps=78,
                 max_label_sequence_length=1, 
                 accuracies=True,
                 beam_width=1,
                 optimizer="Nadam",
                 modelname=None, 
                 modelpath="."):
        """
        Constructor

        Args:
            nb_feat(int): dimension of feature space
            nb_class(int): number of classes
            max_timesteps(int): length of the input sequences of features in time dimension
            max_label_sequence_length(int): length of the output sequences of labels (currently only 1 is supported)
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
                                     custom_objects={"metrics": metrics, 
                                                     "_loss_placeholder": _loss_placeholder})
            # Nadam default optimizer
            self.optimizer = self._model.optimizer

            # reconstruction of class parameters from loaded network
            self.nb_feat = self._model.get_layer("data").input_shape[2]
            self.nb_class = self._model.get_layer("softmax").output_shape[2] - 1
            self.max_timesteps = self._model.get_layer("data").input_shape[1]
            self.max_label_sequence_length = self._model.get_layer("labels").input_shape[1]

            preds_layer = self._model.get_layer("preds")
            if preds_layer is not None:
                self.accuracies = True
                self._model.metrics_names = ["loss", "ctc", "wa", "ua"]
                self.beam_width = preds_layer.get_config()["arguments"]["beam_width"]
            else:
                self.accuracies = False
                self._model.metrics_names = ["loss", "ctc"]
                self.beam_width = 1
        else:
            self.nb_feat = nb_feat
            self.nb_class = nb_class
            self.max_timesteps = max_timesteps
            self.max_label_sequence_length = max_label_sequence_length
            self.accuracies = accuracies
            self.beam_width = beam_width
            self.optimizer = optimizer

            if max_label_sequence_length > 1:
                self.accuracies = False

            # building model from scratch
            self._model = _build_ctc_model(nb_feat=self.nb_feat,
                                           nb_class=self.nb_class,
                                           max_timesteps=self.max_timesteps,
                                           max_label_sequence_length=self.max_label_sequence_length,
                                           accuracies=self.accuracies,
                                           beam_width=self.beam_width,
                                           optimizer=self.optimizer)

        # construction of prediction function
        self._softmax_model = Model(inputs=[self._model.get_layer("data").input], 
                                    outputs=[self._model.get_layer("softmax").output])


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
        X_train, X_train_mask = pad_sequence_into_array(X, maxlen=self.max_timesteps, padding="pre")
        y_train, y_train_mask = pad_sequence_into_array(y, maxlen=self.max_label_sequence_length, padding="pre")
        sample_weight_train = [sample_weight]
        if self.accuracies:
            sample_weight_train += [None] * 2

        if validation_data:
            # if validation data is provided then just take them and transform to the input format of CTC network
            X_val, y_val = validation_data
            # pad/truncate validation data and generate masks for them to take into account only really existing points
            X_val, X_val_mask = pad_sequence_into_array(X_val, maxlen=self.max_timesteps)
            y_val, y_val_mask = pad_sequence_into_array(y_val, maxlen=self.max_label_sequence_length)
            # data formation
            inputs_val = {"data": X_val.astype(np.float32),
                          "labels": y_val.astype(np.int32),
                          "data_len": np.sum(X_val_mask, axis=1, dtype=np.int32),
                          "labels_len": np.sum(y_val_mask, axis=1, dtype=np.int32)
                          }
            outputs_val = {"ctc": np.zeros(y_val.shape),
                           "wa": np.zeros(y_val.shape), 
                           "ua": np.zeros(y_val.shape)
                           }
            validation_data = (inputs_val, outputs_val)
        elif validation_split and 0. < validation_split < 1.:
            # if validation data is not provided but we still want to do a validation we divide data randomly
            # divide indexes
            idxs_train, idxs_val = train_test_split(np.arange(X_train.shape[0]), 
                                                    test_size=validation_split, 
                                                    random_state=validation_random_state)
            # divide data correspondingly to indexes
            X_train, X_val = X_train[idxs_train], X_train[idxs_val]
            X_train_mask, X_val_mask = X_train_mask[idxs_train], X_train_mask[idxs_val]
            y_train, y_val = y_train[idxs_train], y_train[idxs_val]
            y_train_mask, y_val_mask = y_train_mask[idxs_train], y_train_mask[idxs_val]
            if sample_weight_train[0] is not None:
                sample_weight_train[0] = sample_weight_train[0][idxs_train]
            # data formation
            inputs_val = {"data": X_val.astype(np.float32),
                          "labels": y_val.astype(np.int32),
                          "data_len": np.sum(X_val_mask, axis=1, dtype=np.int32),
                          "labels_len": np.sum(y_val_mask, axis=1, dtype=np.int32)
                          }
            outputs_val = {"ctc": np.zeros(y_val.shape),
                           "wa": np.zeros(y_val.shape), 
                           "ua": np.zeros(y_val.shape)
                           }
            validation_data = (inputs_val, outputs_val)
        else:
            # do no validation
            validation_data = None

        # input format to CTC network requires 4 inputs: input data, labels and two masks.
        # output is at least one dummy output to forward errors computed in Lambda layer to the optimizer
        inputs_train = {"data": X_train.astype(np.float32),
                        "labels": y_train.astype(np.int32),
                        "data_len": np.sum(X_train_mask, axis=1, dtype=np.int32),
                        "labels_len": np.sum(y_train_mask, axis=1, dtype=np.int32)
                        }
        outputs_train = {"ctc": np.zeros(y_train.shape),
                         "wa": np.zeros(y_train.shape), 
                         "ua": np.zeros(y_train.shape)
                         }
        # recalculate metrics that are datasetwise (e.g. unweughted accuracy)
        if validation_data is not None:
            val_metrics_update = ValidationMetricsCorrectionCallback()
            callbacks_new = callbacks + [val_metrics_update]
        else:
            callbacks_new = callbacks
        
        # start fitting
        return self._model.fit(inputs_train, outputs_train, 
                               batch_size=batch_size, epochs=epochs, verbose=verbose, 
                               callbacks=callbacks_new, 
                               validation_split=0., validation_data=validation_data, 
                               shuffle=shuffle, sample_weight=sample_weight_train)


    def predict(self, X, batch_size=64, beam_width=-1, top_paths=1, return_probas=False, verbose=0):
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
        X, X_mask = pad_sequence_into_array(X, maxlen=self.max_timesteps)

        # generating valid mask
        data_len = np.sum(X_mask, axis=1, dtype=np.int32)
        # create tensor variable
        data_len = K.variable(data_len)

        # do a prediction; it will be the output of the softmax
        softmax = self._softmax_model.predict(X, batch_size=batch_size)
        # create tensor variable from predictions
        softmax = K.variable(softmax)
        # decode softmax output to obtain label sequence
        if beam_width == -1:
            beam_width = self.beam_width
        y_pred, log_probas = K.ctc_decode(y_pred=softmax, 
                                          input_length=K.reshape(data_len, [-1]), 
                                          greedy=False, 
                                          beam_width=beam_width, 
                                          top_paths=top_paths)
        y_pred_evaluated = []
        log_probas_evaluated = K.eval(log_probas)
        for path_num in range(top_paths):
            y_pred_evaluated.append(K.eval(y_pred[path_num][:, :self.max_label_sequence_length]))

        if return_probas:
            return y_pred_evaluated, np.exp(log_probas_evaluated)
        else:
            return y_pred_evaluated

    
    def evaluate(self, X, y, batch_size=64, verbose=0, beam_width=-1):
        """
        """

        X_padded, X_mask = pad_sequence_into_array(X, maxlen=self.max_timesteps)
        y_padded, y_mask = pad_sequence_into_array(y, maxlen=self.max_label_sequence_length)
        
        inputs = {"data": X_padded.astype(np.float32),
                  "labels": y_padded.astype(np.int32),
                  "data_len": np.sum(X_mask, axis=1, dtype=np.int32),
                  "labels_len": np.sum(y_mask, axis=1, dtype=np.int32)
                 }
        
        losses = self._model.predict(inputs, batch_size=batch_size)

        results = {"ctc": np.mean(losses[0])}

        if self.accuracies:
            preds = self.predict(X, batch_size=batch_size, beam_width=beam_width)[0]

            results["wa"] = metrics.weighted_accuracy(y, preds)
            results["ua"] = metrics.unweighted_accuracy(y, preds)
        
        return results


    def summary(self):
        """
        Print the summary of a model
        
        Args:
            None
        
        Return:
            None
        """
        Model(inputs=[self._model.get_layer("data").input], 
              outputs=[self._model.get_layer("softmax").output]).summary()
    

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


def _ctc_lambda_func(inputs):
    """ Calculates CTC objective function using masks

    Args:
        y_pred(Tensor): output of softmax layer, has shape of (None, TIMESTEPS, NB_CLASS + 1)
        labels(Tensor): true label sequences, has a shape of (None, MAX_LABEL_SEQUENCE_LENGTH)
        input_length(Tensor): real lengths of input sequences, has s shape of (None, )
        label_length(Tensor): real lengths of label sequences, has s shape of (None, )

    Return:
        ctc_obj(Tensor): CTC errors for every input entity, has a shape of (None, 1)
    """

    # parsing input dict of tensors
    preds, labels, data_len, labels_len = inputs
        
    # shift is critical here since the first couple outputs of the RNN tend to be garbage:
    shift = 0
    preds = preds[:, shift:, :]
    data_len -= shift
        
    # CTC objective calculation
    ctc_obj = K.ctc_batch_cost(y_true=labels, 
                               y_pred=preds, 
                               input_length=data_len, 
                               label_length=labels_len)

    return K.squeeze(ctc_obj, axis=1)


def _pred_lambda_func(args, beam_width=1):
    """ Calculate accuracy metric given by acc_func

    Args:
        y_pred(Tensor): output of softmax layer, has shape of (None, TIMESTEPS, NB_CLASS + 1)
        labels(Tensor): true label sequences, has a shape of (None, MAX_LABEL_SEQUENCE_LENGTH)
        input_length(Tensor): real lengths of input sequences, has s shape of (None, )
        label_length(Tensor): real lengths of label sequences, has s shape of (None, )
        acc_func(function(*, *)): function to calculate accuracy

    Return:
        acc(Tensor): tensor with all the same entities equal to accuracy returned by acc_func
    """

    # Parsing args
    softmax, data_len = args
    # Decode using CTC way
    preds = K.ctc_decode(y_pred=softmax,
                         input_length=K.reshape(data_len, [-1]), 
                         greedy=False, 
                         beam_width=beam_width, 
                         top_paths=1)[0][0]
    return preds

def _wa_lambda_func(args):

    y_true, y_pred = args

    acc = K.zeros_like(y_true[:, 0], dtype="float32") + metrics.weighted_accuracy_tf(y_true, y_pred)

    return acc


def _ua_lambda_func(args):
    
    y_true, y_pred = args

    acc = K.zeros_like(y_true[:, 0], dtype="float32") + metrics.unweighted_accuracy_tf(y_true, y_pred)
    
    return acc


def _loss_placeholder(y_true, y_pred):
    """
    Placeholder for calculating loss function in compile step. All the real loss computations
    are done in Lambda layers

    Args:
        y_true(Tensor): no matter what, usually zero tensor
        y_pred(Tensor): previously calculated loss functions in Lambda layers

    Return:
        loss(Tensor): losses
    """
    loss = y_pred
    return loss


class ValidationMetricsCorrectionCallback(Callback):
    
    def __init__(self, **kwargs):
        super(ValidationMetricsCorrectionCallback, self).__init__(**kwargs)
        
    def on_train_begin(self, logs={}):
        wa_layer = self.model.get_layer("wa")
        if wa_layer is not None:
            self.accuracies = True
        else:
            self.accuracies = False
    
    def on_epoch_end(self, epoch, logs={}):
        if self.accuracies and self.validation_data is not None:
            losses = self.model.predict(dict(zip(["data", "labels", "data_len", "labels_len"], 
                                                 self.validation_data[:4])), 
                                        batch_size=self.validation_data[0].shape[0])
            losses = dict(zip(["ctc", "wa", "ua"], losses))
            for l in losses.items():
                logs["val_" + l[0]] = np.mean(l[1], dtype=type(logs["val_" + l[0]]))


def _build_ctc_model(nb_feat,
                     nb_class,
                     max_timesteps,
                     max_label_sequence_length=1,
                     accuracies=True,
                     beam_width=1, 
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
                         return_sequences=True
                         )(blstm_output1)
    backward_lstm2 = LSTM(units=64,
                          return_sequences=True,
                          go_backwards=True
                          )(blstm_output1)
    blstm_output2 = Concatenate(axis=-1)([forward_lstm2, backward_lstm2])

    # Dense classification layers
    hidden = TimeDistributed(Dense(512, activation="relu"))(blstm_output2)

    # Softmax layer which accounts fot NULL class introduced in CTC
    softmax = TimeDistributed(Dense(nb_class + 1, activation="softmax"), name="softmax")(hidden)

    # Additional layers to enable CTC loss function calculation through Lambda layer

    # Get true labels as an input. During test stage can be filled with whatever you want.
    labels = Input(name="labels", shape=[max_label_sequence_length], dtype="float32")
    # real lengths of input sequences. Used as a mask
    data_len = Input(name="data_len", shape=[1], dtype="int32")
    # real lengths of labels sequences. Used as a mask
    labels_len = Input(name="labels_len", shape=[1], dtype="int32")

    # Lambda layer for CTC loss computations
    ctc_loss = Lambda(_ctc_lambda_func,
                      output_shape=(1,),
                      name="ctc")([softmax, labels, data_len, labels_len])
    
    # Lambda layer for predicted labels
    preds = Lambda(_pred_lambda_func, 
                   output_shape=(max_label_sequence_length,), 
                   arguments={"beam_width": beam_width}, 
                   name="preds")([softmax, data_len])

    # Lambda layer for weighted accuracy computations
    wa = Lambda(_wa_lambda_func, 
                output_shape=(1,), 
                name="wa")([labels, preds])
    # Lambda layer for unweighted accuracy computations
    ua = Lambda(_ua_lambda_func, 
                output_shape=(1,), 
                name="ua")([labels, preds])

    # Construct different outputs with and without accuracies
    if accuracies:
        # Desired outputs
        output = [ctc_loss, wa, ua]
        # Their weights in final loss which we want to optimize
        lw = [1.0, 0.0, 0.0]
    else:
        # Desired outputs
        output = [ctc_loss]
        # Their weights in final loss which we want to optimize
        lw = [1.0]

    # Build final model and compile it
    model = Model(inputs=[data, labels, data_len, labels_len], outputs=output)
    model.compile(loss=_loss_placeholder,
                  loss_weights=lw,
                  optimizer=optimizer)
    model.metrics_names = ["loss", "ctc", "wa", "ua"]
    return model
