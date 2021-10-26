import numpy as np
import tensorflow as tf
import sklearn.metrics


########################################################################################################################
#                                                 Numpy metrics                                                        #
########################################################################################################################


def confusion_matrix(y_true, y_pred, normed=False):
    """
    Return the confusion matrix between real labels and predictions
    By definition a confusion matrix C is such that C_{i, j}
    is equal to the number of observations known to be in group i but
    predicted to be in group j

    Args:
        y_true(ndarray): real labels
        y_pred(ndarray): predictions
        normed(bool): whether to norm the matrix (can return nan's)

    Return:
        conf_mat(ndarray): confusion matrix, with true labels in first dimension and predicted in second
    """
    
    # flatten and cast labels
    y_true = np.array(y_true, dtype=np.int32).ravel()
    y_pred = np.array(y_pred, dtype=np.int32).ravel()
    # get confusion matrix
    conf_mat = sklearn.metrics.confusion_matrix(y_true, y_pred)
    # norm if needed
    if normed:
        with np.errstate(divide='ignore', invalid='ignore'):
            conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    return conf_mat


def weighted_accuracy(y_true, y_pred):
    """
    Computes overall accuracy over all predictions

    Args:
        y_true(ndarray): real labels
        y_pred(ndarray): predictions

    Return:
        wa(ndarray): weighted accuracy
    """

    if len(y_true.shape) == 2 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)

    # flatten and cast labels
    y_true = np.array(y_true, dtype=np.int32).ravel()
    y_pred = np.array(y_pred, dtype=np.int32).ravel()
    # check if y_pred is empty
    if y_pred.shape[0] == 0:
        return 0.0
    else:
        return np.mean(np.array(y_true == y_pred, dtype=np.float32))


def unweighted_accuracy(y_true, y_pred):
    """
    Computes mean class accuracy

    Args:
        y_true(ndarray): real labels
        y_pred(ndarray): predictions

    Return:
        ua(ndarray): unweighted accuracy
    """

    if len(y_true.shape) == 2 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)

    # flatten and cast labels
    y_true = np.array(y_true, dtype=np.int32).ravel()
    y_pred = np.array(y_pred, dtype=np.int32).ravel()
    # check if y_pred is empty
    if y_pred.shape[0] == 0:
        return 0.0
    else:
        # Unweighted accuracy is the mean value of diagonal elements of normed confusion matrix.
        # If y_pred contains -1 (which means no label in CTC) than corresponding row in normed
        # matrix will contain only nan's and won't be taken into account.
        conf_mat = confusion_matrix(y_true, y_pred, True)
        return np.nanmean(np.diag(conf_mat))


########################################################################################################################
#                                               Tensorflow metrics                                                     #
########################################################################################################################


def weighted_accuracy_tf(y_true, y_pred):
    """
    Computes overall accuracy over all predictions for TF

    Args:
        y_true(Tensor): real labels
        y_pred(Tensor): predictions

    Return:
        wa(Tensor): unweighted accuracy
    """

    # accuracy can be calculated only for one-label predictions
    y_pred = y_pred[:, :1]

    # flatten and cast labels
    y_true = tf.cast(tf.reshape(y_true, [-1]), dtype="int32")
    y_pred = tf.cast(tf.reshape(y_pred, [-1]), dtype="int32")

    # if y_pred is not empty
    def pred_not_empty():
        return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), dtype="float32"))

    # if y_pred is empty
    def pred_empty():
        return tf.constant(0.0)

    # check emptiness of y_pred
    wa = tf.cond(tf.equal(tf.shape(y_pred)[0], 0), 
                 pred_empty, 
                 pred_not_empty)
    return wa


def weighted_accuracy_ohe_tf(y_true, y_pred):
    """
    Computes overall accuracy over all predictions for TF

    Args:
        y_true(Tensor): real labels
        y_pred(Tensor): predictions

    Return:
        wa(Tensor): unweighted accuracy
    """

    # accuracy can be calculated only for one-label predictions
    y_true = tf.argmax(y_true, axis=1)
    y_pred = tf.argmax(y_pred, axis=1)

    return weighted_accuracy_tf(tf.reshape(y_true, [-1, 1]), 
                                tf.reshape(y_pred, [-1, 1]))


def unweighted_accuracy_tf(y_true, y_pred):
    """
    Computes mean class accuracy for TF

    Args:
        y_true(Tensor): real labels
        y_pred(Tensor): predictions

    Return:
        ua(Tensor): unweighted accuracy
    """

    # accuracy can be calculated only for one-label predictions
    y_pred = y_pred[:, :1]

    # flatten and cast labels
    y_true = tf.cast(tf.reshape(y_true, [-1]), dtype="int32")
    y_pred = tf.cast(tf.reshape(y_pred, [-1]), dtype="int32")
    
    # if y_pred is not empty
    def pred_not_empty():
        # y_pred may contain -1's (means no label in CTC) and TF function will break.
        # To avoid it 1 is added to all labels and predictions. Then this row and column are excluded
        # from the confusion matrix. It does not affect the answer because in true labels there is no
        # -1 labels. After that we check only those labels that present in y_true otherwise we're
        # going to divide by zero.
        conf_mat = tf.contrib.metrics.confusion_matrix(tf.add(y_pred, 1), tf.add(y_true, 1))
        right_answers = tf.cast(tf.diag_part(conf_mat), dtype="float32")[1:]
        normalizer = tf.reduce_sum(tf.transpose(conf_mat), 1)[1:]
        idx = tf.reshape(tf.where(~tf.equal(normalizer, 0)), [-1])

        # idx is always not empty, but TF don't know it and we need explicitly show it
        return tf.cond(tf.equal(tf.shape(idx)[0], 0), 
                       lambda: tf.constant(0.0), 
                       lambda: tf.reduce_mean(tf.gather(right_answers, idx) / 
                                              tf.cast(tf.gather(normalizer, idx), dtype="float32")))
    
    # if y_pred is empty
    def pred_empty():
        return tf.constant(0.0)
        
    # check emptiness of y_pred
    ua = tf.cond(tf.equal(tf.shape(y_pred)[0], 0), 
                 pred_empty, 
                 pred_not_empty)    
    return ua


def unweighted_accuracy_ohe_tf(y_true, y_pred):
    """
    Computes mean class accuracy for TF

    Args:
        y_true(Tensor): real labels
        y_pred(Tensor): predictions

    Return:
        ua(Tensor): unweighted accuracy
    """

    # accuracy can be calculated only for one-label predictions
    y_true = tf.argmax(y_true, axis=1)
    y_pred = tf.argmax(y_pred, axis=1)

    return unweighted_accuracy_tf(tf.reshape(y_true, [-1, 1]), 
                                  tf.reshape(y_pred, [-1, 1]))
