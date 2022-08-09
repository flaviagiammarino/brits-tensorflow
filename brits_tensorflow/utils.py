import numpy as np
import tensorflow as tf

def get_inputs(x):
    
    '''
    Derive the masking vectors and calculate the time gaps.
    See Section 3 of the BRITS paper.
    
    Parameters:
    __________________________________
    x: np.array.
        Time series, array with shape (samples, features) where samples is the length of the time series
        and features is the number of time series.

    Returns:
    __________________________________
    inputs: tf.Tensor.
        Model inputs, tensor with shape (samples, features, 3) where 3 is the number of model inputs
        (time series, masking vectors and time gaps).
    '''
    
    # Derive the masking vector.
    m = np.where(np.isnan(x), 0, 1)
    
    # Calculate the time gaps.
    d = np.zeros(x.shape)
    for t in range(1, x.shape[0]):
        d[t, :] = np.where(m[t - 1, :] == 0, d[t - 1, :] + 1, 1)

    # Standardize the time gaps.
    d = (d - d.mean(axis=0)) / (d.std(axis=0) + 1e-5)

    # Mask the inputs.
    x = np.where(m == 0, 0, x)
    
    # Cast the inputs to float tensors.
    x = tf.expand_dims(tf.cast(x, tf.float32), axis=-1)
    m = tf.expand_dims(tf.cast(m, tf.float32), axis=-1)
    d = tf.expand_dims(tf.cast(d, tf.float32), axis=-1)

    # Concatenate the inputs.
    inputs = tf.concat([x, m, d], axis=-1)
    
    return inputs
 