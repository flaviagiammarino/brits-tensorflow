import numpy as np
import pandas as pd
import tensorflow as tf

from brits_tensorflow.utils import get_inputs
from brits_tensorflow.modules import RITS

class BRITS:
    
    def __init__(self, x, units, timesteps):
        
        '''
        Implementation of multivariate time series imputation model introduced in Cao, W., Wang, D., Li, J.,
        Zhou, H., Li, L. and Li, Y., 2018. BRITS: Bidirectional recurrent imputation for time series.
        Advances in neural information processing systems, 31.
        
        Parameters:
        __________________________________
        x: np.array.
            Time series, array with shape (samples, features) where samples is the length of the time series
            and features is the number of time series.
        
        units: int.
            Number of hidden units of the recurrent layer.

        timesteps: int.
            Number of time steps.
        '''
        
        self.x = x
        self.x_min = np.nanmin(x, axis=0)
        self.x_max = np.nanmax(x, axis=0)
        self.samples = x.shape[0]
        self.features = x.shape[1]
        self.units = units
        self.timesteps = timesteps
  
    def fit(self, learning_rate=0.001, batch_size=32, epochs=100, verbose=True):
        
        '''
        Train the model.
        
        Parameters:
        __________________________________
        learning_rate: float.
            Learning rate.
            
        batch_size: int.
            Batch size.
            
        epochs: int.
            Number of epochs.
            
        verbose: bool.
            True if the training history should be printed in the console, False otherwise.
        '''
        
        # Scale the time series.
        x = (self.x - self.x_min) / (self.x_max - self.x_min)

        # Get the inputs in the forward direction.
        forward = get_inputs(x)

        # Get the inputs in the backward direction.
        backward = get_inputs(np.flip(x, axis=0))

        # Generate the input sequences.
        dataset = tf.keras.utils.timeseries_dataset_from_array(
            data=tf.concat([forward, backward], axis=-1),
            targets=None,
            sequence_length=self.timesteps,
            sequence_stride=self.timesteps,
            batch_size=batch_size,
            shuffle=True
        )
        
        # Build the model.
        model = build_fn(
            timesteps=self.timesteps,
            features=self.features,
            units=self.units
        )
        
        # Define the training loop.
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        @tf.function
        def train_step(data):
            with tf.GradientTape() as tape:
                
                # Calculate the loss.
                _, loss = model(data)
            
            # Calculate the gradient.
            gradient = tape.gradient(loss, model.trainable_variables)
        
            # Update the weights.
            optimizer.apply_gradients(zip(gradient, model.trainable_variables))
        
            return loss

        # Train the model.
        for epoch in range(epochs):
            for data in dataset:
                loss = train_step(data)
            if verbose:
                print('epoch: {}, loss: {:,.6f}'.format(1 + epoch, loss))

        # Save the model.
        self.model = model
    
    def impute(self, x):
        
        '''
        Impute the time series.
        
        Parameters:
        __________________________________
        x: np.array.
            Time series, array with shape (samples, features) where samples is the length of the time series
            and features is the number of time series.
            
        Returns:
        __________________________________
        imputed: pd.DataFrame.
            Data frame with imputed time series.
        '''
        
        if x.shape[1] != self.features:
            raise ValueError(f'Expected {self.features} features, found {x.shape[1]}.')
        
        else:
            
            # Scale the time series.
            x = (x - self.x_min) / (self.x_max - self.x_min)
    
            # Get the inputs in the forward direction.
            forward = get_inputs(x)
    
            # Get the inputs in the backward direction.
            backward = get_inputs(np.flip(x, axis=0))
    
            # Generate the input sequences.
            dataset = tf.keras.utils.timeseries_dataset_from_array(
                data=tf.concat([forward, backward], axis=-1),
                targets=None,
                sequence_length=self.timesteps,
                sequence_stride=self.timesteps,
                batch_size=1,
                shuffle=False
            )
            
            # Generate the imputations.
            imputed = tf.concat([self.model(data)[0] for data in dataset], axis=0).numpy()
            imputed = np.concatenate([imputed[i, :, :] for i in range(imputed.shape[0])], axis=0)
            imputed = self.x_min + (self.x_max - self.x_min) * imputed

            return imputed


def build_fn(timesteps, features, units):
    
    '''
    Build the model, see Section 4.2 of the BRITS paper.
    
    Parameters:
    __________________________________
    features: int.
        Number of time series.

    timesteps: int.
        Number of time steps.
    
    units: int.
        Number of hidden units of the recurrent layer.
    '''
    
    # Define the input layer, the model takes 3 inputs (time series, masking
    # vectors and time gaps) for each direction (forward and backward).
    inputs = tf.keras.layers.Input(shape=(timesteps, features, 6))
    
    # Get the imputations and loss in the forward directions.
    forward_imputations, forward_loss = RITS(units=units)(inputs[:, :, :, :3])

    # Get the imputations and loss in the backward directions.
    backward_imputations, backward_loss = RITS(units=units)(inputs[:, :, :, 3:])
    
    # Average the imputations across both directions (forward and backward).
    outputs = (forward_imputations + backward_imputations) / 2

    # Sum the losses (forward loss, backward loss and consistency loss).
    loss = forward_loss + backward_loss + tf.reduce_mean(tf.abs(forward_imputations - backward_imputations))
    
    return tf.keras.Model(inputs, (outputs, loss))
