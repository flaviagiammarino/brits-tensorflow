import tensorflow as tf

class FeatureRegression(tf.keras.layers.Layer):
    
    def __init__(self):
        
        '''
        Feature regression layer, see Equation (7) in Section 4.3 of the BRITS paper.
        
        Parameters:
        __________________________________
        None.
        '''

        super(FeatureRegression, self).__init__()
        
    def build(self, input_shape):
        self.w = self.add_weight('w', shape=[int(input_shape[-1]), int(input_shape[-1])])
        self.b = self.add_weight('b', shape=[int(input_shape[-1])])
        self.d = tf.ones([input_shape[-1], input_shape[-1]]) - tf.eye(input_shape[-1], input_shape[-1])
        
    def call(self, inputs):
        
        '''
        Parameters:
        __________________________________
        inputs: tf.Tensor.
            Complement vector, tensor with shape (samples, features) where samples is the
            batch size and features is the number of time series.

        Returns:
        __________________________________
        tf.Tensor.
            Feature-based estimation, tensor with shape (samples, features) where samples
            is the batch size and features is the number of time series.
        '''
        
        return tf.matmul(inputs, self.w * self.d) + self.b


class TemporalDecay(tf.keras.layers.Layer):
    
    def __init__(self, units):
        
        '''
        Temporal decay layer, see Equation (3) in Section 4.1.1 of the BRITS paper.

        Parameters:
        __________________________________
        units: int.
            Number of hidden units of the recurrent layer.
        '''
        
        self.units = units
        super(TemporalDecay, self).__init__()
        
    def build(self, input_shape):
        self.w = self.add_weight('w', shape=[int(input_shape[-1]), self.units])
        self.b = self.add_weight('b', shape=[self.units])
        
    def call(self, inputs):
        
        '''
        Parameters:
        __________________________________
        inputs: tf.Tensor.
            Time gaps, tensor with shape (samples, features) where samples is the
            batch size and features is the number of time series.

        Returns:
        __________________________________
        tf.Tensor.
            Temporal decay, tensor with shape (samples, units) where samples is the
            batch size and units is the number of hidden units of the recurrent layer.
        '''
        
        return tf.exp(- tf.nn.relu(tf.matmul(inputs, self.w) + self.b))


class RITS(tf.keras.layers.Layer):
    
    def __init__(self, units):
        
        '''
        RITS layer, see Section 4.3 of the BRITS paper.

        Parameters:
        __________________________________
        units: int.
            Number of hidden units of the recurrent layer.
        '''
        
        self.units = units
        self.rnn_cell = None
        self.temp_decay = None
        self.hist_reg = None
        self.feat_reg = None
        self.weight_combine = None
        super(RITS, self).__init__()
        
    def build(self, input_shape):
        
        if self.rnn_cell is None:
            self.rnn_cell = tf.keras.layers.LSTMCell(units=self.units)

        if self.temp_decay is None:
            self.temp_decay = TemporalDecay(units=self.units)
        
        if self.hist_reg is None:
            self.hist_reg = tf.keras.layers.Dense(units=input_shape[2])

        if self.feat_reg is None:
            self.feat_reg = FeatureRegression()

        if self.weight_combine is None:
            self.weight_combine = tf.keras.layers.Dense(units=input_shape[2], activation='sigmoid')
        
    def call(self, inputs):
        
        '''
        Parameters:
        __________________________________
        inputs: tf.Tensor.
            Model inputs, tensor with shape (samples, timesteps, features, 3) where samples is the
            batch size, timesteps is the number of time steps, features is the number of time series
            and 3 is the number of model inputs (time series, masking vectors and time gaps).

        Returns:
        __________________________________
        outputs: tf.Tensor.
            Imputations, tensor with shape (samples, timesteps, features) where samples is the
            batch size, timesteps is the number of time steps and features is the number of
            time series.
        
        loss: tf.Tensor.
            Loss value, scalar tensor.
        '''
        
        # Get the inputs (time series, masking vectors and time gaps).
        values = tf.cast(inputs[:, :, :, 0], dtype=tf.float32)
        masks = tf.cast(inputs[:, :, :, 1], dtype=tf.float32)
        deltas = tf.cast(inputs[:, :, :, 2], dtype=tf.float32)

        # Initialize the outputs (imputations).
        outputs = tf.TensorArray(
            element_shape=(inputs.shape[0], inputs.shape[2]),
            size=inputs.shape[1],
            dynamic_size=False,
            dtype=tf.float32
        )
        
        # Initialize the states (memory state and carry state).
        h = tf.zeros(shape=(tf.shape(inputs)[0], self.units))
        c = tf.zeros(shape=(tf.shape(inputs)[0], self.units))
        
        # Initialize the loss (mean absolute error).
        loss = 0.
        
        # Loop across the time steps.
        for t in tf.range(inputs.shape[1]):
            
            # Extract the inputs for the considered time step.
            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]
            
            # Run the history-based estimation, see Equation (1) in Section 4.1.1 of the BRITS paper.
            x_h = self.hist_reg(h)

            # Derive the complement vector, see Equation (2) in Section 4.1.1 of the BRITS paper.
            x_c = m * x + (1 - m) * x_h

            # Derive the temporal decay, see Equation (3) in Section 4.1.1 of the BRITS paper.
            gamma = self.temp_decay(d)
            
            # Run the feature-based estimation, see Equation (7) in Section 4.3 of the BRITS paper.
            z_h = self.feat_reg(x_c)
            
            # Derive the weights of the history-based and feature-based estimation, see Equation (8) in Section 4.3 of the BRITS paper.
            beta = self.weight_combine(tf.concat([gamma, m], axis=-1))

            # Combine the history-based and feature-based estimation, see Equation (9) in Section 4.3 of the BRITS paper.
            c_h = beta * z_h + (1 - beta) * x_h

            # Update the loss.
            loss += tf.reduce_sum((tf.abs(x - x_h) * m) / (tf.reduce_sum(m) + 1e-5))
            loss += tf.reduce_sum((tf.abs(x - z_h) * m) / (tf.reduce_sum(m) + 1e-5))
            loss += tf.reduce_sum((tf.abs(x - c_h) * m) / (tf.reduce_sum(m) + 1e-5))
            
            # Update the outputs.
            c_c = m * x + (1 - m) * c_h
            outputs = outputs.write(index=t, value=c_c)
            
            # Update the states.
            h, [h, c] = self.rnn_cell(inputs=tf.concat([c_c, m], axis=-1), states=[h * gamma, c])

        # Reshape the outputs.
        outputs = tf.transpose(outputs.stack(), [1, 0, 2])

        # Average the loss.
        loss /= (3 * inputs.shape[1])
        
        return outputs, loss
