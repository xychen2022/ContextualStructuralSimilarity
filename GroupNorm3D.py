from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
from tensorflow.keras import backend as K
import tensorflow as tf

class GroupNormalization(Layer):
    
    def __init__(self,
                 groups=1,
                 axis=-1,
                 epsilon=1e-5,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(GroupNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
    
    def build(self, input_shape):
        dim = input_shape[self.axis]
        
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')
    
        if dim < self.groups:
            raise ValueError('Number of groups (' + str(self.groups) + ') cannot be '
                             'more than the number of channels (' +
                             str(dim) + ').')
        
        if dim % self.groups != 0:
            raise ValueError('Number of channels must be divisible by the number of groups')
        
        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)
        
        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, **kwargs):
        if self.axis == -1 or self.axis == 4:
            inputs = K.permute_dimensions(inputs, [0, 4, 1, 2, 3])
        
        input_shape = K.int_shape(inputs)    
        original_shape = [tf.shape(inputs)[0],] + list(input_shape[1:])
        
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[1] = input_shape[1]

        reshape_group_shape = [tf.shape(inputs)[i] for i in range(len(input_shape))]
        reshape_group_shape[1] = tf.shape(inputs)[1] // self.groups
        group_shape = [tf.shape(inputs)[0], self.groups]
        group_shape.extend(reshape_group_shape[1:])
        
        inputs = K.reshape(inputs, group_shape)
        
        mean = K.mean(inputs, axis=[2, 3, 4, 5], keepdims=True)
        variance = K.var(inputs, axis=[2, 3, 4, 5], keepdims=True)
        
        inputs = (inputs - mean) / (K.sqrt(variance + self.epsilon))
        inputs = K.reshape(inputs, original_shape)
        
        outputs = inputs
    
        # Broadcast all parameters.
        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            outputs = outputs * broadcast_gamma
            
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            outputs = outputs + broadcast_beta

        if self.axis == -1 or self.axis == 4:
            outputs = K.permute_dimensions(outputs, [0, 2, 3, 4, 1])
        
        return outputs

    def get_config(self):
        config = {
            'groups': self.groups,
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
            }
        base_config = super(GroupNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shape):
        return input_shape
