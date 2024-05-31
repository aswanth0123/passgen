import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Assume _default_weightnorm is defined somewhere in your code
_default_weightnorm = False

def Linear(name, input_dim, output_dim, inputs, biases=True, initialization=None, weightnorm=None, gain=1.):
    with tf.name_scope(name) as scope:
        print(f"Linear layer: {name}, Input dim: {input_dim}, Output dim: {output_dim}")

        def uniform(stdev, size):
            return np.random.uniform(low=-stdev * np.sqrt(3), high=stdev * np.sqrt(3), size=size).astype('float32')

        if initialization == 'lecun':
            weight_values = uniform(np.sqrt(1./input_dim), (input_dim, output_dim))
        elif initialization == 'glorot' or (initialization is None):
            weight_values = uniform(np.sqrt(2./(input_dim+output_dim)), (input_dim, output_dim))
        elif initialization == 'he':
            weight_values = uniform(np.sqrt(2./input_dim), (input_dim, output_dim))
        elif initialization == 'glorot_he':
            weight_values = uniform(np.sqrt(4./(input_dim+output_dim)), (input_dim, output_dim))
        elif initialization == 'orthogonal' or (initialization is None and input_dim == output_dim):
            def sample(shape):
                flat_shape = (shape[0], np.prod(shape[1:]))
                a = np.random.normal(0.0, 1.0, flat_shape)
                u, _, v = np.linalg.svd(a, full_matrices=False)
                q = u if u.shape == flat_shape else v
                return q.reshape(shape).astype('float32')
            weight_values = sample((input_dim, output_dim))
        elif initialization[0] == 'uniform':
            weight_values = np.random.uniform(low=-initialization[1], high=initialization[1], size=(input_dim, output_dim)).astype('float32')
        else:
            raise Exception('Invalid initialization!')

        weight_values *= gain
        print(f"Weight values shape: {weight_values.shape}, dtype: {weight_values.dtype}")

        weight = lib.param(name + '.W', weight_values)
        print(f"Created weight: {weight}")

        if weightnorm is None:
            weightnorm = _default_weightnorm
        if weightnorm:
            norm_values = np.sqrt(np.sum(np.square(weight_values), axis=0))
            target_norms = lib.param(name + '.g', norm_values)
            norms = tf.sqrt(tf.reduce_sum(tf.square(weight), reduction_indices=[0]))
            weight = weight * (target_norms / norms)

        if inputs.get_shape().ndims == 2:
            result = tf.matmul(inputs, weight)
        else:
            reshaped_inputs = tf.reshape(inputs, [-1, input_dim])
            result = tf.matmul(reshaped_inputs, weight)
            result_shape = tf.concat([tf.shape(inputs)[:-1], [output_dim]], 0)
            result = tf.reshape(result, result_shape)

        if biases:
            bias_values = np.zeros((output_dim,), dtype='float32')
            result = tf.nn.bias_add(result, lib.param(name + '.b', bias_values))

        print(f"Result shape: {result.get_shape()}, dtype: {result.dtype}")
        return result

# Mock lib.param function for debugging
class Lib:
    def __init__(self):
        self.params = {}

    def param(self, name, value):
        if name in self.params:
            return self.params[name]
        else:
            print(f"Creating new parameter: {name}, Value shape: {value.shape}, dtype: {value.dtype}")
            result = tf.Variable(value, name=name)
            self.params[name] = result
            return result

# Initialize the lib object
lib = Lib()

# Mock the rest of your functions
def ResBlock(name, inputs, layer_dim):
    # Simple pass-through for debugging
    print(f"ResBlock: {name}, Inputs shape: {inputs.get_shape()}")
    return inputs

def softmax(inputs, output_dim):
    # Simple softmax for debugging
    return tf.nn.softmax(inputs)


# Mock the call to Generator for debugging
class Args:
    batch_size = 32
    seq_length = 10
    layer_dim = 128

args = Args()
charmap = [0] * 10  # Mock charmap for debugging

# Call the Generator function

