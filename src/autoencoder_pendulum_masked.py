import tensorflow as tf
# import tensorflow.compat.v1 as tf
import numpy as np


def full_network(params):
    """
    Define the full network architecture.

    Arguments:
        params - Dictionary object containing the parameters that specify the training.
        See README file for a description of the parameters.

    Returns:
        network - Dictionary containing the tensorflow objects that make up the network.
    """
    input_dim = params['input_dim']
    latent_dim = params['latent_dim']
    activation = params['activation']
    poly_order = params['poly_order']
    if 'include_sine' in params.keys():
        include_sine = params['include_sine']
    else:
        include_sine = False
    library_dim = params['library_dim']
    model_order = params['model_order']

    network = {}

    random_mask = tf.placeholder(tf.float32, shape=[None, input_dim], name='random_mask')

    x = tf.placeholder(tf.float32, shape=[None, input_dim], name='x')
    dx = tf.placeholder(tf.float32, shape=[None, input_dim], name='dx')
    if model_order == 2:
        ddx = tf.placeholder(tf.float32, shape=[None, input_dim], name='ddx')

    x_masked = x * random_mask
    dx_masked = dx * random_mask
    if model_order == 2:
        ddx_masked = ddx * random_mask

    if activation == 'linear':
        z, x_decode, encoder_weights, encoder_biases, decoder_weights, decoder_biases = linear_autoencoder(x_masked, input_dim, latent_dim)
    else:
        z, x_decode, encoder_weights, encoder_biases, decoder_weights, decoder_biases = nonlinear_autoencoder(x_masked, input_dim, latent_dim, params['widths'], activation=activation)
    
    if model_order == 1:
        dz = z_derivative(x_masked, dx_masked, encoder_weights, encoder_biases, activation=activation)
        Theta = sindy_library_tf(z, latent_dim, poly_order, include_sine)
    else:
        dz,ddz = z_derivative_order2(x_masked, dx_masked, ddx_masked, encoder_weights, encoder_biases, activation=activation)
        Theta = sindy_library_tf_order2(z, dz, latent_dim, poly_order, include_sine)

    if params['coefficient_initialization'] == 'xavier':
        sindy_coefficients = tf.get_variable('sindy_coefficients', shape=[library_dim,latent_dim], initializer=tf.contrib.layers.xavier_initializer())
    elif params['coefficient_initialization'] == 'specified':
        sindy_coefficients = tf.get_variable('sindy_coefficients', initializer=params['init_coefficients'])
    elif params['coefficient_initialization'] == 'constant':
        # value = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        value = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        sindy_coefficients = tf.get_variable('sindy_coefficients', shape=[library_dim,latent_dim], initializer=tf.constant_initializer(value))

    elif params['coefficient_initialization'] == 'normal':
        sindy_coefficients = tf.get_variable('sindy_coefficients', shape=[library_dim,latent_dim], initializer=tf.initializers.random_normal(mean=0.0, stddev=0.5, seed=2))
    
    if params['sequential_thresholding']:
        coefficient_mask = tf.placeholder(tf.float32, shape=[library_dim,latent_dim], name='coefficient_mask')
        sindy_predict = tf.matmul(Theta, coefficient_mask*sindy_coefficients)
        network['coefficient_mask'] = coefficient_mask
    else:
        sindy_predict = tf.matmul(Theta, sindy_coefficients)

    if model_order == 1:
        dx_decode = z_derivative(z, sindy_predict, decoder_weights, decoder_biases, activation=activation)
        dx_decode_tf = tf.gradients(x_decode, z)
    else:
        dx_decode,ddx_decode = z_derivative_order2(z, dz, sindy_predict, decoder_weights, decoder_biases,
                                             activation=activation)
        dx_decode_tf = tf.gradients(x_decode, z)
    # Init Stochastic Approximation paramsters
    p_star = tf.zeros_like(sindy_coefficients)
    d_star0 = tf.zeros_like(sindy_coefficients)
    d_star1 = tf.zeros_like(sindy_coefficients)
    
    # Init vals
    p_init = tf.constant(1.0)
    d_init = tf.constant(5e-4)
    p_star = tf.add(p_star, p_init)
    d_star0 = tf.add(d_star0, d_init)
    d_star1 = tf.add(d_star1, d_init)
    
#     p_star = tf.FloatTensor(p_star)
#     d_star0 = tf.FloatTensor(d_star0)
#     d_star1 = tf.FloatTensor(d_star1)
    
#     p_star = tf.cast(p_star, tf.float32)
#     d_star0 = tf.cast(d_star0, tf.float32)
#     d_star1 = tf.cast(d_star1, tf.float32)
    
#     print(type(p_star))
#     print(type(d_star0))
#     print(type(d_star1))
    
    
    
    network['x'] = x
    network['dx'] = dx
    network['z'] = z
    network['dz'] = dz
    network['x_decode'] = x_decode
    network['dx_decode'] = dx_decode
    network['dx_decode_tf'] = dx_decode_tf
    network['encoder_weights'] = encoder_weights
    network['encoder_biases'] = encoder_biases
    network['decoder_weights'] = decoder_weights
    network['decoder_biases'] = decoder_biases
    network['Theta'] = Theta
    network['sindy_coefficients'] = sindy_coefficients
    network['p_star'] = p_star
    network['d_star0'] = d_star0
    network['d_star1'] = d_star1
    
    if model_order == 1:
        network['dz_predict'] = sindy_predict
    else:
        network['ddz'] = ddz
        network['ddz_predict'] = sindy_predict
        network['ddx'] = ddx
        network['ddx_decode'] = ddx_decode

    return network

def define_loss_init(network, params):
    """
    Create the loss functions.
    Arguments:
        network - Dictionary object containing the elements of the network architecture.
        This will be the output of the full_network() function.
    """
    x = network['x']
    x_decode = network['x_decode']
    if params['model_order'] == 1:
        dz = network['dz']
        dz_predict = network['dz_predict']
        dx = network['dx']
        dx_decode = network['dx_decode']
    else:
        ddz = network['ddz']
        ddz_predict = network['ddz_predict']
        ddx = network['ddx']
        ddx_decode = network['ddx_decode']
    sindy_coefficients = params['coefficient_mask']*network['sindy_coefficients']

    losses = {}
    losses['decoder'] = tf.reduce_sum((x - x_decode)**2) * params['loss_weight_decoder']
#     losses['decoder'] = tf.reduce_sum(tf.abs(x - x_decode)) * params['loss_weight_decoder']
    if params['model_order'] == 1:
        losses['sindy_z'] = tf.reduce_sum((dz - dz_predict)**2) * params['loss_weight_sindy_z']
        losses['sindy_x'] = tf.reduce_sum((dx - dx_decode)**2) * params['loss_weight_sindy_x']
    else:
        losses['sindy_z'] = tf.reduce_sum((ddz - ddz_predict)**2) * params['loss_weight_sindy_z']
        losses['sindy_x'] = tf.reduce_sum((ddx - ddx_decode)**2) * params['loss_weight_sindy_x']
    if params['prior'] == None:
        losses['sindy_regularization'] = tf.reduce_mean(tf.abs(sindy_coefficients)) * params['loss_weight_sindy_regularization']
    if params['prior'].lower() == "laplace":
        losses['sindy_regularization'] = laplace_prior(network, params) * params['loss_weight_sindy_regularization']
    if params['prior'].lower() == "spike-and-slab":
        losses['sindy_regularization'] = spike_and_slab_prior_init(network, params) * params['loss_weight_sindy_regularization']
        
    loss = losses['decoder'] \
           + losses['sindy_z'] \
           + losses['sindy_x']
#     loss *= (50*1000)
    loss += losses['sindy_regularization']
           
    
    loss_refinement = losses['decoder'] \
                      + losses['sindy_z'] \
                      + losses['sindy_x']

    return loss, losses, loss_refinement

def laplace_prior(network, params):
#     print("Init Laplace prior")
    sindy_coefficients = params['coefficient_mask']*network['sindy_coefficients']
    prior_loss = tf.reduce_mean(tf.abs(sindy_coefficients))
    return prior_loss

# TODO: Sep/9/2021
def spike_and_slab_prior_init(network, params):
    sindy_coefficients = params['coefficient_mask']*network['sindy_coefficients']
    
    # TODO: start work from here
    prior_loss = 0
    prior_loss += tf.reduce_mean(tf.square(sindy_coefficients)*network['d_star1'])/(2.0*params['sigma']**2)
    prior_loss += tf.reduce_mean(tf.abs(sindy_coefficients)*network['d_star0'])/(params['sigma'])
    prior_loss -= tf.reduce_sum(tf.log(params['pi']/(1-params['pi'])) * network['p_star'])
    
    return prior_loss

def xi_noise_gaussian(network, params):
    alpha = 0.01
    temp = 10.0
#     noise_std = np.sqrt(2 * alpha * params['learning_rate'])
    noise_std = np.sqrt(2 * alpha)
    sindy_coefficients = params['coefficient_mask']*network['sindy_coefficients']
    noise_ = tf.distributions.Normal(0., temp*noise_std*tf.ones(sindy_coefficients.get_shape()))
    noise_loss = tf.reduce_sum(sindy_coefficients * noise_.sample())
    return noise_loss

def linear_autoencoder(x, input_dim, d):
    # z,encoder_weights,encoder_biases = encoder(x, input_dim, latent_dim, [], None, 'encoder')
    # x_decode,decoder_weights,decoder_biases = decoder(z, input_dim, latent_dim, [], None, 'decoder')
    z,encoder_weights,encoder_biases = build_network_layers(x, input_dim, latent_dim, [], None, 'encoder')
    x_decode,decoder_weights,decoder_biases = build_network_layers(z, latent_dim, input_dim, [], None, 'decoder')

    return z, x_decode, encoder_weights, encoder_biases,decoder_weights,decoder_biases


def nonlinear_autoencoder(x, input_dim, latent_dim, widths, activation='elu'):
    """
    Construct a nonlinear autoencoder.

    Arguments:

    Returns:
        z -
        x_decode -
        encoder_weights - List of tensorflow arrays containing the encoder weights
        encoder_biases - List of tensorflow arrays containing the encoder biases
        decoder_weights - List of tensorflow arrays containing the decoder weights
        decoder_biases - List of tensorflow arrays containing the decoder biases
    """
    if activation == 'relu':
        activation_function = tf.nn.relu
    elif activation == 'elu':
        activation_function = tf.nn.elu
    elif activation == 'sigmoid':
        activation_function = tf.sigmoid
    else:
        raise ValueError('invalid activation function')
    # z,encoder_weights,encoder_biases = encoder(x, input_dim, latent_dim, widths, activation_function, 'encoder')
    # x_decode,decoder_weights,decoder_biases = decoder(z, input_dim, latent_dim, widths[::-1], activation_function, 'decoder')
    z,encoder_weights,encoder_biases = build_network_layers(x, input_dim, latent_dim, widths, activation_function, 'encoder')
    x_decode,decoder_weights,decoder_biases = build_network_layers(z, latent_dim, input_dim, widths[::-1], activation_function, 'decoder')

    return z, x_decode, encoder_weights, encoder_biases, decoder_weights, decoder_biases


def build_network_layers(input, input_dim, output_dim, widths, activation, name):
    """
    Construct one portion of the network (either encoder or decoder).

    Arguments:
        input - 2D tensorflow array, input to the network (shape is [?,input_dim])
        input_dim - Integer, number of state variables in the input to the first layer
        output_dim - Integer, number of state variables to output from the final layer
        widths - List of integers representing how many units are in each network layer
        activation - Tensorflow function to be used as the activation function at each layer
        name - String, prefix to be used in naming the tensorflow variables

    Returns:
        input - Tensorflow array, output of the network layers (shape is [?,output_dim])
        weights - List of tensorflow arrays containing the network weights
        biases - List of tensorflow arrays containing the network biases
    """
    weights = []
    biases = []
    last_width=input_dim
    for i,n_units in enumerate(widths):
        W = tf.get_variable(name+'_W'+str(i), shape=[last_width,n_units],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))
        b = tf.get_variable(name+'_b'+str(i), shape=[n_units],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))
        input = tf.matmul(input, W) + b
        if activation is not None:
            input = activation(input)
        last_width = n_units
        weights.append(W)
        biases.append(b)
    W = tf.get_variable(name+'_W'+str(len(widths)), shape=[last_width,output_dim],
        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))
    b = tf.get_variable(name+'_b'+str(len(widths)), shape=[output_dim],
        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))
    input = tf.matmul(input,W) + b
    weights.append(W)
    biases.append(b)
    return input, weights, biases

def sindy_library_tf(z, latent_dim, poly_order, include_sine=False):
    """
    Build the SINDy library.

    Arguments:
        z - 2D tensorflow array of the snapshots on which to build the library. Shape is number of
        time points by the number of state variables.
        latent_dim - Integer, number of state variable in z.
        poly_order - Integer, polynomial order to which to build the library. Max value is 5.
        include_sine - Boolean, whether or not to include sine terms in the library. Default False.

    Returns:
        2D tensorflow array containing the constructed library. Shape is number of time points by
        number of library functions. The number of library functions is determined by the number
        of state variables of the input, the polynomial order, and whether or not sines are included.
    """
    library = [tf.ones(tf.shape(z)[0])]

    for i in range(latent_dim):
        library.append(z[:,i])

    if poly_order > 1:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                library.append(tf.multiply(z[:,i], z[:,j]))

    if poly_order > 2:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    library.append(z[:,i]*z[:,j]*z[:,k])

    if poly_order > 3:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        library.append(z[:,i]*z[:,j]*z[:,k]*z[:,p])

    if poly_order > 4:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        for q in range(p,latent_dim):
                            library.append(z[:,i]*z[:,j]*z[:,k]*z[:,p]*z[:,q])

    if include_sine:
        for i in range(latent_dim):
            library.append(tf.sin(z[:,i]))

    return tf.stack(library, axis=1)


def sindy_library_tf_order2(z, dz, latent_dim, poly_order, include_sine=False):
    """
    Build the SINDy library for a second order system. This is essentially the same as for a first
    order system, but library terms are also built for the derivatives.
    """
#     library = [tf.ones(tf.shape(z)[0])]
    library = []

    z_combined = tf.concat([z, dz], 1)

    for i in range(2*latent_dim):
        library.append(z_combined[:,i])

    if poly_order > 1:
        for i in range(2*latent_dim):
            for j in range(i,2*latent_dim):
                library.append(tf.multiply(z_combined[:,i], z_combined[:,j]))

    if poly_order > 2:
        for i in range(2*latent_dim):
            for j in range(i,2*latent_dim):
                for k in range(j,2*latent_dim):
                    library.append(z_combined[:,i]*z_combined[:,j]*z_combined[:,k])

    if poly_order > 3:
        for i in range(2*latent_dim):
            for j in range(i,2*latent_dim):
                for k in range(j,2*latent_dim):
                    for p in range(k,2*latent_dim):
                        library.append(z_combined[:,i]*z_combined[:,j]*z_combined[:,k]*z_combined[:,p])

    if poly_order > 4:
        for i in range(2*latent_dim):
            for j in range(i,2*latent_dim):
                for k in range(j,2*latent_dim):
                    for p in range(k,2*latent_dim):
                        for q in range(p,2*latent_dim):
                            library.append(z_combined[:,i]*z_combined[:,j]*z_combined[:,k]*z_combined[:,p]*z_combined[:,q])

    if include_sine:
        for i in range(2*latent_dim):
            library.append(tf.sin(z_combined[:,i]))

    return tf.stack(library, axis=1)


def z_derivative(input, dx, weights, biases, activation='elu'):
    """
    Compute the first order time derivatives by propagating through the network.

    Arguments:
        input - 2D tensorflow array, input to the network. Dimensions are number of time points
        by number of state variables.
        dx - First order time derivatives of the input to the network.
        weights - List of tensorflow arrays containing the network weights
        biases - List of tensorflow arrays containing the network biases
        activation - String specifying which activation function to use. Options are
        'elu' (exponential linear unit), 'relu' (rectified linear unit), 'sigmoid',
        or linear.

    Returns:
        dz - Tensorflow array, first order time derivatives of the network output.
    """
    dz = dx
    if activation == 'elu':
        for i in range(len(weights)-1):
            input = tf.matmul(input, weights[i]) + biases[i]
            dz = tf.multiply(tf.minimum(tf.exp(input),1.0),
                                  tf.matmul(dz, weights[i]))
            input = tf.nn.elu(input)
        dz = tf.matmul(dz, weights[-1])
    elif activation == 'relu':
        for i in range(len(weights)-1):
            input = tf.matmul(input, weights[i]) + biases[i]
            dz = tf.multiply(tf.to_float(input>0), tf.matmul(dz, weights[i]))
            input = tf.nn.relu(input)
        dz = tf.matmul(dz, weights[-1])
    elif activation == 'sigmoid':
        for i in range(len(weights)-1):
            input = tf.matmul(input, weights[i]) + biases[i]
            input = tf.sigmoid(input)
            dz = tf.multiply(tf.multiply(input, 1-input), tf.matmul(dz, weights[i]))
        dz = tf.matmul(dz, weights[-1])
    else:
        for i in range(len(weights)-1):
            dz = tf.matmul(dz, weights[i])
        dz = tf.matmul(dz, weights[-1])
    return dz


def z_derivative_order2(input, dx, ddx, weights, biases, activation='elu'):
    """
    Compute the first and second order time derivatives by propagating through the network.

    Arguments:
        input - 2D tensorflow array, input to the network. Dimensions are number of time points
        by number of state variables.
        dx - First order time derivatives of the input to the network.
        ddx - Second order time derivatives of the input to the network.
        weights - List of tensorflow arrays containing the network weights
        biases - List of tensorflow arrays containing the network biases
        activation - String specifying which activation function to use. Options are
        'elu' (exponential linear unit), 'relu' (rectified linear unit), 'sigmoid',
        or linear.

    Returns:
        dz - Tensorflow array, first order time derivatives of the network output.
        ddz - Tensorflow array, second order time derivatives of the network output.
    """
    dz = dx
    ddz = ddx
    if activation == 'elu':
        for i in range(len(weights)-1):
            input = tf.matmul(input, weights[i]) + biases[i]
            dz_prev = tf.matmul(dz, weights[i])
            elu_derivative = tf.minimum(tf.exp(input),1.0)
            elu_derivative2 = tf.multiply(tf.exp(input), tf.to_float(input<0))
            dz = tf.multiply(elu_derivative, dz_prev)
            ddz = tf.multiply(elu_derivative2, tf.square(dz_prev)) \
                  + tf.multiply(elu_derivative, tf.matmul(ddz, weights[i]))
            input = tf.nn.elu(input)
        dz = tf.matmul(dz, weights[-1])
        ddz = tf.matmul(ddz, weights[-1])
    elif activation == 'relu':
        # NOTE: currently having trouble assessing accuracy of 2nd derivative due to discontinuity
        for i in range(len(weights)-1):
            input = tf.matmul(input, weights[i]) + biases[i]
            relu_derivative = tf.to_float(input>0)
            dz = tf.multiply(relu_derivative, tf.matmul(dz, weights[i]))
            ddz = tf.multiply(relu_derivative, tf.matmul(ddz, weights[i]))
            input = tf.nn.relu(input)
        dz = tf.matmul(dz, weights[-1])
        ddz = tf.matmul(ddz, weights[-1])
    elif activation == 'sigmoid':
        for i in range(len(weights)-1):
            input = tf.matmul(input, weights[i]) + biases[i]
            input = tf.sigmoid(input)
            dz_prev = tf.matmul(dz, weights[i])
            sigmoid_derivative = tf.multiply(input, 1-input)
            sigmoid_derivative2 = tf.multiply(sigmoid_derivative, 1 - 2*input)
            dz = tf.multiply(sigmoid_derivative, dz_prev)
            ddz = tf.multiply(sigmoid_derivative2, tf.square(dz_prev)) \
                  + tf.multiply(sigmoid_derivative, tf.matmul(ddz, weights[i]))
        dz = tf.matmul(dz, weights[-1])
        ddz = tf.matmul(ddz, weights[-1])
    else:
        for i in range(len(weights)-1):
            dz = tf.matmul(dz, weights[i])
            ddz = tf.matmul(ddz, weights[i])
        dz = tf.matmul(dz, weights[-1])
        ddz = tf.matmul(ddz, weights[-1])
    return dz,ddz
