import numpy as np
from autoencoder import *


def add_constraints(network, loss, losses, loss_refinement):
    coefficients_flattened = tf.reshape(network['sindy_coefficients'], [-1])
    C = np.zeros((10,30), dtype=np.float32)
    C[0,[8,16,25]] = 1
    C[1,4] = 0
    C[2,17] = 0
    C[3,29] = 0
    C[4,5] = 1
    C[4,14] = -1
    C[5,7] = 1
    C[5,15] = -1
    C[6,6] = 1
    C[6,24] = -1
    C[7,9] = 1
    C[7,26] = -1
    C[8,18] = 1
    C[8,27] = -1
    C[9,19] = 1
    C[9,28] = -1
    constraint_matrix = tf.constant(C)
    lagrange_multiplier = tf.get_variable('lagrange_multiplier', shape=[10],
        initializer=tf.contrib.layers.xavier_initializer())
    network['constraint_matrix'] = constraint_matrix
    network['lagrange_multiplier'] = lagrange_multiplier
    
    constraint_loss = tf.tensordot(lagrange_multiplier,
                                   tf.tensordot(constraint_matrix, coefficients_flattened, axes=1),
                                   axes=1)
    losses['constraint'] = constraint_loss
    loss += constraint_loss
    loss_refinement += constraint_loss

    return network, loss, losses, loss_refinement
