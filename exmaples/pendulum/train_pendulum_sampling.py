import sys
sys.path.append("../../src")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import datetime
import pandas as pd
import numpy as np
from example_pendulum import get_pendulum_data
from sindy_utils import library_size
from training_pendulum import train_network
import tensorflow as tf
# import tensorflow.compat.v1 as tf

training_data = get_pendulum_data(100)
validation_data = get_pendulum_data(10)

print(training_data['x'].shape)

params = {}

params['input_dim'] = training_data['x'].shape[-1]
params['latent_dim'] = 1
params['model_order'] = 2
params['poly_order'] = 3
params['include_sine'] = True
params['library_dim'] = library_size(2*params['latent_dim'], params['poly_order'], params['include_sine'], True)

# sequential thresholding parameters
params['sequential_thresholding'] = True
params['coefficient_threshold'] = 0.1
params['threshold_frequency'] = 100
params['threshold_start'] = 0
params['coefficient_mask'] = np.ones((params['library_dim'], params['latent_dim']))
# params['coefficient_initialization'] = 'constant'
params['coefficient_initialization'] = 'normal'

# loss function weighting
params['loss_weight_decoder'] = 1.0
params['loss_weight_sindy_x'] = 5e-3
params['loss_weight_sindy_z'] = 5e-5

params['activation'] = 'sigmoid'
params['widths'] = [128,64,32]

# training parameters
params['epoch_size'] = training_data['x'].shape[0]
params['batch_size'] = 1000
# params['learning_rate'] = 1e-3
params['learning_rate'] = 1e-3
# params['learning_rate'] = 

params['data_path'] = os.getcwd() + '/'
params['print_progress'] = True
params['print_frequency'] = 25

# training time cutoffs
params['max_epochs'] = 3001
params['refinement_epochs'] = 1001

# Bayesian parameters
# Setting of Spike-slab prior
params['prior'] = "spike-and-slab"
params['loss_weight_sindy_regularization'] = 0.8e-3

# Setting of Laplace prior
# params['prior'] = "laplace"
# params['loss_weight_sindy_regularization'] = 1e-5
# params['learning_rate'] = 1e-4

params['pi'] = 0.083
params['c_std'] = 3.0
params["epsilon"] = 0.05
params["decay"] = 0.05
params["sigma"] = 1.0
params["init_sigma"] = 0.0
params["cycle_sgld"] = 500

num_experiments = 1
df = pd.DataFrame()
for i in range(num_experiments):
    print('EXPERIMENT %d' % i)
    
    print("Hyper-parameters and experimental setup")
    print(params)

    params['coefficient_mask'] = np.ones((params['library_dim'], params['latent_dim']))

    params['save_name'] = 'pendulum_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")

    tf.reset_default_graph()

    results_dict = train_network(training_data, validation_data, params)
    df = df.append({**results_dict, **params}, ignore_index=True)

df.to_pickle('experiment_results_' + datetime.datetime.now().strftime("%Y%m%d%H%M") + '.pkl')
