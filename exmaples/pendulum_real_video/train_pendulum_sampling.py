import sys
sys.path.append("../../src")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import datetime
import pandas as pd
import numpy as np
from example_pendulum import get_pendulum_data, load_pendulum_data
from sindy_utils import library_size
from training_pendulum_real_video import train_network
import tensorflow as tf
import random

training_data, validation_data = load_pendulum_data()

print(training_data['x'].shape)

params = {}

params['input_dim'] = training_data['x'].shape[-1]
params['latent_dim'] = 1
params['model_order'] = 2
params['poly_order'] = 3
params['include_sine'] = True
params['library_dim'] = library_size(2*params['latent_dim'], params['poly_order'], params['include_sine'], True) - 1

# sequential thresholding parameters
params['sequential_thresholding'] = True
params['coefficient_threshold'] = 1.0
params['threshold_frequency'] = 500
params['threshold_start'] = 0
params['coefficient_mask'] = np.ones((params['library_dim'], params['latent_dim']))
params['coefficient_initialization'] = 'normal'
# params['coefficient_initialization'] = 'constant'

# loss function weighting original
# change is loss function is necessary
params['loss_weight_decoder'] = 1e3
params['loss_weight_sindy_x'] = 5e-4
params['loss_weight_sindy_z'] = 5e-5

params['activation'] = 'sigmoid'
# params['widths'] = [128,64,32]
params['widths'] = [64,32,16]

# training parameters
params['epoch_size'] = training_data['x'].shape[0]
params['batch_size'] = 10

params['data_path'] = os.getcwd() + '/'
params['print_progress'] = True
params['print_frequency'] = 25

# training time cutoffs
params['max_epochs'] = 4001
params['refinement_epochs'] = 4001

# Bayesian parameters
# Setting of Spike-slab prior
params['prior'] = "spike-and-slab"
params['coefficient_threshold'] = 0.1
params['loss_weight_sindy_regularization'] = 0.1
params['learning_rate'] = 1e-3

# Setting of Laplace prior
# params['prior'] = "laplace"
# params['coefficient_threshold'] = 0.1
# params['loss_weight_sindy_regularization'] = 10.0
# params['learning_rate'] = 4e-3
params['l2_weight'] = 0.1
# params['init_sigma'] = 0.1

params['pi'] = 0.091
params['c_std'] = 5.0
params["epsilon"] = 2.6
params["decay"] = 0.01
params["sigma"] = 0.5
params["csgld"] = 500

num_experiments = 1
df = pd.DataFrame()
for i in range(num_experiments):
    print('EXPERIMENT %d' % i)
    
    print("Hyper-parameters and experimental setup")
    print(params)

    params['coefficient_mask'] = np.ones((params['library_dim'], params['latent_dim']))

    params['save_name'] = 'pendulum_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")

    tf.reset_default_graph()
#     tf.random.set_random_seed(4)
#     random.seed(4)
#     np.random.seed(4)

    results_dict = train_network(training_data, validation_data, params)
    df = df.append({**results_dict, **params}, ignore_index=True)

df.to_pickle('experiment_results_' + datetime.datetime.now().strftime("%Y%m%d%H%M") + '.pkl')
