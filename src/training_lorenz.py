import numpy as np
import tensorflow as tf
import pickle
from autoencoder_lorenz import full_network, define_loss_init
import time, math

def train_network(training_data, val_data, params):
    # SET UP NETWORK
    autoencoder_network = full_network(params)
    loss, losses, loss_refinement = define_loss_init(autoencoder_network, params)
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    train_op_refinement = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_refinement)
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    c_std = params["c_std"]
    epsilon = params["epsilon"]

    v0 = epsilon
    v1 = c_std**2
    pi_val = params["pi"]
    lr = params["learning_rate"]
    temp = 1.0
    count = 0
    loss_weight_sindy_regularization = params['loss_weight_sindy_regularization']

    validation_dict = create_feed_dictionary(val_data, params, idxs=None)

    x_norm = np.mean(val_data['x']**2)
    if params['model_order'] == 1:
        sindy_predict_norm_x = np.mean(val_data['dx']**2)
    else:
        sindy_predict_norm_x = np.mean(val_data['ddx']**2)

    validation_losses = []
    sindy_model_terms = [np.sum(params['coefficient_mask'])]
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    print('TRAINING')
    v1_dist = tf.distributions.Normal(loc=tf.constant(0.0), scale=tf.sqrt(v1))
    v0_dist = tf.distributions.Laplace(loc=tf.constant(0.0), scale=v0)
    save_sindy_coeff = np.zeros((110, 20, 3))

    print('TRAINING')
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(params['max_epochs']):
            if (i % params['print_frequency'] == 0):
                print(sess.run(autoencoder_network['p_star']))
                print(sess.run(autoencoder_network['sindy_coefficients']*params['coefficient_mask']))
#             start_time_huge = time.time()
            for j in range(params['epoch_size']//params['batch_size']):
                batch_idxs = np.arange(j*params['batch_size'], (j+1)*params['batch_size'])
                train_dict = create_feed_dictionary(training_data, params, idxs=batch_idxs)
                sess.run(train_op, feed_dict=train_dict)
                if j == 0 and i % 100 == 0:
                    x_decode = sess.run(autoencoder_network['x_decode'], feed_dict=train_dict)
                    dx_decode = sess.run(autoencoder_network['dx_decode'], feed_dict=train_dict)
                    with open('decode_x.npy', 'wb') as f:
                        np.save(f, x_decode)
                    with open('dx_decode.npy', 'wb') as f:
                        np.save(f, dx_decode)
                    with open('z.npy', 'wb') as f:
                        np.save(f, sess.run(autoencoder_network['z'], feed_dict=train_dict))
                    with open('dz.npy', 'wb') as f:
                        np.save(f, sess.run(autoencoder_network['dz'], feed_dict=train_dict))
                    with open('dz_predict.npy', 'wb') as f:
                        np.save(f, sess.run(autoencoder_network['dz_predict'], feed_dict=train_dict))
            # save file
            if i >= params['max_epochs']-100:
                save_sindy_coeff[count] = sess.run(autoencoder_network['sindy_coefficients'])
                count += 1
            if i >= params['max_epochs']-1:
                with open('save_rd.npy', 'wb') as f:
                    np.save(f, save_sindy_coeff)
                    
            if params['prior'] == "spike-and-slab" and i % 2 == 0:
                sindy_coefficients = autoencoder_network['sindy_coefficients']
                p_star = autoencoder_network['p_star']
                
                log_a_star = v1_dist.log_prob(sindy_coefficients)
                log_b_star = v0_dist.log_prob(sindy_coefficients)
                b_divide_a = tf.exp(log_b_star-log_a_star) * (1-params["pi"]) / params["pi"]
                a_divide_a_and_b = 1.0 / (1.0 + b_divide_a)
                
                p_star = tf.add(tf.multiply((1 - params["decay"]), p_star), tf.multiply(params["decay"], a_divide_a_and_b))
                
                autoencoder_network['d_star0'] = tf.add(tf.multiply((1 - params["decay"]), autoencoder_network['d_star0']), tf.multiply(params["decay"], tf.divide((1 - p_star), v0)))
                autoencoder_network['d_star1'] = tf.add(tf.multiply((1 - params["decay"]), autoencoder_network['d_star1']), tf.multiply(params["decay"], tf.divide(p_star, v1)))
                autoencoder_network['p_star'] = tf.multiply(p_star, params['coefficient_mask'])
                
                if i % 500 == 0:
                    # params["learning_rate"] = 1e-3
                    params["learning_rate"] = 1e-3
                params["decay"] /= (1.0008)
                params["learning_rate"] /= (1.0008)
                
            if params['print_progress'] and (i % params['print_frequency'] == 0):
                validation_losses.append(print_progress(sess, i, loss, losses, train_dict, validation_dict, x_norm, sindy_predict_norm_x))

            if params['sequential_thresholding'] and (i % params['threshold_frequency'] == 0) and (i > 0):
                if params['prior'] == "spike-and-slab":
                    params['coefficient_mask'] = np.abs(sess.run(autoencoder_network['p_star'])) > params['coefficient_threshold']
                if params['prior'] == "laplace":
                    params['coefficient_mask'] = np.abs(sess.run(autoencoder_network['sindy_coefficients'])) > params['coefficient_threshold']
                validation_dict['coefficient_mask:0'] = params['coefficient_mask']
                print('THRESHOLDING: %d active coefficients' % np.sum(params['coefficient_mask']))
                sindy_model_terms.append(np.sum(params['coefficient_mask']))

        print('REFINEMENT')
        for i_refinement in range(params['refinement_epochs']):
            for j in range(params['epoch_size']//params['batch_size']):
                batch_idxs = np.arange(j*params['batch_size'], (j+1)*params['batch_size'])
                train_dict = create_feed_dictionary(training_data, params, idxs=batch_idxs)
                sess.run(train_op_refinement, feed_dict=train_dict)
            
            if params['print_progress'] and (i_refinement % params['print_frequency'] == 0):
                validation_losses.append(print_progress(sess, i_refinement, loss_refinement, losses, train_dict, validation_dict, x_norm, sindy_predict_norm_x))

        saver.save(sess, params['data_path'] + params['save_name'])
        pickle.dump(params, open(params['data_path'] + params['save_name'] + '_params.pkl', 'wb'))
        final_losses = sess.run((losses['decoder'], losses['sindy_x'], losses['sindy_z'],
                                 losses['sindy_regularization']),
                                feed_dict=validation_dict)
        if params['model_order'] == 1:
            sindy_predict_norm_z = np.mean(sess.run(autoencoder_network['dz'], feed_dict=validation_dict)**2)
        else:
            sindy_predict_norm_z = np.mean(sess.run(autoencoder_network['ddz'], feed_dict=validation_dict)**2)
        sindy_coefficients = sess.run(autoencoder_network['sindy_coefficients'], feed_dict={})

        results_dict = {}
        results_dict['num_epochs'] = i
        results_dict['x_norm'] = x_norm
        results_dict['sindy_predict_norm_x'] = sindy_predict_norm_x
        results_dict['sindy_predict_norm_z'] = sindy_predict_norm_z
        results_dict['sindy_coefficients'] = sindy_coefficients
        results_dict['loss_decoder'] = final_losses[0]
        results_dict['loss_decoder_sindy'] = final_losses[1]
        results_dict['loss_sindy'] = final_losses[2]
        results_dict['loss_sindy_regularization'] = final_losses[3]
        results_dict['validation_losses'] = np.array(validation_losses)
        results_dict['sindy_model_terms'] = np.array(sindy_model_terms)

        return results_dict


def print_progress(sess, i, loss, losses, train_dict, validation_dict, x_norm, sindy_predict_norm):
    """
    Print loss function values to keep track of the training progress.
    Arguments:
        sess - the tensorflow session
        i - the training iteration
        loss - tensorflow object representing the total loss function used in training
        losses - tuple of the individual losses that make up the total loss
        train_dict - feed dictionary of training data
        validation_dict - feed dictionary of validation data
        x_norm - float, the mean square value of the input
        sindy_predict_norm - float, the mean square value of the time derivatives of the input.
        Can be first or second order time derivatives depending on the model order.
    Returns:
        Tuple of losses calculated on the validation set.
    """
    training_loss_vals = sess.run((loss,) + tuple(losses.values()), feed_dict=train_dict)
    validation_loss_vals = sess.run((loss,) + tuple(losses.values()), feed_dict=validation_dict)
    print("Epoch %d" % i)
    print("   training loss {0}, {1}".format(training_loss_vals[0],
                                             training_loss_vals[1:]))
    print("   validation loss {0}, {1}".format(validation_loss_vals[0],
                                               validation_loss_vals[1:]))
    decoder_losses = sess.run((losses['decoder'], losses['sindy_x']), feed_dict=validation_dict)
    loss_ratios = (decoder_losses[0]/x_norm, decoder_losses[1]/sindy_predict_norm)
    print("decoder loss ratio: %f, decoder SINDy loss  ratio: %f" % loss_ratios)
    return validation_loss_vals


def create_feed_dictionary(data, params, idxs=None):
    """
    Create the feed dictionary for passing into tensorflow.
    Arguments:
        data - Dictionary object containing the data to be passed in. Must contain input data x,
        along the first (and possibly second) order time derivatives dx (ddx).
        params - Dictionary object containing model and training parameters. The relevant
        parameters are model_order (which determines whether the SINDy model predicts first or
        second order time derivatives), sequential_thresholding (which indicates whether or not
        coefficient thresholding is performed), coefficient_mask (optional if sequential
        thresholding is performed; 0/1 mask that selects the relevant coefficients in the SINDy
        model), and learning rate (float that determines the learning rate).
        idxs - Optional array of indices that selects which examples from the dataset are passed
        in to tensorflow. If None, all examples are used.
    Returns:
        feed_dict - Dictionary object containing the relevant data to pass to tensorflow.
    """
    if idxs is None:
        idxs = np.arange(data['x'].shape[0])
    feed_dict = {}
    feed_dict['x:0'] = data['x'][idxs]
    feed_dict['dx:0'] = data['dx'][idxs]
    if params['model_order'] == 2:
        feed_dict['ddx:0'] = data['ddx'][idxs]
    if params['sequential_thresholding']:
        feed_dict['coefficient_mask:0'] = params['coefficient_mask']
    feed_dict['learning_rate:0'] = params['learning_rate']
    return feed_dict