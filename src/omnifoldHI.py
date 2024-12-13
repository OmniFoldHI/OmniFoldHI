import numpy as np
import copy
from matplotlib import pyplot as plt

from keras import regularizers
from keras.layers import Dense, Input
from keras.models import Model

import os
import random
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

def weighted_binary_crossentropy (y_true, y_pred):
    weights = tf.gather(y_true, [1], axis=1) # event weights
    y_true = tf.gather(y_true, [0], axis=1) # actual y_true for loss
    
    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)  # AF: makes sure there are no explosions

    t_loss = -weights * ((y_true) * K.log(y_pred) +
                         (1 - y_true) * K.log(1 - y_pred))
    
    return K.mean(t_loss)

def weighted_binary_crossentropy_square (y_true, y_pred):
    weights = tf.gather(y_true, [1], axis=1) # event weights
    y_true = tf.gather(y_true, [0], axis=1) # actual y_true for loss
    
    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

    t_loss = -weights**2 * ((y_true) * K.log(y_pred) +
                            (1 - y_true) * K.log(1 - y_pred))
    
    return K.mean(t_loss)

def weighted_maximum_likelihood_cl (y_true, y_pred):
    weights = tf.gather(y_true, [1], axis=1) # event weights
    y_true = tf.gather(y_true, [0], axis=1) # actual y_true for loss
    
    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)  # AF: makes sure there are no explosions

    t_loss = -weights * ((y_true) * K.log(y_pred) +
                         (1 - y_true) * (1 - y_pred))
    
    return K.mean(t_loss)

def weighted_maximum_likelihood_cl_square (y_true, y_pred):
    weights = tf.gather(y_true, [1], axis=1) # event weights
    y_true = tf.gather(y_true, [0], axis=1) # actual y_true for loss
    
    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)  # AF: makes sure there are no explosions

    t_loss = -weights**2 * ((y_true) * K.log(y_pred) +
                         (1 - y_true) * (1 - y_pred))
    
    return K.mean(t_loss)

def loss_functional (loss='bce', weights_square=False):
    if loss == 'bce':
        return weighted_binary_crossentropy if not weights_square else weighted_binary_crossentropy_square
    elif loss == 'mlc':
        return weighted_maximum_likelihood_cl if not weights_square else weighted_maximum_likelihood_cl_square

def reweight (events, model, batch_size=10000, loss='bce', weights_square=False):
    f = model.predict(events, batch_size=batch_size)
    if loss == 'bce':
        weights = f / (1. - f)
    elif loss == 'mlc':
        weights = f

    weights_out = np.squeeze(np.nan_to_num(weights))
    if weights_square: weights_out **= .5

    return weights_out

def multifold (evs_list_syn_C, evs_list_syn_E, evs_list_nat_E, val_F, val_T,
               n_iterations=4, weights_syn=[], weights_nat=[], network_seed=239,
               loss='bce', NN1_arch=[100,100], NN2_arch=[100,100], epochs=None, batch_size=10000,
               weights_square=False, ignore_fake=True):
    '''
    Trash from data will automatically be taken out of evs_nat_E.

    Parameters
    ----------
    evs_list_syn_C : `[obsA_evs_syn_C, obsB_evs_syn_C, ...]` list of observables' synthetic cause events
    evs_list_syn_E : `[obsA_evs_syn_E, obsB_evs_syn_E, ...]` list of observables' synthetic effect events
    evs_list_nat_E : `[obsA_evs_nat_E, obsB_evs_nat_E, ...]` list of observables' nature effect events
    val_F : value used for fake cause events
    val_T : value used for trash effect events
    n_iterations : number of unfolding iterations
    weights_syn : array of synthetic event weight
    weights_nat : array of nature event weight
    network_seed : random seed for newtwork reproducibility
    loss : loss function that can be used
        'bce' : Binary Cross Entropy
        'mlc' : Maximum Likelihood classifier
    NN1_arch : `[width_hidden_layer_1, width_hidden_layer_2, ...]` list of widths of network hidden layers for step I of multifold
    NN2_arch : `[width_hidden_layer_1, width_hidden_layer_2, ...]` list of widths of network hidden layers for step II of multifold
    epochs : maximum number of epoch to train each network
    batch_size : number of sample per gradient descent (once all the samples are used, and epoch is completed)

    Returns
    -------
    dictionary with:
    weights : array `[ [weights_pull_it0, weitghs_push_it0], [weights_pull_it1, weitghs_push_it1], [weights_pull_it2, weitghs_push_it2], ... ]`
    train_loss : array `[ [NN1_train_loss_it0, NN2_train_loss_it0], [NN1_train_loss_it1, NN2_train_loss_it1], [NN1_train_loss_it2, NN2_train_loss_it2], ... ]`
    val_loss : same but val_loss
    train_accuracy : same but train_accuracy
    val_accuracy : same but val_accuracy

    axis0 for each iteration;
    axis1 for step I and II

    Use
    ---
    Final unfolding weight to use are in `weights[multifold_it,1]`
    '''

    # Initialization seed and enabeling deterministic operations
    random.seed(network_seed)
    np.random.seed(network_seed+934)
    tf.random.set_seed(network_seed+534)
    #tf.config.experimental.enable_op_determinism()

    # GPU usage
    if len(tf.config.experimental.list_physical_devices('GPU')):
        print('Using available GPU.')

    # NN architectures
    # Choice of hidden layers widths
    NN1_hidden_widths = NN1_arch
    NN2_hidden_widths = NN2_arch
    
    ###
    # NN inputs
    NN1_input = Input((len(evs_list_syn_C), ))
    NN2_input = Input((len(evs_list_syn_C), ))

    ###
    # NN hidden layers
    NN1_hidden_layers = [ Dense(NN1_hidden_widths[0], activation='relu',)(NN1_input) ]
    for width in NN1_hidden_widths[1:]:
        NN1_hidden_layers.append( Dense(width, activation='relu')(NN1_hidden_layers[-1]) )

    NN2_hidden_layers = [ Dense(NN2_hidden_widths[0], activation='relu')(NN2_input) ]
    for width in NN2_hidden_widths[1:]:
        NN2_hidden_layers.append( Dense(width, activation='relu')(NN2_hidden_layers[-1]) )

    ###
    # NN outputs and model
    NN1_output = Dense(1, activation='sigmoid')(NN1_hidden_layers[-1])
    NN2_output = Dense(1, activation='sigmoid')(NN2_hidden_layers[-1])
    NN1_model = Model(inputs=NN1_input, outputs=NN1_output)
    NN2_model = Model(inputs=NN2_input, outputs=NN2_output)

    ###
    # NN training hyperparameters
    earlystopping = EarlyStopping(patience=10,
                                  verbose=0,
                                  restore_best_weights=True)
    if not epochs:
        epochs = 200
        callbacks = [earlystopping]
    else:
        callbacks = None
    

    # Events
    # deep copy inputs to not change the originals and transpose to serve as input
    events_gen = np.array(evs_list_syn_C).T *1
    events_sim = np.array(evs_list_syn_E).T *1
    events_dat = np.array(evs_list_nat_E).T *1

    ###
    # correct val_FT
    val_FT = -10
    events_gen[events_gen[:,0]==val_F] = val_FT
    events_sim[events_sim[:,0]==val_T] = val_FT

    ###
    # standardize inputs
    X_1 = np.concatenate((events_sim, events_dat))
    X_2 = np.concatenate((events_gen, events_gen))
    mean_x_1, std_x_1 = (np.mean(X_1[X_1[:,0]!=val_FT], axis=0),np.std(X_1[X_1[:,0]!=val_FT], axis=0))
    mean_x_2, std_x_2 = (np.mean(X_2[X_2[:,0]!=val_FT], axis=0),np.std(X_2[X_2[:,0]!=val_FT], axis=0))
    events_gen[events_gen[:,0]!=val_FT] = (events_gen[events_gen[:,0]!=val_FT]-mean_x_2)/std_x_2
    events_sim[events_sim[:,0]!=val_FT] = (events_sim[events_sim[:,0]!=val_FT]-mean_x_1)/std_x_1
    events_dat[events_dat[:,0]!=val_FT] = (events_dat[events_dat[:,0]!=val_FT]-mean_x_1)/std_x_1

    # Weights
    # initial weights
    if len(weights_syn):
        weights_pull = weights_syn *1
        weights_push = weights_syn *1
    else:
        weights_pull = np.ones(len(events_sim))
        weights_push = np.ones(len(events_gen))

    ###
    # data weights
    if len(weights_nat):
        weights_dat = weights_nat *1
    else:
        weights_dat = np.ones(len(events_dat))

    # Make sure no trash in dat
    weights_dat = weights_dat[events_dat[:,0]!=val_T]
    events_dat  = events_dat[events_dat[:,0]!=val_T]

    # Create output arrays
    # multifold weights
    weights = np.empty( (n_iterations+1, 2, len(weights_syn)) )
    weights[0] = (weights_pull, weights_push)

    ###
    # performance metrics
    train_loss     = np.full( (n_iterations+1, 2, epochs), np.nan )
    val_loss       = np.full( (n_iterations+1, 2, epochs), np.nan )
    train_accuracy = np.full( (n_iterations+1, 2, epochs), np.nan )
    val_accuracy   = np.full( (n_iterations+1, 2, epochs), np.nan )

    # Weight iteration
    for i in range(n_iterations):

        print(f"\nIteration: {i+1}/{n_iterations}")

        print("\nstep I")

        # Events and weights to use for step I
        # X_1 -> (sim,dat)
        # Y_1 -> ( 0 , 1 )
        mask = events_sim[:,0] != val_FT
        xvals_1 = np.concatenate((events_sim[mask],events_dat))
        yvals_1 = np.concatenate((np.zeros(len(events_sim[mask])),np.ones(len(events_dat))))
        weights_1 = np.concatenate((weights_push[mask], weights_dat))

        # train and test sets
        X_train_1, X_test_1, Y_train_1, Y_test_1, w_train_1, w_test_1 = train_test_split(xvals_1, yvals_1, weights_1)

        # zip ("hide") the weights with the labels
        Y_train_1 = np.stack((Y_train_1, w_train_1), axis=1)
        Y_test_1 = np.stack((Y_test_1, w_test_1), axis=1)
        
        # NN1 initialization and training
        NN1_model.compile(loss=loss_functional(loss, weights_square),
                          optimizer='Adam',
                          metrics=['accuracy'],
                          weighted_metrics=[])
        NN1_history = NN1_model.fit(X_train_1,
                                    Y_train_1,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    validation_data=(X_test_1, Y_test_1),
                                    callbacks=callbacks,
                                    verbose=1)

        # Step I event reweighting
        weights_pull = reweight(events_sim, NN1_model, batch_size, loss, weights_square)
        weights_pull[events_sim[:,0]==val_FT] = 0

        # Save weights
        weights[i+1, 0] = weights_pull

        # Save NN1 performace metrics
        epoch1_length = len(NN1_history.history['loss'])
        train_loss[i+1,0][:epoch1_length]     = NN1_history.history['loss']
        val_loss[i+1,0][:epoch1_length]       = NN1_history.history['val_loss']
        train_accuracy[i+1,0][:epoch1_length] = NN1_history.history['accuracy']
        val_accuracy[i+1,0][:epoch1_length]   = NN1_history.history['val_accuracy']

        # Print info
        print('Normalization deviation - step 1:',np.mean(np.sum(weights_dat) / np.sum(weights_pull)))

        print('\nstep II')

        # Events and weights to use for step II
        # X_2 -> (gen,gen)
        # Y_2 -> ( 0 , 1 )
        mask_sim = events_sim[:,0] != val_FT
        mask_gen = events_gen[:,0] != val_FT
        #mask     = mask_sim  # SEE DIFFERENCE LATER WHEN USEING THIS MASK INSTEAD!!!
        mask     = mask_sim * mask_gen if ignore_fake else mask_sim
        xvals_2 = np.concatenate([events_gen[mask],events_gen[mask]])
        yvals_2 = np.concatenate([np.zeros(len(events_gen[mask])),np.ones(len(events_gen[mask]))])
        weights_2 = np.concatenate([np.ones(len(events_gen[mask])),weights_pull[mask]*weights_push[mask]])

        # train and test sets
        X_train_2, X_test_2, Y_train_2, Y_test_2, w_train_2, w_test_2 = train_test_split(xvals_2, yvals_2, weights_2)

        # zip ("hide") the weights with the labels
        Y_train_2 = np.stack((Y_train_2, w_train_2), axis=1)
        Y_test_2 = np.stack((Y_test_2, w_test_2), axis=1)
        
        # NN2 initialization and training
        NN2_model.compile(loss=loss_functional(loss, weights_square),
                          optimizer='Adam',
                          metrics=['accuracy'],
                          weighted_metrics=[])
        NN2_history = NN2_model.fit(X_train_2,
                                    Y_train_2,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    validation_data=(X_test_2, Y_test_2),
                                    callbacks=callbacks,
                                    verbose=1)
        
        # Step II event reweighting
        if ignore_fake:
            print('Ignoring fakes')
            weights_push[~mask_gen*~mask_sim] = np.mean( weights_pull[~mask_gen*mask_sim]*weights_push[~mask_gen*mask_sim] )
            weights_push[~mask_gen* mask_sim] = weights_pull[~mask_gen*mask_sim]*weights_push[~mask_gen*mask_sim]
            weights_push[ mask_gen] = reweight(events_gen[mask_gen], NN2_model, batch_size, loss, weights_square)
        else:
            print('Standard step II')
            weights_push = reweight(events_gen, NN2_model, batch_size, loss, weights_square)

        # Save weights
        weights[i+1, 1] = weights_push

        # Save NN2 performace metrics
        epoch2_length = len(NN2_history.history['loss'])
        train_loss[i+1,1][:epoch2_length]     = NN2_history.history['loss']
        val_loss[i+1,1][:epoch2_length]       = NN2_history.history['val_loss']
        train_accuracy[i+1,1][:epoch2_length] = NN2_history.history['accuracy']
        val_accuracy[i+1,1][:epoch2_length]   = NN2_history.history['val_accuracy']

        # Print info
        print('Normalization deviation - step 2:',np.mean(np.sum(weights_pull[mask]) / np.sum(weights_push[mask])))

    # MultiFold output
    return {
        'weights': weights,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy
    }

def multifold_unc (evs_list_syn_C, evs_list_syn_E, evs_list_nat_E, val_F, val_T,
                   n_iterations=4, loss='bce',
                   NN1_arch=[100,100], NN2_arch=[100,100], epochs=None, batch_size=10000,
                   unc_n_stat=2, unc_n_syst=1):
    '''
    Parameters
    ----------
    evs_list_syn_C : `[obsA_evs_syn_C, obsB_evs_syn_C, ...]` list of observables' synthetic cause events
    evs_list_syn_E : `[obsA_evs_syn_E, obsB_evs_syn_E, ...]` list of observables' synthetic effect events
    evs_list_nat_E : `[obsA_evs_nat_E, obsB_evs_nat_E, ...]` list of observables' nature effect events
    val_F : value used for fake cause events
    val_T : value used for trash effect events
    n_iterations : number of unfolding iterations
    model_step1 : network architectrure used for step 1 of omnifold
    model_step2 : network architectrure used for step 2 of omnifold
    loss : loss function that can be used
        'bce' : Binary Cross Entropy
        'mlc' : Maximum Likelihood classifier
    unc_n_stat : number of omnifold runs to compute statistical uncertainty (minimum 1 for one omnifold run)
    unc_n_syst : number of omnifold runs to compute systematic uncertainty

    Returns
    -------
    `[weights_stat, weights_syst]`

    weights_stat : `[weigths_stat1, weights_stat2, ...]` list of weights from `omnifold_weights()` for statistical uncertaity (if only one presented, uncertainty comes from histo variance)
    weights_syst : `[weigths_syst1, weights_syst2, ...]` list of weights from `omnifold_weights()` for systematic uncertaity

    Use
    ---
    As parameters in `weights_to_histo (evs_list_syn_C, evs_list_syn_E, bins_list_C, weights_stat, weights_syst, val_F, val_T)`
    '''

    # Statistical uncertainty
    weights_stat = []
    for i in range(unc_n_stat):

        # Set weights
        np.random.seed(i)
        network_seed = 1
        weights_syn  = np.ones(len(evs_list_syn_E[0]))

        if ( i>0 and unc_n_stat<=2 ):
            print(f'\nmultifold uncertainty stat: run {i+1}/{unc_n_stat} - unfolding weights squared')
            weights_nat = np.ones(len(evs_list_nat_E[0]))
            weights_square = True
        else:
            print(f'\nmultifold uncertainty stat: run {i+1}/{unc_n_stat}')
            weights_nat = np.random.poisson(1, size=len(evs_list_nat_E[0])) if i>0 else np.ones(len(evs_list_nat_E[0]))
            weights_square = False

        # Compute onmifold weights and append
        multifold_out = multifold(evs_list_syn_C, evs_list_syn_E, evs_list_nat_E, val_F, val_T,
                                  n_iterations=n_iterations, weights_syn=weights_syn, weights_nat=weights_nat, network_seed=network_seed,
                                  loss=loss, NN1_arch=NN1_arch, NN2_arch=NN2_arch, epochs=epochs, batch_size=batch_size,
                                  weights_square=weights_square)

        weights = multifold_out['weights']
        weights_stat.append(weights)

        # Performance metrics from 1st stat run
        if i==0:
            train_loss     = multifold_out['train_loss']
            val_loss       = multifold_out['val_loss']
            train_accuracy = multifold_out['train_accuracy']
            val_accuracy   = multifold_out['val_accuracy']

    weights_stat = np.array(weights_stat)

    # Systematic uncertainty
    weights_syst = [weights_stat[0]]  # The first run of stat can count for the syst
    np.random.seed(8)
    for i in range(unc_n_syst-1):  # -1 because we arleady have one run done from stat
        print(f'\nmultifold uncertainty syst: run {i+2}/{unc_n_syst}')

        # Set weights
        weights_syn  = np.ones(len(evs_list_syn_E[0]))
        weights_nat  = np.ones(len(evs_list_nat_E[0]))
        network_seed = np.random.randint(100)

        # Compute onmifold weights and append
        multifold_out = multifold(evs_list_syn_C, evs_list_syn_E, evs_list_nat_E, val_F, val_T,
                                  n_iterations=n_iterations, weights_syn=weights_syn, weights_nat=weights_nat, network_seed=network_seed,
                                  loss=loss, NN1_arch=NN1_arch, NN2_arch=NN2_arch, epochs=epochs, batch_size=batch_size,
                                  weights_square=False)

        weights = multifold_out['weights']
        weights_syst.append(weights)

    weights_syst = np.array(weights_syst)

    return {
        'weights_stat': weights_stat,
        'weights_syst': weights_syst,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy
    }

def multifold_unc_from_dict (dict_data, obs_unfold,
                             n_iterations=4, loss='bce',
                             NN1_arch=[100,100], NN2_arch=[100,100], epochs=None, batch_size=10000,
                             unc_n_stat=2, unc_n_syst=1):

    evs_list_syn_C = [ dict_data[obs_label]['evs_syn_C'] for obs_label in obs_unfold ]
    evs_list_syn_E = [ dict_data[obs_label]['evs_syn_E'] for obs_label in obs_unfold ]
    evs_list_nat_E = [ dict_data[obs_label]['evs_nat_E'] for obs_label in obs_unfold ]
    val_F          = dict_data[obs_unfold[0]]['val_F']
    val_T          = dict_data[obs_unfold[0]]['val_T']

    return multifold_unc(evs_list_syn_C, evs_list_syn_E, evs_list_nat_E, val_F, val_T,
                         n_iterations=n_iterations, loss=loss,
                         NN1_arch=NN1_arch, NN2_arch=NN2_arch, epochs=epochs, batch_size=batch_size,
                         unc_n_stat=unc_n_stat, unc_n_syst=unc_n_syst)

def weights_to_histo (evs_list_syn_C, evs_list_syn_E, val_F, val_T,
                      bins_list_C, weights_stat, weights_syst, multifold_it=-1):
    '''
    Parameters
    ----------
    evs_list_syn_C : `[obsA_evs_syn_C, obsB_evs_syn_C, ...]` list of observables' synthetic cause events
    evs_list_syn_E : `[obsA_evs_syn_E, obsB_evs_syn_E, ...]` list of observables' synthetic effect events
    bins_list_C : `[binsA_C, binsB_C, ...]` list of observables' bins
    weights_stat : `[weigths_stat1, weights_stat2, ...]` list of weights from `omnifold_weights()` for statistical uncertaity (if only one presented, uncertainty comes from histo variance)
    weights_syst : `[weigths_syst1, weights_syst2, ...]` list of weights from `omnifold_weights()` for systematic uncertaity
    val_F : value used for fake cause events
    val_T : value used for trash effect events

    Returns
    -------
    !!! NOT UP TO DATE !!!
    `[hist_unf_C, hist_unf_F, hist_unf_T]`

    hist_unf_C : `[hist_list_unf_C, hist_list_unf_C_unc_stat, hist_list_unf_C_unc_syst, bins_list_C]` histogram info for unfolded causes
        hist_list_unf_C : `[obsA_hist_unf_C, obsB_hist_unf_C, ...]` list of observables' histograms of unfolded causes
        hist_list_unf_C_unc_stat : `[obsA_hist_unf_C_unc_stat, obsB_hist_unf_C_unc_stat, ...]` list of observables' respective statistical uncertainties
        hist_list_unf_C_unc_syst : `[obsA_hist_unf_C_unc_syst, obsB_hist_unf_C_inc_syst, ...]` list of observables' respective systematic uncertainties
        bins_list_C : `[binsA_C, binsB_C, ...]` list of observables' bins

    hist_unf_F : `[hist_list_unf_F, hist_list_unf_F_unc_stat, hist_list_unf_F_unc_syst]` histogram info for unfolded fakes
        hist_list_unf_F : `[obsA_hist_unf_F, obsB_hist_unf_F, ...]` list of observables' histograms of unfolded fakes
        hist_list_unf_F_unc_stat : `[obsA_hist_unf_F_unc_stat, obsB_hist_unf_F_unc_stat, ...]` list of observables' respective statistical uncertainties
        hist_list_unf_F_unc_syst : `[obsA_hist_unf_F_unc_syst, obsB_hist_unf_F_inc_syst, ...]` list of observables' respective systematic uncertainties

    hist_unf_T : `[hist_list_unf_T, hist_list_unf_T_unc_stat, hist_list_unf_T_unc_syst]` histogram info for unfolded trash
        hist_list_unf_T : `[obsA_hist_unf_T, obsB_hist_unf_T, ...]` list of observables' histograms of unfolded trash
        hist_list_unf_T_unc_stat : `[obsA_hist_unf_T_unc_stat, obsB_hist_unf_T_unc_stat, ...]` list of observables' respective statistical uncertainties
        hist_list_unf_T_unc_syst : `[obsA_hist_unf_T_unc_syst, obsB_hist_unf_T_inc_syst, ...]` list of observables' respective systematic uncertainties
    '''

    histograms_data = []

    for evs_syn_C, evs_syn_E, bins_C in zip(evs_list_syn_C, evs_list_syn_E, bins_list_C):

        # bins
        binwidth_C   = bins_C[1] - bins_C[0]  # assuming linear binning
        midbin_C     = (bins_C[:-1]+bins_C[1:])/2
        midbin_C_unc = binwidth_C/2 * np.ones(len(midbin_C))

        # bins to return
        midbin_FTC        = np.hstack([ val_F , val_T , midbin_C ])
        midbin_FTC_unc    = np.hstack([ 0     , 0     , midbin_C_unc ])

        # statistical uncertaity

        # compute histograms and uncertainty w/ normalization
        if len(weights_stat) <= 2:
            # uncertainty stat comes from sum w^2 in case len(weights_stat) == 1
            # uncertianty comes from sum w^2 of unfolded (w^2) in case len(weights_stat) == 2

            # values for normalization
            # (only the first weights are used to compute normalization)
            nevs_C_stat = np.sum( weights_stat[0][multifold_it,1] )
            nevs_E_stat = np.sum( weights_stat[0][multifold_it,1][evs_syn_E!=val_T] )

            hist_C_stat      = np.array([ np.histogram(evs_syn_C, bins_C, weights=weights_stat[0][multifold_it,1])[0] ])  # array so it can be stacked for hist_C
            hist_C_stat     /= binwidth_C * nevs_C_stat
            hist_C_stat_unc  = np.sqrt( np.histogram(evs_syn_C, bins_C, weights=weights_stat[-1][multifold_it,1]**2)[0] )
            hist_C_stat_unc /= binwidth_C * nevs_C_stat

            hist_F_stat      = np.array([ np.histogram(evs_syn_C, bins=[val_F-.01, val_F+.01], weights=weights_stat[0][multifold_it,1])[0] ])  # array so it can be stacked for hist_F
            hist_F_stat     /= nevs_C_stat
            hist_F_stat_unc  = np.sqrt( np.histogram(evs_syn_C, bins=[val_F-.01, val_F+.01], weights=weights_stat[-1][multifold_it,1]**2)[0] )
            hist_F_stat_unc /= nevs_C_stat

            hist_T_stat      = np.array([ np.histogram(evs_syn_E, bins=[val_T-.01, val_T+.01], weights=weights_stat[0][multifold_it,1])[0] ])  # array so it can be stacked for hist_T
            hist_T_stat     /= nevs_E_stat
            hist_T_stat_unc  = np.sqrt( np.histogram(evs_syn_E, bins=[val_T-.01, val_T+.01], weights=weights_stat[-1][multifold_it,1]**2)[0] )
            hist_T_stat_unc /= nevs_E_stat
        else: # uncertainty stat comes from std of histograms

            # values for normalization
            nevs_C_stat = np.array([ np.sum( weights[multifold_it,1] ) for weights in weights_stat ])
            nevs_E_stat = np.array([ np.sum( weights[multifold_it,1][evs_syn_E!=val_T] ) for weights in weights_stat ])

            hist_C_stat     = np.array([ np.histogram(evs_syn_C, bins_C, weights=weights[multifold_it,1])[0] for weights in weights_stat ])
            hist_C_stat    /= binwidth_C * nevs_C_stat[:,None]  # [:,None] to expand dimensions: (n_hists,) -> (n_hists,1)
            hist_C_stat_unc = np.std (hist_C_stat, axis=0)

            hist_F_stat     = np.array([ np.histogram(evs_syn_C, bins=[val_F-.01, val_F+.01], weights=weights[multifold_it,1])[0] for weights in weights_stat ])
            hist_F_stat    /= nevs_C_stat[:,None]
            hist_F_stat_unc = np.std (hist_F_stat, axis=0)

            hist_T_stat     = np.array([ np.histogram(evs_syn_E, bins=[val_T-.01, val_T+.01], weights=weights[multifold_it,1])[0] for weights in weights_stat ])
            hist_T_stat    /= nevs_E_stat[:,None]
            hist_T_stat_unc = np.std (hist_T_stat, axis=0)

        ###
        # histogram uncertainty stat to return
        hist_FTC_unc_stat = np.hstack([ hist_F_stat_unc , hist_T_stat_unc , hist_C_stat_unc ])

        # systematic uncertainty
        # values for normalization
        nevs_C_syst = np.array([ np.sum( weights[multifold_it,1] ) for weights in weights_syst ])
        nevs_E_syst = np.array([ np.sum( weights[multifold_it,1][evs_syn_E!=val_T] ) for weights in weights_syst ])

        ###
        # compute histograms and uncertainty w/ normalization
        hist_C_syst     = np.array([ np.histogram(evs_syn_C, bins_C, weights=weights[multifold_it,1])[0] for weights in weights_syst ])
        hist_C_syst    /= binwidth_C * nevs_C_syst[:,None]
        hist_C_syst_unc = np.std (hist_C_syst, axis=0)

        hist_F_syst     = np.array([ np.histogram(evs_syn_C, bins=[val_F-.01, val_F+.01], weights=weights[multifold_it,1])[0] for weights in weights_syst ])
        hist_F_syst    /= nevs_C_syst[:,None]
        hist_F_syst_unc = np.std (hist_F_syst, axis=0)
        
        hist_T_syst     = np.array([ np.histogram(evs_syn_E, bins=[val_T-.01, val_T+.01], weights=weights[multifold_it,1])[0] for weights in weights_syst ])
        hist_T_syst    /= nevs_E_syst[:,None]
        hist_T_syst_unc = np.std (hist_T_syst, axis=0)

        ###
        # histograms to return
        hist_FTC_unc_syst = np.hstack([ hist_F_syst_unc , hist_T_syst_unc , hist_C_syst_unc ])

        # histogram means
        hist_C      = np.vstack([ hist_C_stat, hist_C_syst[1:] ]) if len(hist_C_syst)>1 else hist_C_stat  # 1st weights_stat = 1st weitghs_syst
        hist_C_mean = np.mean(hist_C, axis=0)

        hist_F      = np.vstack([ hist_F_stat, hist_F_syst[1:] ]) if len(hist_F_syst)>1 else hist_F_stat
        hist_F_mean = np.mean(hist_F, axis=0)

        hist_T      = np.vstack([ hist_T_stat, hist_T_syst[1:] ]) if len(hist_T_syst)>1 else hist_T_stat
        hist_T_mean = np.mean(hist_T, axis=0)

        hist_FTC_mean = np.hstack([ hist_F_mean, hist_T_mean, hist_C_mean ])

        # append to observables' list of histograms
        histograms_data.append( np.vstack([ midbin_FTC, midbin_FTC_unc, hist_FTC_mean, hist_FTC_unc_stat, hist_FTC_unc_syst ]) )

    return histograms_data

def create_dict_multifoldHI (dict_data, weights_stat, weights_syst, multifold_it=-1):

    # create dictionary with unfolded info and histograms
    dict_multifoldHI = {}

    # copy relevant info from dict_data to dict_ibu, so dict_ibu can bve used alone
    # (unfolded histograms are created for all available histograms, and not only the ones used in multifold)
    dict_data_obs = list(dict_data.keys())
    keys_to_copy = ['xlim_C', 'xlim_E', 'xlabel', 'ylabel', 'val_F', 'val_T',
                    'bins_C', 'bins_E', 'midbin_C', 'midbin_E', 'midbin_C_unc', 'midbin_E_unc']
    for obs_label in dict_data_obs:
        dict_multifoldHI[obs_label] = { key: copy.deepcopy(dict_data[obs_label][key]) for key in keys_to_copy }

    # histogram for all the observables in dict_data
    dict_data_evs_list_syn_C = [ dict_data[obs_label]['evs_syn_C'] for obs_label in dict_data_obs ]
    dict_data_evs_list_syn_E = [ dict_data[obs_label]['evs_syn_E'] for obs_label in dict_data_obs ]         
    dict_data_bins_list_C    = [ dict_data[obs_label]['bins_C']    for obs_label in dict_data_obs ]

    # Fake and trash values
    val_F = dict_data[obs_label]['val_F']
    val_T = dict_data[obs_label]['val_T']

    histograms = weights_to_histo(dict_data_evs_list_syn_C, dict_data_evs_list_syn_E, val_F, val_T,
                                  dict_data_bins_list_C, weights_stat, weights_syst, multifold_it)
    
    for obs_label, histo in zip(dict_data_obs, histograms):
        
        dict_multifoldHI[obs_label]['hist_unf_C'] = histo[2][2:]
        dict_multifoldHI[obs_label]['hist_unf_C_unc_stat'] = histo[3][2:]
        dict_multifoldHI[obs_label]['hist_unf_C_unc_syst'] = histo[4][2:]

        dict_multifoldHI[obs_label]['hist_unf_F'] = histo[2][0]
        dict_multifoldHI[obs_label]['hist_unf_F_unc_stat'] = histo[3][0]
        dict_multifoldHI[obs_label]['hist_unf_F_unc_syst'] = histo[4][0]

        dict_multifoldHI[obs_label]['hist_unf_T'] = histo[2][1]
        dict_multifoldHI[obs_label]['hist_unf_T_unc_stat'] = histo[3][1]
        dict_multifoldHI[obs_label]['hist_unf_T_unc_syst'] = histo[4][1]

    return dict_multifoldHI