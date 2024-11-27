#!/usr/bin/env python
# coding: utf-8

#Import the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import csv
from math import ceil
import recodeA
import autoencoder as ae
ae.silence_tensorflow()
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, LeakyReLU, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K

from sklearn.model_selection import StratifiedKFold

# The range of lambda_loss values to try: We are going to obtain 5-fold cross validated validation performances 
# for lambda values between 0-1. These lambda values correspond to the weight of the classifier's binary crossentropy loss
# whereas the weight of each autoencoder's mse loss is equally set to 1-lambda while training the PhenoEncoder.
lambda_loss_values = np.round(np.arange(0.1, 0.35, 0.1), 1)

cnf = 8
run = 3
print("Tensorflow version is:", tf.__version__)

verbosity = 2 #one line per epoch verbosity mode is recommended when not running interactively

# Uncomment below if running on a GPU node.
# Avoiding tensorflow to allocate all the GPU memory
print(f"Number of GPU: {len(tf.config.experimental.list_physical_devices('GPU'))}")
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

print("Loading the recoded data:")
data = recodeA.read_raw("../data/cnf-0{}-run-{}_TruthBlocks_recoded.raw".format(cnf,run)).impute("mode")

data.set_index('IID', inplace=True)
data.drop(columns=['FID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE'], inplace=True)

# Read the pre-defined train and test IDs:
train_IDs = pd.read_csv("../data/cnf-0{}-run-{}.train_IDs.txt".format(cnf,run), sep = " ", header=None)
train_indices = train_IDs.iloc[:,0].values.tolist()

test_IDs = pd.read_csv("../data/cnf-0{}-run-{}.test_IDs.txt".format(cnf,run), sep = " ", header=None)
test_indices = test_IDs.iloc[:,0].values.tolist()

train_raw = data.loc[train_indices]
test_raw = data.loc[test_indices]

def remove_after_(column_name):
    '''
    The recoded raw data has column names with rsid + _ + alternative alleles i.e. 'rs79479029_A' 
    or 'rs190598378;rs546797193_TA' & 'rs190598378;rs546797193_TA' as 2 different columns.
    We need to carefully match the column names with the rsids in the block file.
    '''
    underscore_index = column_name.find('_')
    return column_name[:underscore_index]

train_raw.columns = train_raw.columns.map(remove_after_)
test_raw.columns = test_raw.columns.map(remove_after_)

#Reading the simulated PEPS Phenotypes:
pheno = pd.read_csv("../data/PEPS-Phenotypes/cnf-0{}-run-{}.pheno.csv".format(cnf,run), header=0, index_col='sample')
train_pheno = pheno.loc[train_indices]
test_pheno = pheno.loc[test_indices]

max_bn = 3 # Dimensions of the autoencoder bottlenecks
print("Each block will be encoded into at most {} dimensions.".format(max_bn))

# Custom activation function for the autoencoder ouput layers, tanh+1, to map the output to 0, 1 or 2, is defined here.
get_custom_objects().update({'additive': Activation(ae.additive)})         

def model_builder(X_train):
    K.clear_session()
    inputs=[]
    bottlenecks=[]
    outputs=[]
    
    for i in range(no_of_blocks):  
        block_size = X_train['block{}'.format(i+1)].shape[1] # Number of SNPs in block i
        block_input = Input(shape=(block_size,), name='block{}_input'.format(i+1))
        inputs.append(block_input)  

        bn = min(max_bn, ceil(block_size/10))
        encoder, decoder = ae.model_shape(inputs=block_size, shape='sqrt', hl=4, bn=bn)

        x = LeakyReLU(alpha=0.01)(Dense(units=encoder[0], kernel_initializer='he_uniform')(block_input))

        for nodes in encoder[1:]:
            x = LeakyReLU(alpha=0.01)(Dense(units=nodes, kernel_initializer='he_uniform')(x))

        x = LeakyReLU(alpha=0.01, name='block{}_bottleneck'.format(i+1))(Dense(units=bn, kernel_initializer='he_uniform')(x)) 
        bottlenecks.append(x)
        
        for nodes in decoder:
            x = LeakyReLU(alpha=0.01)(Dense(units=nodes, kernel_initializer='he_uniform')(x))
    
        block_output = Activation(ae.additive, name='block{}_output'.format(i+1))(Dense(units=block_size)(x))
        outputs.append(block_output)

    concatenated_output = Concatenate(name='concatenated_output')(outputs)

    merged_bns = Concatenate()(bottlenecks)
    case_control = LeakyReLU(alpha=0.01)(Dense(units=32, kernel_initializer='he_uniform')(merged_bns))
    case_control = Dropout(0.1)(case_control)
    case_control = LeakyReLU(alpha=0.01)(Dense(units=8, kernel_initializer='he_uniform')(case_control)) 
    pheno_class = Dense(1, activation='sigmoid', name='pheno_class')(case_control)
    
    autoencoders = Model(inputs=inputs, outputs=concatenated_output, name='autoencoders')
    encoders = Model(inputs=inputs, outputs=bottlenecks, name='encoders')
    classifier = Model(inputs=bottlenecks, outputs=pheno_class, name='classifier')
    PhenoEncoder = Model(inputs=inputs, outputs=[concatenated_output, pheno_class], name='PhenoEncoder')

    return autoencoders, encoders, classifier, PhenoEncoder

# Function to freeze/unfreeze layers in a model
def set_trainable(model, trainable):
    for layer in model.layers:
        layer.trainable = trainable

# Compile autoencoders with mse losses
def compile_autoencoders():
    set_trainable(autoencoders, True)
    set_trainable(classifier, False)
    losses = {}
    losses['concatenated_output'] = 'mse'

    _metrics = {}
    _metrics['concatenated_output'] = ae.snp_accuracy

    autoencoders.compile(optimizer=Adam(learning_rate=0.0001), loss=losses, metrics=_metrics)

# Compile classifier with binary crossentropy loss
def compile_classifier():
    set_trainable(autoencoders, False)
    set_trainable(classifier, True)

    classifier.compile(optimizer=Adam(learning_rate=0.0001), 
                       loss='binary_crossentropy', 
                       metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                                tf.keras.metrics.Precision(name='precision'),
                                tf.keras.metrics.Recall(name='recall'),
                                tf.keras.metrics.AUC(name='auc'),
                                tf.keras.metrics.AUC(name='prc', curve='PR')
                                ])

def compile_phenoencoder(lambda_loss=0.5):
    '''
    Compile PhenoEncoder model with multiple losses
    Autoencoders' mean squared error loss is weighted equally with '1-lambda_loss', default 0.5 
    while the classifier's binary crossentropy loss is weighted with lambda_loss, default 0.5
    '''
    set_trainable(autoencoders, True)
    set_trainable(classifier, False)
    losses = {}
    _loss_weights = {}
    _metrics = {}
    losses['concatenated_output'] = 'mse'
    _loss_weights['concatenated_output'] = 1-lambda_loss
    _metrics['concatenated_output'] = ae.snp_accuracy

    losses['pheno_class'] = 'binary_crossentropy'
    _loss_weights['pheno_class'] = lambda_loss
    _metrics['pheno_class'] = [tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                                tf.keras.metrics.Precision(name='precision'),
                                tf.keras.metrics.Recall(name='recall'),
                                tf.keras.metrics.AUC(name='auc'),
                                tf.keras.metrics.AUC(name='prc', curve='PR')
                                ]

    PhenoEncoder.compile(optimizer=Adam(learning_rate=0.0001), 
                         loss=losses, 
                         loss_weights=_loss_weights,
                         metrics=_metrics)
    
def pretrain_ae(data, labels, _epochs=25, BatchSize=32, validation_data=None):
    compile_autoencoders()
    
    if validation_data is not None:
        history = autoencoders.fit(data, labels, epochs=_epochs, batch_size=BatchSize, validation_data=validation_data, verbose=verbosity)
    else:
        history = autoencoders.fit(data, labels, epochs=_epochs, batch_size=BatchSize, verbose=verbosity)

    return history


def pretrain_classifier(data, labels, _epochs=25, BatchSize=32, validation_data=None):
    compile_classifier()
    
    if validation_data is not None:
        history = classifier.fit(data, labels, epochs=_epochs, batch_size=BatchSize, validation_data=validation_data, verbose=verbosity)
    else:
        history = classifier.fit(data, labels, epochs=_epochs, batch_size=BatchSize, verbose=verbosity)

    return history

num_folds=3
# Split your data into K stratified folds to keep the balance of labels
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

# Dictionary to store all cross validation histories for each lambda value
all_lambda_histories = {lambda_loss: [] for lambda_loss in lambda_loss_values}

# Dictionary to store average evaluation results for each lambda_loss value
train_average_eval_results = {lambda_loss: [] for lambda_loss in lambda_loss_values}
val_average_eval_results = {lambda_loss: [] for lambda_loss in lambda_loss_values}

# Iterate over the values of lambda_loss
for lambda_loss in lambda_loss_values:
    print(f"\nTraining with lambda_loss={lambda_loss}")
    
    # A list to store training & validation histories during both pretraining and PhenoEncoder training for all CV folds.
    all_folds_histories = []
    
    # Lists to store evaluation results for each cross validation fold.
    train_eval_results_per_fold = []
    val_eval_results_per_fold = []

    # Perform K-fold cross-validation
    fold_number = 1
    for train_index, val_index in skf.split(train_raw, train_pheno):
        print(f"\nFold {fold_number}/{num_folds}")
        train_raw_, val_raw_ = train_raw.iloc[train_index], train_raw.iloc[val_index]
        train_pheno_, val_pheno_ = train_pheno.iloc[train_index], train_pheno.iloc[val_index]
        
        # These dictionaries will contain subsets of train_raw based on each line in the file.
        # Each recoded haplotype block data can be accessed using keys like 'block1', 'block2', etc.
        # The dictionaries will be renewed for each CV fold.
        X_train = {} 
        X_val = {} 

        # Read truth SNPs from the blocks file
        with open('../data/cnf-0{}-run-{}.onlyTruth.blocks'.format(cnf, run), 'r') as file:
            for line_idx, line in enumerate(file, 1):
                rs_ids = line.strip().split()
                # Subset the train_raw_ DataFrame based on the SNP ids in the haplotype block
                block_train = train_raw_[rs_ids[1:]]
                if not block_train.empty:
                    X_train['block{}'.format(line_idx)] = block_train
                    # Subset the val_raw_ DataFrame based on the SNP ids in the haplotype block
                    block_val = val_raw_[rs_ids[1:]]
                    X_val['block{}'.format(line_idx)] = block_val 
                else:
                    print(f"Warning: No matching columns found for SNPs in haploblock {line_idx}.")

        no_of_blocks = len(X_train)
        print("The recoded data loaded. There are {} blocks.".format(no_of_blocks))

        # The models shold be built from scratch for each CV fold.
        autoencoders, encoders, classifier, PhenoEncoder = model_builder(X_train)

        input_data = {'block{}_input'.format(i+1): X_train['block{}'.format(i+1)] for i in range(no_of_blocks)}
        train_blocks = [X_train[f'block{i+1}'] for i in range(no_of_blocks)]
        output_data = np.concatenate(train_blocks, axis=1)
        classifier_outputs = {'pheno_class': train_pheno_}
        
        # Evaluate the trained model on the validation fold:
        val_inputs = {'block{}_input'.format(i+1): X_val['block{}'.format(i+1)] for i in range(no_of_blocks)}
        val_blocks = [X_val[f'block{i+1}'] for i in range(no_of_blocks)]
        val_outputs = np.concatenate(val_blocks, axis=1)
        val_classifier_outputs = {'pheno_class': val_pheno_}
        
        # Pretrain autoencoders:
        pretrain_epochs = 10
        BatchSize = 32
       
        ae_pretrain_history = pretrain_ae(data=input_data, 
                                          labels=output_data, 
                                          _epochs=pretrain_epochs, 
                                          BatchSize=BatchSize, 
                                          validation_data=(val_inputs, val_outputs))
        print("Autoencoder pretraining completed.")

        # Extract the bottleneck outputs from pretrained PhenoEncoder
        bottleneck_data = encoders.predict(input_data) # For training samples in this fold.
        val_bottleneck_data = encoders.predict(val_inputs) # An for validation samples in this fold.
        
        # Pretrain classifier:
        pretrain_epochs = 5
        BatchSize = 32
        classifier_pretrain_history = pretrain_classifier(data=bottleneck_data, 
                                                          labels=classifier_outputs, 
                                                          _epochs=pretrain_epochs, 
                                                          BatchSize=BatchSize, 
                                                          validation_data=(val_bottleneck_data, val_classifier_outputs))
        print("Classifier pretraining completed.")

        # Initialize a new list to store training histories for each CV fold
        fold_histories = [ae_pretrain_history.history, classifier_pretrain_history.history]

        '''
        Once the autoencoders and the classifier is pretrained respectively, since the classifier requires the raw data to be encoded in the bottlenecks,
        the joint training -simultaneous training of the autoencoders and the classifier- follows. The autoencoders and the classifier are now compiled as part of a bigger "PhenoEncoder" model.
        '''
        outputs_combined = {'concatenated_output': output_data, 'pheno_class': train_pheno_}
        val_outputs_combined = {'concatenated_output': val_outputs, 'pheno_class': val_pheno_}

        # The PhenoEncoder is compiled using the currently tested lambda value.
        compile_phenoencoder(lambda_loss=lambda_loss) 

        # Continue alternating training
        joint_epochs = 25*2 # PhenoEncoder compression takes place every other epoch, while classifier is trained half the time.
        BatchSize = 32
        '''
        If alternating training is turned on below (training_mode='alternating'), first, the classifier weights are frozen while the autoencoder weights are updated through the joint loss calculation for 1 epoch.
        Next epoch, the autoencoder weights are frozen while the classifier weights are updated with the classifier's loss for one epoch.
        The training goes on in an alternating manner for the number of epochs specificed. 
        Else (training_mode='simultaneous'), all entities of the PhenoEncoder model are trained simultaneously.
        '''
        training_mode='alternating'
        # training_mode='simultaneous'
        callbacks_ = [
            EarlyStopping(monitor='val_pheno_class_loss', 
                          patience=10, 
                          mode='min', 
                          restore_best_weights=True, 
                          start_from_epoch=25),
        #    ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min')
        ]
        if training_mode=='alternating':
            for epoch in range(joint_epochs):
                if epoch % 2 == 0:
                    # Train autoencoders with joint mse + binary crossentropy loss while the classifier weights are frozen 
                    history = PhenoEncoder.fit(input_data, 
                                               outputs_combined, 
                                               epochs=1, 
                                               batch_size=BatchSize,
                                               validation_data=(val_inputs, val_outputs_combined),
                                               verbose=verbosity)
                else:
                    # Train classifier while freezing the autoencoders
                    # Extract the bottleneck outputs from current encoders
                    bottleneck_data = encoders.predict(input_data) # For training samples in this fold.
                    val_bottleneck_data = encoders.predict(val_inputs) # An for validation samples in this fold.
                    history = classifier.fit(bottleneck_data, 
                                             classifier_outputs, 
                                             epochs=1, 
                                             batch_size=BatchSize, 
                                             validation_data=(val_bottleneck_data, val_classifier_outputs),
                                             verbose=verbosity)
                fold_histories.append(history.history)
        else:
            history = PhenoEncoder.fit(input_data, 
                                       outputs_combined, 
                                       epochs=joint_epochs, 
                                       batch_size=BatchSize,
                                       validation_data=(val_inputs, val_outputs_combined),
                                       callbacks=callbacks_,
                                       verbose=verbosity)
            fold_histories.append(history.history)

        # Evaluate the model on the train set
        train_evaluation_result = PhenoEncoder.evaluate(input_data, outputs_combined, verbose=verbosity)

        # Evaluate the model on the validation set
        val_evaluation_result = PhenoEncoder.evaluate(val_inputs, val_outputs_combined, verbose=verbosity)

        # Print or store the evaluation result
        print(f"Evaluation result for lambda_loss={lambda_loss}: {val_evaluation_result}")

        # Store the histories logged for the current fold
        all_folds_histories.append(fold_histories)

        # Store the evaluation result for the current fold
        train_eval_results_per_fold.append(train_evaluation_result)
        val_eval_results_per_fold.append(val_evaluation_result)

        fold_number += 1
        metric_names = PhenoEncoder.metrics_names # This is the list of metric names from the model's compiled metrics
        del autoencoders
        del encoders
        del classifier
        del PhenoEncoder
    
    # Store all_fold_histories for the current lambda value in the dictionary
    all_lambda_histories[lambda_loss] = all_folds_histories
    
    # Calculate and store the average evaluation result across all folds for the current lambda_loss value
    train_average_eval_result = np.mean(train_eval_results_per_fold, axis=0)
    train_average_eval_results[lambda_loss] = train_average_eval_result
    val_average_eval_result = np.mean(val_eval_results_per_fold, axis=0)
    val_average_eval_results[lambda_loss] = val_average_eval_result

# We log the overall results for the classifier performance after cross validation
for i, (lambda_loss, average_eval_result) in enumerate(train_average_eval_results.items()):
    # The average performance for each classifier metric is printed:
    print(f"\nAverage train performance for lambda_loss={lambda_loss}:")
    for metric_name, average_value in zip(metric_names[-5:], average_eval_result[-5:]):
        print(f"{metric_name}: {average_value}")
    
for i, (lambda_loss, average_eval_result) in enumerate(val_average_eval_results.items()):
    # The average performance for each classifier metric is printed:
    print(f"\nAverage validation performance for lambda_loss={lambda_loss}:")
    for metric_name, average_value in zip(metric_names[-5:], average_eval_result[-5:]):
        print(f"{metric_name}: {average_value}")

# Save the dictionary to a pickle file
output_file_path = '../results/all_lambda_histories1.pkl'
with open(output_file_path, 'wb') as file:
    pickle.dump(all_lambda_histories, file)

output_file_path = '../results/train_average_skfcv_results.pkl'
with open(output_file_path, 'wb') as file:
    pickle.dump(train_average_eval_results, file)

output_file_path = '../results/val_average_skfcv_results.pkl'
with open(output_file_path, 'wb') as file:
    pickle.dump(val_average_eval_results, file)

print(f"Dictionaries saved")