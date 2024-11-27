#!/usr/bin/env python
# coding: utf-8

#Import the necessary libraries
import numpy as np
import pandas as pd
from math import ceil
import recodeA
import autoencoder as ae
import pickle
import os
import csv
ae.silence_tensorflow()
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Dense, LeakyReLU, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

cnf = 8
run = 3
print("Tensorflow version is:", tf.__version__)
verbosity = 2 #one line per epoch verbosity mode is recommended when not running interactively

lambda_loss = 0.3
# Uncomment below if running on a GPU node.

# Avoiding tensorflow to allocate all the GPU memory
#from tensorflow.compat.v1.keras import backend as K
print(f"Number of GPU: {len(tf.config.experimental.list_physical_devices('GPU'))}")
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
#K.set_session(sess)
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

X_train = {} 
X_test = {} 
# These dicts will contain subsets of train_raw based on each line in the file
# Each recoded haplotype block data can be accessed using keys like 'block1', 'block2', etc.

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

# Read truth SNPs from the blocks file
with open('../data/cnf-0{}-run-{}.onlyTruth.blocks'.format(cnf, run), 'r') as file:
    for line_idx, line in enumerate(file, 1):
        rs_ids = line.strip().split()
        # Subset the train_raw DataFrame based on the SNP ids in the haplotype block
        block_train = train_raw[rs_ids[1:]]
        if not block_train.empty:
            X_train['block{}'.format(line_idx)] = block_train
            # Subset the test_raw DataFrame based on the SNP ids in the haplotype block
            block_test = test_raw[rs_ids[1:]]
            X_test['block{}'.format(line_idx)] = block_test  
        else:
            print(f"Warning: No matching columns found for SNPs in haploblock {line_idx}.")

#Reading the simulated PEPS Phenotypes:
pheno = pd.read_csv("../data/PEPS-Phenotypes/cnf-0{}-run-{}.pheno.csv".format(cnf,run), header=0, index_col='sample')
train_pheno = pheno.loc[train_indices]
test_pheno = pheno.loc[test_indices]

no_of_blocks = len(X_train)
print("The recoded data loaded. There are {} blocks.".format(no_of_blocks))

max_bn = 3 # Dimensions of the autoencoder bottlenecks
print("Each block will be encoded into at most {} dimensions.".format(max_bn))

# Custom activation function for the autoencoder ouput layers, tanh+1, to map the output to 0, 1 or 2, is defined here.
get_custom_objects().update({'additive': Activation(ae.additive)})        

def model_builder():
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

def pretrain_ae(data, labels, _epochs=25, BatchSize=32, validation_data=None):
    compile_autoencoders()
    
    if validation_data is not None:
        history = autoencoders.fit(data, labels, epochs=_epochs, batch_size=BatchSize, validation_data=validation_data, verbose=verbosity)
    else:
        history = autoencoders.fit(data, labels, epochs=_epochs, batch_size=BatchSize, verbose=verbosity)

    return history

autoencoders, encoders, classifier, PhenoEncoder = model_builder()

pretrain_epochs = 10+25 # total_epochs = pretrain_epochs + joint_epochs 
# The total training epochs for compression corresponds to the sum of pre and joint training epochs 
# --> 10+25=35 epochs.
BatchSize = 32
input_data = {'block{}_input'.format(i+1): X_train['block{}'.format(i+1)] for i in range(no_of_blocks)}
train_blocks = [X_train[f'block{i+1}'] for i in range(no_of_blocks)]
output_data = np.concatenate(train_blocks, axis=1)

# Pretrain autoencoders for 10 epochs
ae_pretrain_history = pretrain_ae(data=input_data, 
                                  labels=output_data, 
                                  _epochs=pretrain_epochs, 
                                  BatchSize=BatchSize)
print("Autoencoder training completed.")

test_inputs = {'block{}_input'.format(i+1): X_test['block{}'.format(i+1)] for i in range(no_of_blocks)}
test_blocks = [X_test[f'block{i+1}'] for i in range(no_of_blocks)]
test_outputs = np.concatenate(test_blocks, axis=1)
scores = autoencoders.evaluate(test_inputs, test_outputs)

print(autoencoders.metrics_names)
print(scores)

encoders.save('../results/unsupervised_ae')

# encoders = load_model('../results/unsupervised_ae')