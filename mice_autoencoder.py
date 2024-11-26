#!/usr/bin/env python
# coding: utf-8

#Import the necessary libraries
import random
random.seed(42)
import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)
import pandas as pd
import recodeA
import autoencoder as ae
import pickle
import os
import csv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection  import train_test_split
ae.silence_tensorflow()
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, concatenate, LeakyReLU, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

data = pd.read_csv("../data/mice_protein_expression_data.csv")

# Encode class values as integers
label_encoder = LabelEncoder()
genotype = data['Genotype']
encoded_gt = label_encoder.fit_transform(genotype)

treatment = pd.read_csv("../data/treatment.txt", sep=" ")
encoded_treatment = label_encoder.fit_transform(treatment['Treatment'])

behavior = pd.read_csv("../data/behavior.txt", sep=" ")
encoded_behavior = label_encoder.fit_transform(behavior['Behavior'])

data = data.iloc[: , 1:78].astype(float)

# We need to straify the data for better estimating the generalizability of our results.
# First create a composite label that combines the genotype, treatment, and behavior.
composite_label = ['{}_{}_{}'.format(gt, tr, bh) for gt, tr, bh in zip(encoded_gt, encoded_treatment, encoded_behavior)]

# Split the data into a training set and a test set with stratification
X_train, X_test, train_pheno, test_pheno, train_indices, test_indices = train_test_split(
    data,
    encoded_gt,
    range(len(data)),
    test_size=0.20,
    random_state=42,
    stratify=composite_label
)

train_treatment = encoded_treatment[train_indices]
test_treatment = encoded_treatment[test_indices]

train_behavior = encoded_behavior[train_indices]
test_behavior = encoded_behavior[test_indices]

no_of_blocks = 1
print("The recoded data loaded. There are {} blocks.".format(no_of_blocks))

bn = 8 # Dimensions of the autoencoder bottlenecks
print("Each block will be encoded into {} dimensions.".format(bn))

def model_builder():
    K.clear_session()
    inputs=[]
    bottlenecks=[]
    outputs=[]
    
    for i in range(no_of_blocks):  
        block_size = X_train.shape[1] # Number of SNPs the block
        block_input = Input(shape=(block_size,), name='block{}_input'.format(i+1))
        inputs.append(block_input)  

        encoder, decoder = ae.model_shape(inputs=block_size, shape='sqrt', hl=4, bn=bn)

        x = LeakyReLU(alpha=0.01)(Dense(units=encoder[0], kernel_initializer='he_uniform')(block_input))

        for nodes in encoder[1:]:
            x = LeakyReLU(alpha=0.01)(Dense(units=nodes, kernel_initializer='he_uniform')(x))

        x = LeakyReLU(alpha=0.01, name='block{}_bottleneck'.format(i+1))(Dense(units=bn, kernel_initializer='he_uniform')(x)) 
        bottlenecks.append(x)
        
        for nodes in decoder:
            x = LeakyReLU(alpha=0.01)(Dense(units=nodes, kernel_initializer='he_uniform')(x))
    
        block_output = Dense(units=block_size, activation='linear', name='block{}_output'.format(i+1))(x)
        outputs.append(block_output)

    merged_bns = concatenate(bottlenecks)
    case_control = LeakyReLU(alpha=0.01)(Dense(units=16, kernel_initializer='he_uniform')(merged_bns))
    case_control = Dropout(0.2)(case_control)
    case_control = LeakyReLU(alpha=0.01)(Dense(units=4, kernel_initializer='he_uniform')(case_control))
    pheno_class = Dense(1, activation='sigmoid', name='pheno_class')(case_control)
    
    autoencoders = Model(inputs=inputs, outputs=outputs, name='autoencoders')
    encoders = Model(inputs=inputs, outputs=bottlenecks, name='encoders')
    classifier = Model(inputs=bottlenecks, outputs=pheno_class, name='classifier')
    PhenoEncoder = Model(inputs=inputs, outputs=outputs + [pheno_class], name='PhenoEncoder')

    return autoencoders, encoders, classifier, PhenoEncoder

autoencoders, encoders, classifier, PhenoEncoder = model_builder()

# Function to freeze/unfreeze layers in a model
def set_trainable(model, trainable):
    for layer in model.layers:
        layer.trainable = trainable

# Define the performance metrics for autoencoders and classifier
classifier_metrics = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
      tf.keras.metrics.AUC(name='prc', curve='PR')
]

# Compile autoencoders with mse losses
def compile_autoencoders():
    set_trainable(autoencoders, True)
    set_trainable(classifier, False)
    
    losses = {}
    
    for i in range(no_of_blocks):
        losses['block{}_output'.format(i+1)] = 'mse'

    autoencoders.compile(optimizer=Adam(learning_rate=0.001), loss=losses)

# Compile classifier with binary cross entropy loss
def compile_classifier():
    set_trainable(autoencoders, False)
    set_trainable(classifier, True)
    
    losses = {}
    losses['pheno_class'] = 'binary_crossentropy'

    _metrics = {}
    _metrics['pheno_class'] = classifier_metrics

    classifier.compile(optimizer=Adam(learning_rate=0.001), loss=losses, metrics=_metrics)


def compile_phenoencoder(lambda_loss):
    '''
    Compile PhenoEncoder model with multiple losses
    Autoencoder losses are weighted equally with 'lambda_loss', default 0.5 
    while the classifier's mse loss is weighted with 1-lambda_loss, default 0.5
    '''
    set_trainable(autoencoders, True)
    set_trainable(classifier, False)
    
    losses = {}
    _loss_weights = {}
    _metrics = {}
    for i in range(no_of_blocks):
        losses['block{}_output'.format(i+1)] = 'mse'
        _loss_weights['block{}_output'.format(i+1)] = 1
        _metrics['block{}_output'.format(i+1)] = 'mse'
    
    losses['pheno_class'] = 'binary_crossentropy'
    _loss_weights['pheno_class'] = lambda_loss
    _metrics['pheno_class'] = classifier_metrics

    PhenoEncoder.compile(optimizer=Adam(learning_rate=0.001), 
                         loss=losses, 
                         loss_weights=_loss_weights,
                         metrics=_metrics)
    
def pretrain_ae(data, labels, _epochs=25, BatchSize=32):
    compile_autoencoders()

    history = autoencoders.fit(data, labels, epochs=_epochs, batch_size=BatchSize, verbose=2)
    return history

pretrain_epochs = 60
BatchSize = 32
input_data = X_train
output_data = X_train

# Pretrain autoencoders
ae_pretrain_history = pretrain_ae(data=input_data, labels=output_data, _epochs=pretrain_epochs, BatchSize=BatchSize)
print("Autoencoder pretraining completed.")

encoders.save('../results/AE')