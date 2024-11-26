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
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['font.size'] = 20
mpl.rcParams['figure.dpi'] = 300
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

# encode class values as integers
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

# Function to freeze/unfreeze layers in a model
def set_trainable(model, trainable):
    for layer in model.layers:
        layer.trainable = trainable

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

    classifier.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=[
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
      tf.keras.metrics.AUC(name='prc', curve='PR')
    ])

def compile_phenoencoder(lambda_loss=0.5):
    '''
    Compile PhenoEncoder model with multiple losses
    Autoencoder losses are weighted equally with 'lambda_loss', default 0.5 
    while the classifier's mse loss is weighted with 1-lambda_loss, default 0.5
    '''
    set_trainable(autoencoders, True)
    set_trainable(classifier, False)

    pe_losses = {'pheno_class': 'binary_crossentropy'}
    pe_metrics = {'pheno_class': [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
      tf.keras.metrics.AUC(name='prc', curve='PR')
    ]}
    pe_loss_weights = {'pheno_class': lambda_loss}

    for i in range(no_of_blocks):
        block_output_name = 'block{}_output'.format(i+1)
        pe_losses[block_output_name] = 'mse'
        pe_metrics[block_output_name] = 'mse'
        pe_loss_weights[block_output_name] = 1 - lambda_loss

    PhenoEncoder.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=pe_losses,
        loss_weights=pe_loss_weights,
        metrics=pe_metrics
    )
    
def pretrain_ae(data, labels, _epochs=25, BatchSize=32):
    compile_autoencoders()

    history = autoencoders.fit(data, labels, epochs=_epochs, batch_size=BatchSize, verbose=2)
    return history

def pretrain_classifier(data, labels, _epochs=25, BatchSize=32):
    compile_classifier()

    history = classifier.fit(data, labels, epochs=_epochs, batch_size=BatchSize, verbose=2)
    return history

def evaluate_model(PhenoEncoder, X_train, y_train, X_test, y_test):
    # Evaluate on training data
    train_metrics = PhenoEncoder.evaluate(X_train, {'block1_output': X_train, 'pheno_class': y_train}, verbose=0)
    
    # Evaluate on validation data
    test_metrics = PhenoEncoder.evaluate(X_test, {'block1_output': X_test, 'pheno_class': y_test}, verbose=0)
    
    return {
        'train': {
            'total_loss': train_metrics[0],
            'mse_loss': train_metrics[1],            # block1_output_loss corresponds to MSE loss
            'bce_loss': train_metrics[2],            # pheno_class_loss corresponds to BCE loss
            'mse_metric': train_metrics[3],          # block1_output_mse is an additional MSE metric
            'accuracy': train_metrics[4],            # pheno_class_accuracy
            'precision': train_metrics[5],           # pheno_class_precision
            'recall': train_metrics[6],              # pheno_class_recall
            'auc': train_metrics[7],                 # pheno_class_auc
            'prc': train_metrics[8]                  # pheno_class_prc
        },
        'test': {
            'total_loss': test_metrics[0],
            'mse_loss': test_metrics[1],              # block1_output_loss corresponds to MSE loss
            'bce_loss': test_metrics[2],              # pheno_class_loss corresponds to BCE loss
            'mse_metric': test_metrics[3],            # block1_output_mse is an additional MSE metric
            'accuracy': test_metrics[4],              # pheno_class_accuracy
            'precision': test_metrics[5],             # pheno_class_precision
            'recall': test_metrics[6],                # pheno_class_recall
            'auc': test_metrics[7],                   # pheno_class_auc
            'prc': test_metrics[8]                    # pheno_class_prc
        }
    }
    
autoencoders, encoders, classifier, PhenoEncoder = model_builder()

pretrain_epochs = 10
BatchSize = 32
input_data = X_train
output_data = X_train
classifier_outputs = {'pheno_class': train_pheno}

# Pretrain autoencoders for 10 epochs
ae_pretrain_history = pretrain_ae(data=input_data, labels=output_data, _epochs=pretrain_epochs, BatchSize=BatchSize)
print("Autoencoder pretraining completed.")

# Extract the bottleneck outputs from pretrained PhenoEncoder
bottleneck_data = encoders.predict(input_data)

pretrain_epochs = 10
BatchSize = 32

# Pretrain classifier for 10 epochs
classifier_pretrain_history = pretrain_classifier(data=bottleneck_data, labels=classifier_outputs, _epochs=pretrain_epochs, BatchSize=BatchSize)
print("Classifier pretraining completed.")

joint_epochs = 100
BatchSize = 32
outputs_combined = {'block1_output': X_train, 'pheno_class': train_pheno}
compile_phenoencoder(lambda_loss=0.7) 

# Continue alternating training for 100 epochs during which
# the PhenoEncoder is trained for 50, while the classifier for the other 50
for epoch in range(joint_epochs):
    print(f"\nEpoch {epoch + 1}/{joint_epochs}")
    if epoch % 2 == 0:
        # Train autoencoders with joint mse + binary crossentropy loss
        PhenoEncoder.fit(input_data, outputs_combined, epochs=1, batch_size=BatchSize, verbose=2)
    else:
        # Train classifier while freezing the autoencoders
        bottleneck_data = encoders.predict(input_data)
        classifier.fit(bottleneck_data, classifier_outputs, epochs=1, batch_size=BatchSize, verbose=2)

# Evaluate the trained model on the test samples:

evaluate_model(PhenoEncoder, X_train, train_pheno, X_test, test_pheno)

encoders.save('../results/PE')

