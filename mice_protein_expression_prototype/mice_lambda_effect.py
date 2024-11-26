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
mpl.rcParams['figure.dpi'] = 300
import recodeA
import autoencoder as ae
import pickle
import os
import csv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection  import train_test_split
ae.silence_tensorflow()
from tensorflow.keras.models import Model
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
X_train_, X_test, train_pheno_, test_pheno, composite_label_train_, composite_label_test = train_test_split(
    data,
    encoded_gt,
    composite_label,
    test_size=0.20,
    random_state=42,
    stratify=composite_label
)

X_train, X_val, train_pheno, val_pheno = train_test_split(
    X_train_,
    train_pheno_,
    test_size=0.25,
    random_state=42,
    stratify=composite_label_train_
)

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
def compile_autoencoders(autoencoders, classifier):
    set_trainable(autoencoders, True)
    set_trainable(classifier, False)
    
    losses = {}
    
    for i in range(no_of_blocks):
        losses['block{}_output'.format(i+1)] = 'mse'

    autoencoders.compile(optimizer=Adam(learning_rate=0.001), loss=losses)

# Compile classifier with binary cross entropy loss
def compile_classifier(autoencoders, classifier):
    set_trainable(autoencoders, False)
    set_trainable(classifier, True)

    classifier.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=[
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
      tf.keras.metrics.AUC(name='prc', curve='PR')
    ])

def compile_phenoencoder(PhenoEncoder, autoencoders, classifier, lambda_loss=0.5):
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
    
def pretrain_ae(autoencoders, classifier, data, labels, val_data, val_labels, _epochs=25, BatchSize=32):
    compile_autoencoders(autoencoders, classifier)

    history = autoencoders.fit(data, labels, epochs=_epochs, batch_size=BatchSize, validation_data=(val_data, val_labels), verbose=2)
    return history

def pretrain_classifier(autoencoders, classifier, data, labels, val_data, val_labels, _epochs=25, BatchSize=32):
    compile_classifier(autoencoders, classifier)

    history = classifier.fit(data, labels, epochs=_epochs, batch_size=BatchSize, validation_data=(val_data, val_labels), verbose=2)
    return history

def joint_training(autoencoders, encoders, classifier, PhenoEncoder, input_data, outputs_combined, lambda_val, joint_epochs=100, BatchSize=32):
    compile_phenoencoder(PhenoEncoder, autoencoders, classifier, lambda_loss=lambda_val)
    
    ae_history = []
    classifier_history = []
    
    for epoch in range(joint_epochs):
        if epoch % 2 == 0:
            # Train autoencoders with joint mse + binary crossentropy loss
            history = PhenoEncoder.fit(input_data, outputs_combined, epochs=1, batch_size=BatchSize, validation_data=(X_val, {'block1_output': X_val, 'pheno_class': val_pheno}), verbose=2)
            ae_history.append(history.history)
        else:
            # Train classifier while freezing the autoencoders
            bottleneck_data = encoders.predict(input_data)
            history = classifier.fit(bottleneck_data, train_pheno, epochs=1, batch_size=BatchSize, validation_data=(encoders.predict(X_val), val_pheno), verbose=2)
            classifier_history.append(history.history)
            
    return ae_history, classifier_history

def lambda_runs(lambda_values, joint_epochs=100, pretrain_epochs=25, BatchSize=32):

    for lambda_val in lambda_values:
        print(f"Training with lambda value: {lambda_val}")
        
        # Reinitialize models for each lambda and assign unique names to the models
        autoencoders, encoders, classifier, PhenoEncoder = model_builder()

        input_data = X_train
        output_data = X_train
        
        # Pretrain autoencoders
        ae_pretrain_history = pretrain_ae(autoencoders, classifier, data=input_data, labels=output_data, val_data=X_val, val_labels=X_val, _epochs=pretrain_epochs, BatchSize=BatchSize)
        
        # Extract the bottleneck outputs from pretrained PhenoEncoder
        bottleneck_data = encoders.predict(input_data)
        val_bottleneck_data = encoders.predict(X_val)
        
        # Pretrain classifier
        classifier_pretrain_history = pretrain_classifier(autoencoders, classifier, data=bottleneck_data, labels=train_pheno, val_data=val_bottleneck_data, val_labels=val_pheno, _epochs=pretrain_epochs, BatchSize=BatchSize)
        
        # Perform joint training
        ae_history, classifier_history = joint_training(autoencoders, encoders, classifier, PhenoEncoder, input_data, {'block1_output': X_train, 'pheno_class': train_pheno}, lambda_val, joint_epochs, BatchSize)
        
        results = {
            'ae_pretrain_history': ae_pretrain_history.history,
            'classifier_pretrain_history': classifier_pretrain_history.history,
            'ae_history': ae_history,
            'classifier_history': classifier_history
        }
        
        print(f"Training with lambda: {lambda_val} completed")

        encoders.save('../results/PE_lambda{}'.format(lambda_val))
        print("Encoders saved")
        
        # Save the dictionary to a pickle file
        output_file_path = '../results/lambda{}_val_histories.pkl'.format(lambda_val)
        with open(output_file_path, 'wb') as file:
            pickle.dump(results, file)
        print(f"Dictionary saved to {output_file_path}")

        del autoencoders
        del encoders
        del classifier
        del PhenoEncoder
        results.clear()

lambda_runs(lambda_values=np.round(np.arange(0.1, 1.0, 0.1), 1), joint_epochs=100, pretrain_epochs=10)