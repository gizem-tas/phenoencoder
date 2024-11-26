#!/usr/bin/env python
# coding: utf-8

# Import the necessary libraries
import warnings
warnings.filterwarnings('ignore')
import random
import numpy as np
import tensorflow as tf
import pandas as pd
import recodeA
import autoencoder as ae
import pickle
import os
import csv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score, classification_report
from sklearn.metrics import adjusted_rand_score, rand_score
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
ae.silence_tensorflow()
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, concatenate, LeakyReLU, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC
from tensorflow.keras import backend as K

verbosity = 2 # one line per epoch verbosity mode is recommended when not running interactively

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

bn = 8 # Dimensions of the autoencoder bottlenecks

def model_builder():
    K.clear_session()
    inputs = []
    bottlenecks = []
    outputs = []
    
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

    return classifier

def evaluate_classification(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    def compute_metrics(y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        prc = average_precision_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        return accuracy, precision, recall, auc, prc, report

    train_metrics = compute_metrics(y_train, y_pred_train)
    test_metrics = compute_metrics(y_test, y_pred_test)
    return train_metrics, test_metrics

def evaluate_mlp(model, X_train, X_test, y_train, y_test):
    # Compile the model with Keras metrics
    model.compile(optimizer=Adam(learning_rate=0.0001), 
                  loss='binary_crossentropy', 
                  metrics=[BinaryAccuracy(name='accuracy'),
                           Precision(name='precision'),
                           Recall(name='recall'),
                           AUC(name='auc'),
                           AUC(name='prc', curve='PR')])  # prc for Precision-Recall Curve AUC

    # Fit the model
    model.fit(X_train, y_train, epochs=_epochs, batch_size=BatchSize, verbose=verbosity)

    # Evaluate on train and test data
    train_metrics = model.evaluate(X_train, y_train, verbose=0)
    test_metrics = model.evaluate(X_test, y_test, verbose=0)

    # Convert metrics to a dictionary for better readability
    metric_names = model.metrics_names
    train_metrics_dict = {metric_names[i]: train_metrics[i] for i in range(len(metric_names))}
    test_metrics_dict = {metric_names[i]: test_metrics[i] for i in range(len(metric_names))}

    return train_metrics_dict, test_metrics_dict

# Function to run k-means clustering
def evaluate_clustering(data, true_labels, n_clusters=2, n_runs=100):
    rand_scores = []
    adjusted_rand_scores = []
    for _ in range(n_runs):
        kmeans = KMeans(n_clusters=n_clusters, random_state=None)
        kmeans.fit(data)
        rand_scores.append(rand_score(true_labels, kmeans.labels_))
        adjusted_rand_scores.append(adjusted_rand_score(true_labels, kmeans.labels_))
    return np.mean(rand_scores), np.std(rand_scores), np.mean(adjusted_rand_scores), np.std(adjusted_rand_scores)

# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

encoders_AE = load_model('../results/AE')
encoders_PE = load_model('../results/PE')

_epochs = 25
BatchSize = 32
########################### UNSUPERVISED COMPRESSION ############################

# Autoencoder bottleneck data
bottleneck_data_AE = encoders_AE.predict(X_train_scaled)
test_bottleneck_data_AE = encoders_AE.predict(X_test_scaled)

# Logistic Regression on AE data
log_reg_AE = LogisticRegression()
ae_lr_train_metrics, ae_lr_test_metrics = evaluate_classification(log_reg_AE, bottleneck_data_AE, test_bottleneck_data_AE, train_pheno, test_pheno)

# MLP on AE data
mlp_AE = model_builder()
ae_mlp_train_metrics, ae_mlp_test_metrics = evaluate_mlp(mlp_AE, bottleneck_data_AE, test_bottleneck_data_AE, train_pheno, test_pheno)

# K-means clustering on AE data
rand_score_AE, rand_std_AE, adj_rand_score_AE, adj_rand_std_AE = evaluate_clustering(encoders_AE.predict(data), encoded_gt)

##################### SUPERVISED COMPRESSION --LAMBDA 0.7-- #####################

# Phenoencoder bottleneck data
bottleneck_data_PE = encoders_PE.predict(X_train_scaled)
test_bottleneck_data_PE = encoders_PE.predict(X_test_scaled)

# Logistic Regression on PE data
log_reg_PE = LogisticRegression()
pe_lr_train_metrics, pe_lr_test_metrics = evaluate_classification(log_reg_PE, bottleneck_data_PE, test_bottleneck_data_PE, train_pheno, test_pheno)

# MLP on PE data
mlp_PE = model_builder()
pe_mlp_train_metrics, pe_mlp_test_metrics = evaluate_mlp(mlp_PE, bottleneck_data_PE, test_bottleneck_data_PE, train_pheno, test_pheno)

# K-means clustering on PE data
rand_score_PE, rand_std_PE, adj_rand_score_PE, adj_rand_std_PE = evaluate_clustering(encoders_PE.predict(data), encoded_gt)

# Function to print classification metrics
def print_classification_metrics(name, train_metrics, test_metrics):
    # Check if metrics are dictionaries (for evaluate_mlp) or tuples (for evaluate_classification)
    if isinstance(train_metrics, dict) and isinstance(test_metrics, dict):
        print(f"{name} - Train: "
              f"Accuracy: {train_metrics['accuracy']:.4f}, "
              f"Precision: {train_metrics['precision']:.4f}, "
              f"Recall: {train_metrics['recall']:.4f}, "
              f"AUC: {train_metrics['auc']:.4f}, "
              f"PRC: {train_metrics['prc']:.4f}")
        print(f"{name} - Test: "
              f"Accuracy: {test_metrics['accuracy']:.4f}, "
              f"Precision: {test_metrics['precision']:.4f}, "
              f"Recall: {test_metrics['recall']:.4f}, "
              f"AUC: {test_metrics['auc']:.4f}, "
              f"PRC: {test_metrics['prc']:.4f}")
    else:
        # Unpack tuple metrics for evaluate_classification
        train_acc, train_prec, train_rec, train_auc, train_prc, train_report = train_metrics
        test_acc, test_prec, test_rec, test_auc, test_prc, test_report = test_metrics
        print(f"{name} - Train: "
              f"Accuracy: {train_acc:.4f}, "
              f"Precision: {train_prec:.4f}, "
              f"Recall: {train_rec:.4f}, "
              f"AUC: {train_auc:.4f}, "
              f"PRC: {train_prc:.4f}")
        print(f"{name} - Test: "
              f"Accuracy: {test_acc:.4f}, "
              f"Precision: {test_prec:.4f}, "
              f"Recall: {test_rec:.4f}, "
              f"AUC: {test_auc:.4f}, "
              f"PRC: {test_prc:.4f}")

# Function to print clustering metrics
def print_clustering_metrics(name, rand_score_mean, rand_score_std, adj_rand_score_mean, adj_rand_score_std):
    print(f"{name}: "
          f"Rand Index Mean: {rand_score_mean:.4f}, STD: {rand_score_std:.4f}, "
          f"Adjusted Rand Index Mean: {adj_rand_score_mean:.4f}, STD: {adj_rand_score_std:.4f}")

# Print metrics
print_classification_metrics("Autoencoder - Logistic Regression", ae_lr_train_metrics, ae_lr_test_metrics)
print_classification_metrics("Autoencoder - MLP", ae_mlp_train_metrics, ae_mlp_test_metrics)
print_clustering_metrics("Autoencoder - K-means Clustering", rand_score_AE, rand_std_AE, adj_rand_score_AE, adj_rand_std_AE)

# Print metrics
print_classification_metrics("Phenoencoder - Logistic Regression", pe_lr_train_metrics, pe_lr_test_metrics)
print_classification_metrics("Phenoencoder - MLP", pe_mlp_train_metrics, pe_mlp_test_metrics)
print_clustering_metrics("Phenoencoder - K-means Clustering", rand_score_PE, rand_std_PE, adj_rand_score_PE, adj_rand_std_PE)
