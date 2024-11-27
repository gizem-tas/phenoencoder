#!/usr/bin/env python
# coding: utf-8

# Import the necessary libraries
import warnings
warnings.filterwarnings('ignore')
import random
import numpy as np
import tensorflow as tf
import pandas as pd
from math import ceil
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
from sklearn.cluster import KMeans, SpectralClustering
ae.silence_tensorflow()
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Concatenate, LeakyReLU, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC
from tensorflow.keras import backend as K

cnf = 8
run = 3
print("Tensorflow version is:", tf.__version__)
verbosity = 2 #one line per epoch verbosity mode is recommended when not running interactively

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
data_ = pd.concat([train_raw, test_raw], axis=0)

X_train = {} 
X_test = {} 
block_data = {}
X_train_scaled = {} 
X_test_scaled = {}
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

data_.columns = data_.columns.map(remove_after_)
train_raw.columns = train_raw.columns.map(remove_after_)
test_raw.columns = test_raw.columns.map(remove_after_)

scaler = StandardScaler() # to normalize the data

# Read truth SNPs from the blocks file
with open('../data/cnf-0{}-run-{}.onlyTruth.blocks'.format(cnf, run), 'r') as file:
    for line_idx, line in enumerate(file, 1):
        rs_ids = line.strip().split()
        # Subset the train_raw DataFrame based on the SNP ids in the haplotype block
        block_train = train_raw[rs_ids[1:]]
        if not block_train.empty:
            block_data['block{}_input'.format(line_idx)] = data_[rs_ids[1:]]
            X_train['block{}_input'.format(line_idx)] = block_train
            X_train_scaled['block{}_input'.format(line_idx)] = scaler.fit_transform(X_train['block{}_input'.format(line_idx)])
            # Subset the test_raw DataFrame based on the SNP ids in the haplotype block
            block_test = test_raw[rs_ids[1:]]
            X_test['block{}_input'.format(line_idx)] = block_test  
            X_test_scaled['block{}_input'.format(line_idx)] = scaler.fit_transform(X_test['block{}_input'.format(line_idx)])
        else:
            print(f"Warning: No matching columns found for SNPs in haploblock {line_idx}.")

#Reading the simulated PEPS Phenotypes:
pheno = pd.read_csv("../data/PEPS-Phenotypes/cnf-0{}-run-{}.pheno.csv".format(cnf,run), header=0, index_col='sample')
train_pheno = pheno.loc[train_indices]
test_pheno = pheno.loc[test_indices]
print("Train phenotype value counts:", train_pheno.value_counts())
print("Test phenotype value counts:", test_pheno.value_counts())
pheno_ = pd.concat([train_pheno, test_pheno], axis=0)
pheno_ = pheno_.values.ravel()

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
        block_size = X_train['block{}_input'.format(i+1)].shape[1] # Number of SNPs in block i
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

# Function to run spectral clustering
def evaluate_spectral_clustering(data, true_labels, n_clusters=2, n_runs=10):
    rand_scores = []
    adjusted_rand_scores = []
    
    for _ in range(n_runs):
        # Initialize and fit the Spectral Clustering model
        spectral = SpectralClustering(n_clusters=n_clusters, affinity='rbf', random_state=None)
        predicted_labels = spectral.fit_predict(data)
        
        # Compute the Rand Index and Adjusted Rand Index
        rand_scores.append(rand_score(true_labels, predicted_labels))
        adjusted_rand_scores.append(adjusted_rand_score(true_labels, predicted_labels))
    
    # Return the mean and standard deviation of the Rand Index and Adjusted Rand Index
    return np.mean(rand_scores), np.std(rand_scores), np.mean(adjusted_rand_scores), np.std(adjusted_rand_scores)


encoders_AE = load_model('../results/unsupervised_ae')
encoders_PE = load_model('../results/PhenoEncoder_lambda0.3')
encoders_PE_ = load_model('../results/PhenoEncoder_lambda0.7')

_epochs = 50
BatchSize = 32
########################### UNSUPERVISED COMPRESSION ############################

# Autoencoder bottleneck data
bottleneck_data_AE = np.hstack(encoders_AE.predict(X_train_scaled, verbose=0))
test_bottleneck_data_AE = np.hstack(encoders_AE.predict(X_test_scaled, verbose=0))

# Logistic Regression on AE data
log_reg_AE = LogisticRegression()
ae_lr_train_metrics, ae_lr_test_metrics = evaluate_classification(log_reg_AE, bottleneck_data_AE, test_bottleneck_data_AE, train_pheno, test_pheno)

# MLP on AE data (expects multiple bottleneck inputs, therefore np.hstack is not used)
mlp_AE = model_builder()
ae_mlp_train_metrics, ae_mlp_test_metrics = evaluate_mlp(mlp_AE, encoders_AE.predict(X_train_scaled, verbose=0), encoders_AE.predict(X_test_scaled, verbose=0), train_pheno, test_pheno)

# K-means clustering on AE data
#rand_score_AE, rand_std_AE, adj_rand_score_AE, adj_rand_std_AE = evaluate_clustering(np.hstack(encoders_AE.predict(block_data, verbose=0)), pheno_)

# Spectral clustering on AE data
#rand_score_AE_spectral, rand_std_AE_spectral, adj_rand_score_AE_spectral, adj_rand_std_AE_spectral = evaluate_spectral_clustering(np.hstack(encoders_AE.predict(block_data)), pheno_)

##################### SUPERVISED COMPRESSION --LAMBDA 0.3-- #####################

# Phenoencoder bottleneck data
bottleneck_data_PE = np.hstack(encoders_PE.predict(X_train_scaled, verbose=0))
test_bottleneck_data_PE = np.hstack(encoders_PE.predict(X_test_scaled, verbose=0))

# Logistic Regression on PE data
log_reg_PE = LogisticRegression()
pe_lr_train_metrics, pe_lr_test_metrics = evaluate_classification(log_reg_PE, bottleneck_data_PE, test_bottleneck_data_PE, train_pheno, test_pheno)

# MLP on PE data (expects multiple bottleneck inputs, therefore np.hstack is not used)
mlp_PE = model_builder()
pe_mlp_train_metrics, pe_mlp_test_metrics = evaluate_mlp(mlp_PE, encoders_PE.predict(X_train_scaled, verbose=0), encoders_PE.predict(X_test_scaled, verbose=0), train_pheno, test_pheno)

# K-means clustering on PE data
#rand_score_PE, rand_std_PE, adj_rand_score_PE, adj_rand_std_PE = evaluate_clustering(np.hstack(encoders_PE.predict(block_data, verbose=0)), pheno_)

# Spectral clustering on PE data
#rand_score_PE_spectral, rand_std_PE_spectral, adj_rand_score_PE_spectral, adj_rand_std_PE_spectral = evaluate_spectral_clustering(np.hstack(encoders_PE.predict(block_data)), pheno_)

##################### SUPERVISED COMPRESSION --LAMBDA 0.7-- #####################

# Phenoencoder bottleneck data
bottleneck_data_PE_ = np.hstack(encoders_PE_.predict(X_train_scaled, verbose=0))
test_bottleneck_data_PE_ = np.hstack(encoders_PE_.predict(X_test_scaled, verbose=0))

# Logistic Regression on PE data
log_reg_PE_ = LogisticRegression()
pe_lr_train_metrics_, pe_lr_test_metrics_ = evaluate_classification(log_reg_PE_, bottleneck_data_PE_, test_bottleneck_data_PE_, train_pheno, test_pheno)

# MLP on PE data (expects multiple bottleneck inputs, therefore np.hstack is not used)
mlp_PE_ = model_builder()
pe_mlp_train_metrics_, pe_mlp_test_metrics_ = evaluate_mlp(mlp_PE_, encoders_PE_.predict(X_train_scaled, verbose=0), encoders_PE_.predict(X_test_scaled, verbose=0), train_pheno, test_pheno)

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
#print_classification_metrics("Autoencoder - Logistic Regression", ae_lr_train_metrics, ae_lr_test_metrics)
print_classification_metrics("Autoencoder - MLP", ae_mlp_train_metrics, ae_mlp_test_metrics)
#print_clustering_metrics("Autoencoder - K-means Clustering", rand_score_AE, rand_std_AE, adj_rand_score_AE, adj_rand_std_AE)
#print_clustering_metrics("Autoencoder - Spectral Clustering", rand_score_AE_spectral, rand_std_AE_spectral, adj_rand_score_AE_spectral, adj_rand_std_AE_spectral)

# Print metrics
#print_classification_metrics("Phenoencoder (Lambda = 0.3) - Logistic Regression", pe_lr_train_metrics, pe_lr_test_metrics)
print_classification_metrics("Phenoencoder (Lambda = 0.3) - MLP", pe_mlp_train_metrics, pe_mlp_test_metrics)
#print_clustering_metrics("Phenoencoder (Lambda = 0.3) - K-means Clustering", rand_score_PE, rand_std_PE, adj_rand_score_PE, adj_rand_std_PE)
#print_clustering_metrics("Phenoencoder - Spectral Clustering", rand_score_PE_spectral, rand_std_PE_spectral, adj_rand_score_PE_spectral, adj_rand_std_PE_spectral)

#print_classification_metrics("Phenoencoder (Lambda = 0.7) - Logistic Regression", pe_lr_train_metrics_, pe_lr_test_metrics_)
print_classification_metrics("Phenoencoder (Lambda = 0.7) - MLP", pe_mlp_train_metrics_, pe_mlp_test_metrics_)