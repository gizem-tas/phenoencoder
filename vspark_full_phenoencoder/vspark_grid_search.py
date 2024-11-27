#!/usr/bin/env python
# coding: utf-8

import classifier_builder as cb
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold, GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import layers

import os
# Silencing tensorflow info messages (warnings and errors can still be logged)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

cnf = 8
run = 3

# Read the pre-defined train and test IDs:
train_IDs = pd.read_csv("../data/cnf-0{}-run-{}.train_IDs.txt".format(cnf,run), sep = " ", header=None)
train_indices = train_IDs.iloc[:,0].values.tolist()

test_IDs = pd.read_csv("../data/cnf-0{}-run-{}.test_IDs.txt".format(cnf,run), sep = " ", header=None)
test_indices = test_IDs.iloc[:,0].values.tolist()

#Reading the autoencoder compressed data:
compressed = pd.read_csv("/hpc/hers_en/gtas/projects/AE_joint_loss/VariantSpark/results/cnf-0{}-run-{}_compressed.csv".format(cnf,run), header=0, index_col='IID')

train_comp = compressed.loc[train_indices]
test_comp = compressed.loc[test_indices]

#Reading the simulated PEPS Phenotypes:
pheno = pd.read_csv("../data/PEPS-Phenotypes/cnf-0{}-run-{}.pheno.csv".format(cnf,run), header=0, index_col='sample')
train_pheno = pheno.loc[train_indices]
test_pheno = pheno.loc[test_indices]

print("Data loaded.")  

# wrap our model into a scikit-learn compatible classifier
print("[INFO] initializing model...")
model = KerasClassifier(build_fn=cb.build_mlp, verbose=0)

# define a grid of the hyperparameter search space
grid = dict(
    numLayers = [2, 3, 4],
    hiddenLayerDims = [[512,256,128,64], [256,256,128,128], [256,128,64,32], [128,64,32,16], [128,128,64,64], [64,64,32,32]],
    learnRate = [1e-2, 1e-3, 1e-4],
    batch_size = [16, 32, 64]
)

kf = KFold(5, shuffle=True, random_state=1)
searcher = GridSearchCV(estimator=model, param_grid=grid, cv=kf, scoring='accuracy', n_jobs=-1)
searchResults = searcher.fit(train_comp, train_pheno, epochs=50, verbose=0)

# summarize grid search information
bestScore = searchResults.best_score_
bestParams = searchResults.best_params_
print("[INFO] best score is {:.2f} using {}".format(bestScore, bestParams))

# extract the best model, make predictions on our data, and show a
# classification report
print("[INFO] evaluating the best model...")

def build_best_mlp(bestParams):
    return cb.build_mlp(
        numLayers=bestParams['numLayers'],
        hiddenLayerDims=bestParams['hiddenLayerDims'],
        learnRate=bestParams['learnRate']
    )

bestModel = build_best_mlp(bestParams)
print(bestModel.summary())

bestModel.fit(train_comp,
              train_pheno,
              epochs=50, 
              batch_size=bestParams['batch_size'],
              verbose=2
              )

train_scores = bestModel.evaluate(train_comp, train_pheno)
print("Best MLP train scores for compressed data:")
print(bestModel.metrics_names)
print(train_scores)

test_scores = bestModel.evaluate(test_comp, test_pheno)
print("Best MLP test scores for compressed data:")
print(bestModel.metrics_names)
print(test_scores)