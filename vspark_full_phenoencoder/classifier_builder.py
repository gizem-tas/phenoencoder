import tensorflow as tf
from tensorflow.keras import layers

import os
# Silencing tensorflow info messages (warnings and errors can still be logged)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
      tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

def build_mlp(numLayers, hiddenLayerDims, learnRate, inputDims=750, Lambda=None, kernelReg=None):
    # Define the input shape
    input_shape = (inputDims,)
    # Create a sequential model
    model = tf.keras.Sequential()
    # Add an input layer
    model.add(layers.Input(shape=input_shape))
	# Add as many hidden layers as specified
    for j in range(numLayers):
            if kernelReg == 'l1':
                model.add(layers.Dense(hiddenLayerDims[j], activation="relu", kernel_regularizer=tf.keras.regularizers.l1(Lambda)))            
            if kernelReg == 'l2':
                model.add(layers.Dense(hiddenLayerDims[j], activation="relu", kernel_regularizer=tf.keras.regularizers.l2(Lambda)))
            else:
                model.add(layers.Dense(hiddenLayerDims[j], activation="relu"))
	# add a sigmoid layer on top
    model.add(layers.Dense(1, activation="sigmoid"))
	# compile the model
    model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=learnRate),
		loss="binary_crossentropy",
		metrics=METRICS)
	# return compiled model
    return model