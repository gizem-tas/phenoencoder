#!/usr/bin/env python
# coding: utf-8

#Import the necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection  import train_test_split
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['figure.dpi'] = 300
import umap

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

# Uncomment below if you have trained the autoencoders already.
encoders = load_model('../results/PE')

import umap
'''PHENOTYPE VISUALIZATION'''
# Assuming 'bottleneck_representation' contains the output of encoders.predict()
# Assuming 'phenotypes' contains the corresponding phenotypes (1 or 0)
compressed_train = encoders.predict(X_train)
# Apply UMAP for dimensionality reduction to 2 dimensions
umap_model = umap.UMAP(n_components=2)
transformed_data = umap_model.fit_transform(compressed_train)

# Define the labels for 0 and 1
labels_dict = {0: 'Control', 1: 'Trisomic'}

# Visualize the transformed data
scatter = plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=train_pheno, cmap='viridis')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.title('UMAP Visualization of PhenoEncoder Embeddings of Train Samples')
handles, _ = scatter.legend_elements()
plt.legend(handles, [labels_dict[label] for label in range(len(labels_dict))], title='Phenotype')
plt.show()

compressed_test = encoders.predict(X_test)
# Apply UMAP for dimensionality reduction to 2 dimensions
umap_model = umap.UMAP(n_components=2)
transformed_data = umap_model.fit_transform(compressed_test)

# Visualize the transformed data
scatter = plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=test_pheno, cmap='viridis')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.title('UMAP Visualization of PhenoEncoder Embeddings of Test Samples')
handles, _ = scatter.legend_elements()
plt.legend(handles, [labels_dict[label] for label in range(len(labels_dict))], title='Phenotype')
plt.show()

'''TREATMENT VISUALIZATION'''
compressed_train = encoders.predict(X_train)
# Apply UMAP for dimensionality reduction to 2 dimensions
umap_model = umap.UMAP(n_components=2)
transformed_data = umap_model.fit_transform(compressed_train)

# Define the labels
labels_dict = {1: 'Saline', 0: 'Memantine'}

# Visualize the transformed data
scatter = plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=train_treatment, cmap='spring')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.title('UMAP Visualization of PhenoEncoder Embeddings of Train Samples')
handles, _ = scatter.legend_elements()
plt.legend(handles, [labels_dict[label] for label in range(len(labels_dict))], title='Treatment')
plt.show()

compressed_test = encoders.predict(X_test)
# Apply UMAP for dimensionality reduction to 2 dimensions
umap_model = umap.UMAP(n_components=2)
transformed_data = umap_model.fit_transform(compressed_test)

# Visualize the transformed data
scatter = plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=test_treatment, cmap='spring')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.title('UMAP Visualization of PhenoEncoder Embeddings of Test Samples')
handles, _ = scatter.legend_elements()
plt.legend(handles, [labels_dict[label] for label in range(len(labels_dict))], title='Treatment')
plt.show()

'''BEHAVIOR VISUALIZATION'''
compressed_train = encoders.predict(X_train)
# Apply UMAP for dimensionality reduction to 2 dimensions
umap_model = umap.UMAP(n_components=2)
transformed_data = umap_model.fit_transform(compressed_train)

# Define the labels
labels_dict = {1: 'shock-context', 0: 'context-shock'}

# Visualize the transformed data
scatter = plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=train_behavior, cmap='autumn')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.title('UMAP Visualization of PhenoEncoder Embeddings of Train Samples')
handles, _ = scatter.legend_elements()
plt.legend(handles, [labels_dict[label] for label in range(len(labels_dict))], title='Behavior')
plt.show()

compressed_test = encoders.predict(X_test)
# Apply UMAP for dimensionality reduction to 2 dimensions
umap_model = umap.UMAP(n_components=2)
transformed_data = umap_model.fit_transform(compressed_test)

# Visualize the transformed data
scatter = plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=test_behavior, cmap='autumn')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.title('UMAP Visualization of PhenoEncoder Embeddings of Test Samples')
handles, _ = scatter.legend_elements()
plt.legend(handles, [labels_dict[label] for label in range(len(labels_dict))], title='Behavior')
plt.show()

'''ENTIRE DATASET'''
'''BEHAVIOR VISUALIZATION'''
compressed = encoders.predict(data)
# Apply UMAP for dimensionality reduction to 2 dimensions
umap_model = umap.UMAP(n_components=2)
transformed_data = umap_model.fit_transform(compressed)

# Define the labels
labels_dict = {1: 'shock-context', 0: 'context-shock'}

# Visualize the transformed data
scatter = plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=encoded_behavior, cmap='bwr')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
handles, _ = scatter.legend_elements()
plt.legend(handles, [labels_dict[label] for label in range(len(labels_dict))], title='Behavior')
plt.show()

'''PHENOTYPE VISUALIZATION'''
# Assuming 'bottleneck_representation' contains the output of encoders.predict()
# Assuming 'phenotypes' contains the corresponding phenotypes (1 or 0)
compressed = encoders.predict(data)
# Apply UMAP for dimensionality reduction to 2 dimensions
umap_model = umap.UMAP(n_components=2)
transformed_data = umap_model.fit_transform(compressed)

# Define the labels for 0 and 1
labels_dict = {0: 'Control', 1: 'Trisomic'}

# Visualize the transformed data
scatter = plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=encoded_gt, cmap='viridis')
plt.xlabel('UMAP Dimension 1')
handles, _ = scatter.legend_elements()
plt.legend(handles, [labels_dict[label] for label in range(len(labels_dict))], title='Phenotype')
plt.show()


mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['font.size'] = 24
mpl.rcParams['figure.dpi'] = 300
# Create a figure with two subplots side by side
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

'''BEHAVIOR VISUALIZATION'''
compressed_behavior = encoders.predict(data, verbose=0)
# Apply UMAP for dimensionality reduction to 2 dimensions
umap_model_behavior = umap.UMAP(n_components=2)
transformed_behavior = umap_model_behavior.fit_transform(compressed_behavior)

# Define the labels
labels_dict_behavior = {1: 'shock-context', 0: 'context-shock'}

# Visualize the transformed data for behavior
scatter_behavior = axs[0].scatter(transformed_behavior[:, 0], transformed_behavior[:, 1], c=encoded_behavior, cmap='bwr')
axs[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
handles_behavior, _ = scatter_behavior.legend_elements()
axs[0].legend(handles_behavior, [labels_dict_behavior[label] for label in range(len(labels_dict_behavior))], title='Behavior', markerscale=2)
axs[0].set_title('UMAP')

'''PHENOTYPE VISUALIZATION'''
compressed_pheno = encoders.predict(data, verbose=0)
# Apply UMAP for dimensionality reduction to 2 dimensions
umap_model_pheno = umap.UMAP(n_components=2)
transformed_pheno = umap_model_pheno.fit_transform(compressed_pheno)

# Define the labels for 0 and 1
labels_dict_pheno = {0: 'Control', 1: 'Trisomic'}

# Visualize the transformed data for phenotype
scatter_pheno = axs[1].scatter(transformed_pheno[:, 0], transformed_pheno[:, 1], c=encoded_gt, cmap='viridis')
axs[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)  # Hide ticks and labels
handles_pheno, _ = scatter_pheno.legend_elements()
axs[1].legend(handles_pheno, [labels_dict_pheno[label] for label in range(len(labels_dict_pheno))], title='Phenotype', markerscale=2)
axs[1].set_title('UMAP')

# Show the figure with both subplots
plt.tight_layout()
plt.show()