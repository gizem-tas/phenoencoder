# PhenoEncoder
This directory includes the python scripts as part of the PhenoEncoder project, using datasets from UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/Mice+Protein+Expression) and VariantSpark (http://gigadb.org/dataset/100759) including genomic variants associated with simulated complex phenotypes, in their respective directories.
![phenoencoder_diagram](https://github.com/user-attachments/assets/2e7ad93b-20be-4006-80aa-bd19f87268c1)

## The Prototoype Model
For the Mice Protein Expression levels, the input data dimensionality is 77, which is computationally feasible to process through a single autoencoder. Therefore, we do not perform window-based preprocessing of the input data for these set of experiments, the PhenoEncoder model for this case consists of a single autoencoder and the auxiliary classifier.
![prototoype_model_diagram](https://github.com/user-attachments/assets/ea396e65-47b6-4b06-8be0-e90b4b223e7c)

## The Full PhenoEncoder
The VariantSpark dataset includes real genotyping data of 2,504 samples from 1000-Genomes Project for which binary phenotype labels are simulated using the Polygenic Epistatic Phenotype Simulator (PEPS) (https://pubmed.ncbi.nlm.nih.gov/38269921/). In our experiments, we used the phenotype simulated with the options cnf=8 and run=3. It is possible to modify these options in the scripts for testing our methodology under different PEPS phenotypes.

For the multiple autoencoder architecture, SNP datasets need to be pre-processed. The necesssary order of the corresponding pre-processing scripts for this set of experiments is: 
