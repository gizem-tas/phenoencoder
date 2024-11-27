# PhenoEncoder
This directory includes the python scripts as part of the PhenoEncoder project, using datasets from UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/Mice+Protein+Expression) which contains continuous protein expression levels of trisomic and healthy mice, and VariantSpark (http://gigadb.org/dataset/100759) including genomic variants associated with simulated complex phenotypes, in their respective directories.

<p align="center">
  <img width=50% height=50% src="https://github.com/user-attachments/assets/2e7ad93b-20be-4006-80aa-bd19f87268c1">
</p>

## The Prototoype Model
For the Mice Protein Expression levels, the input data dimensionality is 77, which is computationally feasible to process through a single autoencoder. Therefore, we do not perform window-based preprocessing of the input data for these set of experiments, the PhenoEncoder model for this case consists of a single autoencoder and the auxiliary classifier.

<p align="center">
  <img width=50% height=50% src="https://github.com/user-attachments/assets/ea396e65-47b6-4b06-8be0-e90b4b223e7c">
</p>

## The Full PhenoEncoder
The VariantSpark dataset includes real genotyping data of 2,504 samples from 1000-Genomes Project for which binary phenotype labels are simulated using the Polygenic Epistatic Phenotype Simulator (PEPS) (https://pubmed.ncbi.nlm.nih.gov/38269921/). In our experiments, we used the phenotype simulated with the options cnf=8 and run=3. It is possible to modify these options in the scripts for testing our methodology under different PEPS phenotypes.

### Data Preprocessing
For the multiple autoencoder architecture, SNP datasets need to be pre-processed.
Estimating the haplotype blocks from vcf files:
```bash
for chr_no in {1..22};
  plink --vcf <vcf_file_name> --blocks no-pheno-req --blocks-max-kb 10000 --blocks-min-maf 0.01 --blocks-recomb-highci 0.7 --blocks-strong-highci 0.85 --blocks-strong-lowci 0.5001 --chr ${i} --out <output_file_name>;
done
```
The necesssary order of the preprocessing scripts for extracting truth blocks: 
```bash
1. truth_blocks.py # Find the truth blocks
2. count_truth_SNPs.py # Finds the SNPs lost in between haplotype block boundaries
3. unfound_truth_blocks.py # Finds lost SNPs' neighbor blocks and updates only_truth_blocks file
4. list_snps_in_truth_blocks.py # Writes the SNPs in truth blocks to <SNPsinTruthBlocks.txt> file  
```
To recode the SNPs in truth blocks into allelic dosage values:
```bash
plink --vcf <vcf_file_name> --extract <SNPsinTruthBlocks.txt> --make-bed --out <TruthBlocks>
plink --bfile <TruthBlocks> --recodeA --out <TruthBlocks_recoded>
```
Resulting recoded data file <TruthBlocks_recoded.raw>  has the following format:
```bash
FID  IID  PAT  MAT  SEX  PHENOTYPE snp001  snp002 ...  snpXXX
```
See https://github.com/gizem-tas/haploblock-autoencoders/blob/main/README.md for the module recodeA.py for handling recoded files.
