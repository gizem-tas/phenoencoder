#!/usr/bin/env python
# coding: utf-8

import csv

cnf = 8
run = 3

# Read the truth SNP IDs from the output.csv file generated in the previous step
truth_snps = set()
with open('../data/cnf-0{}-run-{}.TruthSNP.rsID.csv'.format(cnf, run), 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        rs_id = row[-1]  # Assuming rs ID is in the last column
        if rs_id != 'N/A':
            truth_snps.add(rs_id)

# Create a set to store truth SNPs from the second file
present_snps = set()

# Read truth SNPs from the second file
with open('../data/cnf-0{}-run-{}.onlyTruth.blocks'.format(cnf, run), 'r') as file2:
    for line in file2:
        rs_ids = line.strip().split()
        present_snps.update(rs_ids)

# Calculate the number of truth SNPs not present in the second file
unfound_truth_snps = truth_snps - present_snps
count_unfound = len(unfound_truth_snps)

print(f"Number of truth SNPs not present in the second file: {count_unfound}")
'''
textfile = open('../data/cnf-0{}-run-{}.SNPsinTruthBlocks.txt'.format(cnf, run), "w")
for element in present_snps:
    textfile.write(str(element) + "\n")
textfile.close()
'''
textfile = open('../data/cnf-0{}-run-{}.unfoundTruthSNPs.txt'.format(cnf, run), "w")
for element in unfound_truth_snps:
    textfile.write(str(element) + "\n")
textfile.close()