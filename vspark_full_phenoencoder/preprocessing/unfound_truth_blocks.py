#!/usr/bin/env python
# coding: utf-8

import csv
import pandas as pd
cnf = 8
run = 3

# Read the truth SNP IDs from the output.csv file generated previously
truth_snps = pd.read_csv('../data/cnf-0{}-run-{}.TruthSNP.rsID.csv'.format(cnf, run), header=None)

# Open and read the text file
with open('../data/cnf-0{}-run-{}.unfoundTruthSNPs.txt'.format(cnf, run), "r") as textfile:
    lines = textfile.readlines()

# If you want to remove the newline characters at the end of each line:
unfound_truth_snps = [line.strip() for line in lines]

unfound_chr_pos = truth_snps[truth_snps.iloc[:,1].isin(unfound_truth_snps)].iloc[:,0]
print(unfound_chr_pos)

for snp in unfound_chr_pos:
#for snp in ["8:74809059:G:T"]: #For testing 
    rsid = truth_snps[truth_snps.iloc[:,0]==snp].iloc[:,1]
    print(rsid.values)
    chr_no = snp.split(":")[0]
    pos = int(snp.split(":")[1])
    print(snp, chr_no, pos)
    blocks=pd.read_fwf("../data/blocks_chr{}.blocks.det".format(chr_no),
                       widths=(5,15,15,10,5,100000))
    before = blocks[blocks['BP2'] < pos].tail(1)
    after = blocks[blocks['BP1'] > pos].head(1)
    combined = [bef + " " + rsid + " " + aft for bef, rsid, aft in zip(before.SNPS.values, rsid.values, after.SNPS.values)]
    combined = [element.replace('|', ' ') for element in combined]
    print(combined[0])
    # Open the file in append mode and write the new line
    with open('../data/cnf-0{}-run-{}.onlyTruth.blocks'.format(cnf, run), 'a') as file:
        file.write('* ' + combined[0] + '\n')  # Add a new line starting with *
