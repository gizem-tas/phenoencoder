#!/usr/bin/env python
# coding: utf-8

import csv

cnf = 8
run = 3

present_snps = set()

# Read truth SNPs from the second file
with open('../data/cnf-0{}-run-{}.onlyTruth.blocks'.format(cnf, run), 'r') as file2:
    for line in file2:
        rs_ids = line.strip().split()
        present_snps.update(rs_ids)


textfile = open('../data/cnf-0{}-run-{}.SNPsinTruthBlocks.txt'.format(cnf, run), "w")
for element in present_snps:
    textfile.write(str(element) + "\n")
textfile.close()