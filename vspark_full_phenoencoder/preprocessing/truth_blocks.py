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

# Create a single output file to store all blocks containing truth SNPs
output_file_path = '../data/cnf-0{}-run-{}.onlyTruth.blocks'.format(cnf, run)

with open(output_file_path, 'w') as output_file:
    # Process each chromosome file
    for chromosome_number in range(1, 23):
        chromosome_file_path = f'../data/blocks_chr{chromosome_number}.blocks' 

        with open(chromosome_file_path, 'r') as chromosome_file:
            block = []
            for line in chromosome_file:
                snps_in_block = line.strip().split()
                if any(rs_id in truth_snps for rs_id in snps_in_block):
                    block.extend(snps_in_block)
                else:
                    if block:
                        output_file.write(' '.join(block) + '\n')
                        block = []
            # Write the last block if it contains truth SNPs
            if block:
                output_file.write(' '.join(block) + '\n')

print("Blocks containing truth SNPs written to", output_file_path)