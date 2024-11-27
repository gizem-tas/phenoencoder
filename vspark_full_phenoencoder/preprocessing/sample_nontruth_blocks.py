'''
This script assumes that each line in the block files starts with an asterisk (*) 
followed by SNP names separated by whitespace. 
The script reads the existing truth blocks, generates a list of non-truth blocks,
and then samples new blocks while attempting to match the distribution of block sizes from the truth blocks. 
The final sampled blocks are written to a new file named sampled_blocks.blocks.
'''
import random
from collections import Counter

# Step 1: Read truth.blocks file
with open('truth.blocks', 'r') as truth_file:
    truth_blocks = {line.strip() for line in truth_file}

# Step 2: Read chromosome block files
chromosome_blocks = []
for i in range(1, 23):
    file_name = f'blocks_chr{i}.blocks'
    with open(file_name, 'r') as chromosome_file:
        chromosome_blocks.extend({line.strip() for line in chromosome_file})

# Create a list of blocks that are not in truth.blocks
non_truth_blocks = list(filter(lambda block: block not in truth_blocks, chromosome_blocks))

# Step 3: Calculate distribution of block sizes in truth.blocks
truth_block_sizes = [len(block.split()) - 1 for block in truth_blocks]
truth_block_size_distribution = Counter(truth_block_sizes)

# Step 4: Randomly sample blocks from non-truth blocks while matching the distribution
sampled_blocks = []

for _ in range(len(truth_blocks)):
    # Randomly select a block from non-truth blocks
    block = random.choice(non_truth_blocks)
    
    # Calculate the size of the sampled block
    sampled_size = len(block.split()) - 1
    
    # Match the distribution if possible
    while truth_block_size_distribution[sampled_size] == 0:
        block = random.choice(non_truth_blocks)
        sampled_size = len(block.split()) - 1
    
    # Add the sampled block to the result
    sampled_blocks.append(block)
    
    # Update the distribution
    truth_block_size_distribution[sampled_size] -= 1

# Write the sampled blocks to a new file (e.g., sampled_blocks.blocks)
with open('sampled_blocks.blocks', 'w') as sampled_file:
    sampled_file.write('\n'.join(sampled_blocks))
