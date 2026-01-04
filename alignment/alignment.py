from Bio import pairwise2
from tqdm import tqdm
from joblib import Parallel, delayed

def find_best_match(mutated_seq, wildtype_list):
    """Your original function"""
    best_score = 0
    best_match = None
    seq_align = []
    seq_mutation_rate = []
    
    for wt_seq in wildtype_list:
        alignments = pairwise2.align.globalxx(mutated_seq.lower(), wt_seq.lower())
        seq_align.append(alignments[0].score)
        
        if alignments[0].score > best_score:
            best_score = alignments[0].score
            best_match = wt_seq
          
        if len(mutated_seq) == len(wt_seq):
            mutations = sum(1 for a, b in zip(mutated_seq.lower(), wt_seq.lower()) 
                          if a != b and a != '-' and b != '-')
        else:
            aligned_seq1 = alignments[0].seqA
            aligned_seq2 = alignments[0].seqB
            mutations = sum(1 for a, b in zip(aligned_seq1.lower(), aligned_seq2.lower()) 
                          if a != b and a != '-' and b != '-')
        
        total_length = len(wt_seq)
        mutation_rate = (mutations / total_length) * 100
        seq_mutation_rate.append(mutation_rate)
    
    return best_match, best_score, seq_align, seq_mutation_rate

import pandas as pd
wildtype = './align-mada-protein/pet-2025-wildtype-cds.csv'
mutation = './align-mada-protein/petase_zero_shot_submission_baseline.csv'
wd_df = pd.read_csv(wildtype)
mt_df = pd.read_csv(mutation)
print(wd_df.columns, mt_df.columns)

# Prepare data
wd_seq = wd_df['Wt AA Sequence'].tolist()
mt_seqs = mt_df['sequence'].tolist()

# Install joblib if you haven't: pip install joblib
print("Starting parallel processing...")

results = Parallel(n_jobs=-1, backend='loky')(
    delayed(find_best_match)(mt_seq, wd_seq) 
    for mt_seq in tqdm(mt_seqs, desc="Matching sequences", unit="seq")
)

# Unpack results
best_matches = [r[0] for r in results]
best_scores = [r[1] for r in results]
alignment_matrix = [r[2] for r in results]
mutation_matrix = [r[3] for r in results]

print(f"Completed! Processed {len(best_matches)} sequences")