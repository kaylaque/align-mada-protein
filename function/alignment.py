import argparse
import pandas as pd
from Bio import pairwise2
from tqdm import tqdm
from joblib import Parallel, delayed
import os

def find_best_match(mutated_seq, wildtype_list):
    """Find best wildtype match for a mutated sequence using global alignment."""
    best_score = 0
    best_match = None
    seq_align = []
    seq_mutation_rate = []
    
    for wt_seq in wildtype_list:
        # Perform global alignment
        alignments = pairwise2.align.globalxx(mutated_seq.lower(), wt_seq.lower())
        align_score = alignments[0].score
        seq_align.append(align_score)
        
        # Track best match
        if align_score > best_score:
            best_score = align_score
            best_match = wt_seq
          
        # Calculate mutation rate
        if len(mutated_seq) == len(wt_seq):
            mutations = sum(
                1 for a, b in zip(mutated_seq.lower(), wt_seq.lower()) 
                if a != b and a != '-' and b != '-'
            )
        else:
            aligned_seq1 = alignments[0].seqA
            aligned_seq2 = alignments[0].seqB
            mutations = sum(
                1 for a, b in zip(aligned_seq1.lower(), aligned_seq2.lower()) 
                if a != b and a != '-' and b != '-'
            )
        
        total_length = len(wt_seq)
        mutation_rate = (mutations / total_length) * 100
        seq_mutation_rate.append(mutation_rate)
    
    return best_match, best_score, seq_align, seq_mutation_rate


def main_process(wildtype, mutation, n_jobs, backend, output, seq_col):
        print("BLOSSUM62 ALIGNMENT: Loading data...")
        wd_df = pd.read_csv(wildtype)
        mt_df = pd.read_csv(mutation)
        
        if 'Wt AA Sequence' not in wd_df.columns:
            raise ValueError("Wildtype CSV must contain 'Wt AA Sequence' column")
        if seq_col not in mt_df.columns:
            raise ValueError("Mutation CSV must contain 'sequence' column")

        wd_seq = wd_df['Wt AA Sequence'].tolist()
        mt_seqs = mt_df[seq_col].tolist()

        print(f"Processing {len(mt_seqs)} mutated sequences against {len(wd_seq)} wildtypes...")

        # Run parallel matching
        results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(find_best_match)(mt_seq, wd_seq) 
            for mt_seq in tqdm(mt_seqs, desc="Matching sequences", unit="seq")
        )

        # Unpack results
        best_matches = [r[0] for r in results]
        best_scores = [r[1] for r in results]
        alignment_matrix = [r[2] for r in results]
        mutation_matrix = [r[3] for r in results]

        # Create output DataFrame
        output_df = pd.DataFrame({
            'mutated_sequence': mt_seqs,
            'best_wildtype_match': best_matches,
            'best_alignment_score': best_scores,
            'all_alignment_scores': alignment_matrix,
            'mutation_rates_percent': mutation_matrix
        })

        # Save results
        output_df.to_csv(output, index=False)
        print(f"\n✅ Completed! Results saved to: {output}")
        print(f"   Processed {len(best_matches)} sequences")

def main():
    parser = argparse.ArgumentParser(
        description="Find best wildtype matches for mutated protein sequences."
    )
    parser.add_argument(
        "--wildtype", 
        type=str, 
        required=True,
        help="Path to wildtype CSV file (must contain 'Wt AA Sequence' column)"
    )
    parser.add_argument(
        "--mutation", 
        type=str, 
        required=True,
        help="Path to mutation CSV file (must contain 'sequence' column)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="alignment_results.csv",
        help="Output CSV filename (default: alignment_results.csv)"
    )
    parser.add_argument(
        "--n-jobs", 
        type=int, 
        default=-1,
        help="Number of parallel jobs (-1 = all cores, default: -1)"
    )
    parser.add_argument(
        "--backend", 
        type=str, 
        default="loky",
        choices=["loky", "multiprocessing", "threading"],
        help="Joblib backend (default: loky)"
    )
    
    args = parser.parse_args()

    # Validate input files
    if not os.path.exists(args.wildtype):
        raise FileNotFoundError(f"Wildtype file not found: {args.wildtype}")
    if not os.path.exists(args.mutation):
        raise FileNotFoundError(f"Mutation file not found: {args.mutation}")
    
    main_process(args.wildtype, args.mutation, args.n_jobs, args.backend, args.output)

if __name__ == "__main__":
    main()