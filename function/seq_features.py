# seq_features.py
import numpy as np
from Bio.SeqUtils.IsoelectricPoint import IsoelectricPoint as IP
from Bio.SeqUtils.ProtParam import ProteinAnalysis

def net_charge(seq: str, pH: float) -> float:
    seq = seq.strip()
    if not seq:
        return 0.0
    
    protein = IP(seq)
    return protein.charge_at_pH(pH)

def mean_hydrophobicity(seq: str) -> float:
    protein = ProteinAnalysis(seq)
    return protein.gravy(scale='KyteDoolitle')

def pro_gly_fraction(seq: str) -> float:
    if not seq:
        return 0.0
    c = sum(1 for a in seq if a in ("P", "G"))
    return c / len(seq)

def all_params(sequence: str, params):
    seq_params = dict()
    protein = IP(sequence)
    seq_params['IEP'] = protein.pi()
    seq_params['charge7'] = protein.charge_at_pH(7.4)
    seq_params['charge9'] = protein.charge_at_pH(9.0)
    seq_params['charge55'] = protein.charge_at_pH(5.5)
    seq_params = ProteinAnalysis(sequence)
    seq_params['molecular_weight'] = protein.molecular_weight()
    seq_params['instability_index'] = protein.instability_index()
    seq_params['hydrophobicity'] = protein.gravy(scale='KyteDoolitle') #GRAVY (Grand Average of Hydropathy) according to Kyte and Doolitle, 1982.
    # epsilon_protein = protein.molar_extinction_coefficient()
    # seq_params['extinction_coeff_cys'] = epsilon_protein[0] # reduced cysteines
    # seq_params['extinction_coeff_dis'] = epsilon_protein[1] # disulfid bridges
    seq_params['pg_fraction'] = pro_gly_fraction(sequence)

    if params is None or len(params)==0:
        return(seq_params)
    
    else:
        return(seq_params[params])
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process protein sequences and optionally visualize correlations.")
    parser.add_argument('--input', type=str, default='align-mada-protein/dataset/predictive-pet-zero-shot-test-2025.csv',
                        help='Path to input CSV file')
    parser.add_argument('--output', type=str, default='align-mada-protein/output/param/zero_shot_param.csv',
                        help='Path to output CSV file')
    parser.add_argument('--viz', action='store_true', default=True,
                        help='Enable visualization (correlation heatmap). Default: True')
    parser.add_argument('--plot-output', type=str, default='plot_corr.png',
                        help='Path to save correlation plot')

    args = parser.parse_args()
    viz = True
    import pandas as pd
    df = pd.read_csv(args.input)
    df['protein_params'] = df['sequence'].apply(all_params)
    protein_params_df = pd.json_normalize(df['protein_params'])
    df = pd.concat([df, protein_params_df], axis=1)
    df = df.drop(columns=['protein_params'])
    df.to_csv(args.output, index=False) 

    if args.viz == True:
        import seaborn as sns
        import matplotlib.pyplot as plt
        numerical_cols = df.select_dtypes(include=np.number)

        correlation_matrix = numerical_cols.corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix of Protein Parameters')
        plt.savefig(args.plot_output, dpi=300, bbox_inches='tight')
        plt.show()