from Bio.SeqUtils.IsoelectricPoint import IsoelectricPoint as IP
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def gathering_params(sequence :str):
    params = dict()
    protein = IP(sequence)
    params['IEP'] = protein.pi()
    params['charge7'] = protein.charge_at_pH(7.4)
    params['charge9'] = protein.charge_at_pH(9.0)
    params['charge5'] = protein.charge_at_pH(5.5)
    protein = ProteinAnalysis(sequence)
    params['molecular_weight'] = protein.molecular_weight()
    params['instability_index'] = protein.instability_index()
    epsilon_protein = protein.molar_extinction_coefficient()
    params['extinction_coeff_cys'] = epsilon_protein[0] # reduced cysteines
    params['extinction_coeff_dis'] = epsilon_protein[1] # disulfid bridges
    return params

def main():
    viz = True
    df = pd.read_csv('align-mada-protein/predictive-pet-zero-shot-test-2025.csv')
    df['protein_params'] = df['sequence'].apply(gathering_params)
    protein_params_df = pd.json_normalize(df['protein_params'])
    df = pd.concat([df, protein_params_df], axis=1)
    df = df.drop(columns=['protein_params'])
    df.to_csv('zero_shot_param.csv', index=False) 

    if viz == True:
        numerical_cols = df.select_dtypes(include=np.number)

        correlation_matrix = numerical_cols.corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix of Protein Parameters')
        plt.savefig('plot_corr.png', dpi=300, bbox_inches='tight')
        plt.show()

main()