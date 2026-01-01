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
    seq_params['charge5'] = protein.charge_at_pH(5.5)
    seq_params = ProteinAnalysis(sequence)
    seq_params['molecular_weight'] = protein.molecular_weight()
    seq_params['instability_index'] = protein.instability_index()
    seq_params['hydrophobicity'] = protein.gravy(scale='KyteDoolitle') #GRAVY (Grand Average of Hydropathy) according to Kyte and Doolitle, 1982.
    epsilon_protein = protein.molar_extinction_coefficient()
    seq_params['extinction_coeff_cys'] = epsilon_protein[0] # reduced cysteines
    seq_params['extinction_coeff_dis'] = epsilon_protein[1] # disulfid bridges
    seq_params['PG_fraction'] = pro_gly_fraction(sequence)

    if params is None or len(params)==0:
        return(seq_params)
    
    else:
        return(seq_params[params])