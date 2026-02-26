import pickle
import numpy as np
import pandas as pd
import argparse

from huggingface_hub import snapshot_download, hf_hub_download
import os

import torch
from UniKP.build_vocab import WordVocab
from UniKP.pretrain_trfm import TrfmSeq2seq
from UniKP.utils import split
# build_vocab, pretrain_trfm, utils packages are from SMILES Transformer
from transformers import T5EncoderModel, T5Tokenizer
# transformers package is from ProtTrans
import re
import gc
import numpy as np
import pandas as pd
import pickle
import math


def smiles_to_vec(Smiles):
    pad_index = 0
    unk_index = 1
    eos_index = 2
    sos_index = 3
    mask_index = 4
    vocab = WordVocab.load_vocab('vocab.pkl')
    def get_inputs(sm):
        seq_len = 220
        sm = sm.split()
        if len(sm)>218:
            print('SMILES is too long ({:d})'.format(len(sm)))
            sm = sm[:109]+sm[-109:]
        ids = [vocab.stoi.get(token, unk_index) for token in sm]
        ids = [sos_index] + ids + [eos_index]
        seg = [1]*len(ids)
        padding = [pad_index]*(seq_len - len(ids))
        ids.extend(padding), seg.extend(padding)
        return ids, seg
    def get_array(smiles):
        x_id, x_seg = [], []
        for sm in smiles:
            a,b = get_inputs(sm)
            x_id.append(a)
            x_seg.append(b)
        return torch.tensor(x_id), torch.tensor(x_seg)
    trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    trfm.load_state_dict(torch.load('trfm_12_23000.pkl'))
    trfm.eval()
    x_split = [split(sm) for sm in Smiles]
    xid, xseg = get_array(x_split)
    X = trfm.encode(torch.t(xid))
    return X


def Seq_to_vec(Sequence):
    for i in range(len(Sequence)):
        if len(Sequence[i]) > 1000:
            Sequence[i] = Sequence[i][:500] + Sequence[i][-500:]
    sequences_Example = []
    for i in range(len(Sequence)):
        zj = ''
        for j in range(len(Sequence[i]) - 1):
            zj += Sequence[i][j] + ' '
        zj += Sequence[i][-1]
        sequences_Example.append(zj)
    ###### you should place downloaded model into this directory.
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    gc.collect()
    print(torch.cuda.is_available())
    # 'cuda:0' if torch.cuda.is_available() else
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    features = []
    for i in range(len(sequences_Example)):
        print('For sequence ', str(i+1))
        sequences_Example_i = sequences_Example[i]
        sequences_Example_i = [re.sub(r"[UZOB]", "X", sequences_Example_i)]
        ids = tokenizer.batch_encode_plus(sequences_Example_i, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu().numpy()
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len - 1]
            features.append(seq_emd)
    features_normalize = np.zeros([len(features), len(features[0][0])], dtype=float)
    for i in range(len(features)):
        for k in range(len(features[0][0])):
            for j in range(len(features[i])):
                features_normalize[i][k] += features[i][j][k]
            features_normalize[i][k] /= len(features[i])
    return features_normalize

# Download entire model repository
def download_full_model(model_id, local_dir="./models"):
    """
    Download complete model with all files
    
    Args:
        model_id: HuggingFace model identifier (e.g., "bert-base-uncased")
        local_dir: Local directory to save the model
    """
    print(f"Downloading {model_id}...")
    
    model_path = snapshot_download(
        repo_id=model_id,
        local_dir=os.path.join(local_dir, model_id.replace("/", "_")),
        local_dir_use_symlinks=False
    )
    
    print(f"Model saved to: {model_path}")
    return model_path

class CompatibilityUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'sklearn.tree._tree' and name == 'Tree':
            from sklearn.tree import _tree
            original_tree = _tree.Tree
            
            class CompatibleTree(original_tree):
                def __setstate__(self, state):
                    if 'nodes' in state and isinstance(state['nodes'], np.ndarray):
                        nodes = state['nodes']
                        if 'missing_go_to_left' not in nodes.dtype.names:
                            new_dtype = np.dtype([
                                ('left_child', '<i8'), ('right_child', '<i8'),
                                ('feature', '<i8'), ('threshold', '<f8'),
                                ('impurity', '<f8'), ('n_node_samples', '<i8'),
                                ('weighted_n_node_samples', '<f8'),
                                ('missing_go_to_left', 'u1')
                            ])
                            new_nodes = np.zeros(nodes.shape, dtype=new_dtype)
                            for field in nodes.dtype.names:
                                new_nodes[field] = nodes[field]
                            new_nodes['missing_go_to_left'] = 0
                            state['nodes'] = new_nodes
                    super(CompatibleTree, self).__setstate__(state)
            
            return CompatibleTree
        return super().find_class(module, name)

def load_old_model(model_path):
    with open(model_path, 'rb') as f:
        model = CompatibilityUnpickler(f).load()
    from sklearn.tree import DecisionTreeRegressor

    # After loading model, add this:
    if not hasattr(model, 'estimator'):
        model.estimator = DecisionTreeRegressor()

    if not hasattr(model, 'n_features_in_'):
        if hasattr(model, 'estimators_') and len(model.estimators_) > 0:
            model.n_features_in_ = model.estimators_[0].n_features_in_

    return model
def load_model(MODEL_PATH):
    # Load your model
    model_kcat = load_old_model(MODEL_PATH + '/UniKP for kcat.pkl')
    print("Model loaded successfully!")

    model_km = load_old_model(MODEL_PATH + '/UniKP for Km.pkl')
    print("Model loaded successfully!")

    model_kcatkm = load_old_model(MODEL_PATH + '/UniKP for kcat_Km.pkl')
    print("Model loaded successfully!")
    return(model_kcat, model_km, model_kcatkm)

def main_process(sequences, output):
    # download model from HF
    model_name = "HanselYu/UniKP"
    download_full_model(model_name)
    MODEL_PATH = './models/HanselYu_UniKP'
    smiles = ['OC1=CC=C(C[C@@H](C(O)=O)N)C=C1']
    seq_vec = Seq_to_vec(sequences)
    smiles_vec = smiles_to_vec(smiles)
    fused_vector = np.concatenate((smiles_vec, seq_vec), axis=1)
    
    model_kcat, model_km, model_kcatkm = load_model(MODEL_PATH)
    pre_label = model_kcat.predict(fused_vector)
    pre_label_pow_kcat = [math.pow(10, pre_label[i]) for i in range(len(pre_label))]
    
    pre_label = model_km.predict(fused_vector)
    pre_label_pow_km = [math.pow(10, pre_label[i]) for i in range(len(pre_label))]
    
    pre_label = model_kcatkm.predict(fused_vector)
    pre_label_pow_kcat_km = [math.pow(10, pre_label[i]) for i in range(len(Pre_label))]

    res = pd.DataFrame({'sequences': sequences, 'Smiles': smiles, 
                        'Pre_label_Kcat': pre_label_pow_kcat,
                        'Pre_label_Km': pre_label_pow_km,
                        'Pre_label_Kcat_km': pre_label_pow_kcat_km})
    res.to_csv(output, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Find best wildtype matches for mutated protein sequences."
    )
    parser.add_argument(
        "--input", 
        type=str, 
        required=True,
        help="Path to wildtype CSV file (must contain 'Wt AA Sequence' column)"
    )
    parser.add_argument(
        "--colseq", 
        type=str, 
        required=True,
        help="column for sequence"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="kinetic_parameter.csv",
        help="Output CSV filename (default: alignment_results.csv)"
    )
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    sequences = df[args.colseq].tolist()
    main_process(sequences, args.output)