from smile_to_data import *
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
import numpy as np
from pubchemfp import GetPubChemFPs
def smiles_to_data(smiles: str) -> Data:
    mol = smiles_to_3d_mol(smiles)
    data = mol_to_data(mol)
    return data


def extract_fingerprints(smiles: str) -> torch.Tensor:

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("无法从SMILES生成分子对象，请检查输入的SMILES格式。")


    fp_maccs = np.array(AllChem.GetMACCSKeysFingerprint(mol), dtype=int)
    fp_phaErGfp = np.array(AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1),
                           dtype=int)
    fp_pubcfp = np.array(GetPubChemFPs(mol), dtype=int)
    fp_ecfp2 = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024), dtype=int)


    fingerprint_combined = np.concatenate([fp_maccs, fp_phaErGfp, fp_pubcfp, fp_ecfp2])


    fingerprint_tensor = torch.tensor(fingerprint_combined, dtype=torch.float32)

    return fingerprint_tensor


