from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs

def MACCS_Tanimoto(smile, target_smile):
    smile = Chem.MolFromSmiles(smile) # type: ignore
    target_smile = Chem.MolFromSmiles(target_smile) # type: ignore
    
    smile_MACCS = MACCSkeys.GenMACCSKeys(smile)
    target_smile_MACCS = MACCSkeys.GenMACCSKeys(target_smile)
    
    Similarity = DataStructs.TanimotoSimilarity(smile_MACCS, target_smile_MACCS) # type: ignore
    return Similarity

# mol = Chem.MolFromSmiles("CC1(C)[C@@H](N2[C@@H]([C@H](Cc3cn(nn3)c4ccccc4)C2=O)S1(=O)=O)C(=O)O") # type: ignore
# logp = Descriptors.MolLogP(mol) # type: ignore

# print(logp)




    