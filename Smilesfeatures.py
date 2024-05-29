from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs

def MACCS_Tanimoto(smile, target_smile):
    """
    Given two SMILES strings the Tanimoto similarity is calcualted using the MACCS fingerprinting. This function is used during the RL training to give a score to the generated SMILES. This function uses the MACCS and Tanimoto similarity from RDKit.

    Args:
        smile (str): The (generated) SMILES wich should be comapred to the target SMILES
        target_smile (str): The target SMILES.

    Returns:
        Similarity (float): The Tanimoto similarity score.
    """
    smile = Chem.MolFromSmiles(smile) # type: ignore
    target_smile = Chem.MolFromSmiles(target_smile) # type: ignore
    
    smile_MACCS = MACCSkeys.GenMACCSKeys(smile)
    target_smile_MACCS = MACCSkeys.GenMACCSKeys(target_smile)
    
    Similarity = DataStructs.TanimotoSimilarity(smile_MACCS, target_smile_MACCS) # type: ignore
    return Similarity





    