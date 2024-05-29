from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt
from SmilesGenerate import generate, RL_training_generation
from Smilesfeatures import MACCS_Tanimoto
from VisualizeSmiles import Visualize_Molecules_from_Smiles
import heapq
    
def comp_MACCS(file="SmilesLSTM_CHEMBL_22_50_epoch.pt"):
    smiles, _, _, _ = generate(file=file, batch_size=1000)
    property_score = []
    for smile in smiles:
        property_score.append(MACCS_Tanimoto(smile, r"O=C(\C=C\c1ccc(O)c(OC)c1)CC(=O)\C=C\c2cc(OC)c(O)cc2")) # type: ignore
    
    plt.hist(property_score)
    plt.title("MACCS Tanimoto of generated smiles (base)")
    plt.xlabel("Tanimoto score")
    plt.ylabel("Count")
    plt.savefig("hist_MACCS_before.png")
    
    smiles, pc = RL_training_generation(file, propertie="MACCS", target_mol=r"O=C(\C=C\c1ccc(O)c(OC)c1)CC(=O)\C=C\c2cc(OC)c(O)cc2", device="cuda", num_loop=50, batch_size=1000, lr=0.000001)
    
    property_score = []
    for smile in smiles:
        property_score.append(MACCS_Tanimoto(smile, r"O=C(\C=C\c1ccc(O)c(OC)c1)CC(=O)\C=C\c2cc(OC)c(O)cc2")) # type: ignore
    mol = Chem.MolFromSmiles(smile) # type: ignore
    Visualize_Molecules_from_Smiles(smiles[0:9], "mols_bias_train")
    img = Draw.MolToFile(mol, "bias_train_mol_MACCS.png")
    
    hig3 = heapq.nlargest(3, range(len(property_score)), key=lambda x: property_score[x])
    smal3 = heapq.nsmallest(3, range(len(property_score)), key=lambda x: property_score[x])
    print(hig3, smal3)
    high_smile = []
    for i in hig3:
        high_smile.append(smiles[i])
        
    smal_smile = []
    for i in smal3:
        smal_smile.append(smiles[i])
    Visualize_Molecules_from_Smiles(high_smile, '3high_smiles.png')
    Visualize_Molecules_from_Smiles(smal_smile, '3smal_smiles.png')
        
    plt.hist(property_score)
    plt.title("MACCS Tanimoto of generated smiles (RL)")
    plt.xlabel("Tanimoto score")
    plt.ylabel("Count")
    plt.savefig("hist_MACCS_after.png")

def comp_logp(file="SmilesLSTM_CHEMBL_22_50_epoch.pt"):
    smiles, pc = RL_training_generation(file, propertie="logp", target_logp=6., device="cuda", num_loop=50, batch_size=1000, lr=0.000001)

    property_score = np.zeros((len(smiles)))
    for i, smile in enumerate(smiles):
        mol = Chem.MolFromSmiles(smile) # type: ignore
        property_score[i] = Descriptors.MolLogP(mol) # type: ignore

    Visualize_Molecules_from_Smiles(smiles[0:9], "mols_bias_train_logp")

    plt.hist(property_score)
    plt.title("Logp of generated smiles (RL)")
    plt.xlabel("Logp")
    plt.ylabel("Count")
    plt.savefig("hist_logp_after.png")
    print(mol)