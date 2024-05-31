"""

This module contains the RL comparison routines. These functions are only ment as example routines comapring the samples before and after the RL training loop.

"""

from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt
from SmilesGenerate import generate, RL_training_generation
from Smilesfeatures import MACCS_Tanimoto
from VisualizeSmiles import Visualize_Molecules_from_Smiles
import heapq
   
    
def comp_MACCS(file="SmilesLSTM_CHEMBL_22_50_epoch.pt", target=r"O=C(\C=C\c1ccc(O)c(OC)c1)CC(=O)\C=C\c2cc(OC)c(O)cc2"):
    """
    Comparing the the RL training routine. First SMILES generated with the base model are evaluated and are plotted. Then after the RL training again SMILES are sampled and are scored and plotted. Curcumin is used as an example target mol.

    Args:
        file (str, optional): The file to the trained model. Defaults to "SmilesLSTM_CHEMBL_22_50_epoch.pt".
        target (raw str, optional): The raw string of a target SMILES. Must be a raw string becuase of posible escape caracters being present in SMILES. Defaults to SMILE of Curcumin
    """
    # Generate SMILES beofre the RL training.
    smiles, _, _, _ = generate(file=file, batch_size=1000)
    property_score = []
    for smile in smiles:
        property_score.append(MACCS_Tanimoto(smile, target)) # type: ignore
    
    # Make a hist plot of the Tanimoto score of the gen SMILES before RL training.
    plt.hist(property_score)
    plt.title("MACCS Tanimoto of generated smiles (base)")
    plt.xlabel("Tanimoto score")
    plt.ylabel("Count")
    plt.savefig("hist_MACCS_before.png")
    plt.close()
    
    # The RL training loop
    smiles, pc = RL_training_generation(file, propertie="MACCS", target_mol=target, device="cuda", num_loop=50, batch_size=1000, lr=0.000001)
    
    # Scoring the SMILES after RL training
    property_score = []
    for smile in smiles:
        property_score.append(MACCS_Tanimoto(smile, target)) # type: ignore
    mol = Chem.MolFromSmiles(smile) # type: ignore
    # Saving the first 9 SMILES as an image
    Visualize_Molecules_from_Smiles(smiles[0:9], "mols_bias_train")
    
    # Getting the indicies of the 3 nearest and furthes neighbors of the target mol.
    hig3 = heapq.nlargest(3, range(len(property_score)), key=lambda x: property_score[x])
    smal3 = heapq.nsmallest(3, range(len(property_score)), key=lambda x: property_score[x])

    # Retriving the SMILES from the indicies of the nearest and furthest neighbors
    high_smile = []
    for i in hig3:
        high_smile.append(smiles[i])
        
    smal_smile = []
    for i in smal3:
        smal_smile.append(smiles[i])
    
    # Saving the nearest and furthest neighbors as an image
    Visualize_Molecules_from_Smiles(high_smile, '3high_smiles.png')
    Visualize_Molecules_from_Smiles(smal_smile, '3smal_smiles.png')
    
    # plotting the Tanimoto score of the SMILES after RL training
    plt.hist(property_score)
    plt.title("MACCS Tanimoto of generated smiles (RL)")
    plt.xlabel("Tanimoto score")
    plt.ylabel("Count")
    plt.savefig("hist_MACCS_after.png")
    
    return smiles, pc


def comp_logp(file="SmilesLSTM_CHEMBL_22_50_epoch.pt", target=6.):
    """
        Comparing the the RL training routine. First SMILES generated with the base model are evaluated and are plotted. Then after the RL training again SMILES are sampled and are scored and plotted. Curcumin is used as an example target mol.

    Args:
        file (str, optional): File path to the trained model. Defaults to "SmilesLSTM_CHEMBL_22_50_epoch.pt".
        target (float, optional): The logp target for the RL training . Defaults to 6..
    """
    # Generating SMILES before RL training
    smiles, _, _, _ = generate(file=file, batch_size=1000)
    property_score = np.zeros((len(smiles)))
    # get logp of generated SMILES before RL
    for i, smile in enumerate(smiles):
        mol = Chem.MolFromSmiles(smile) # type: ignore
        property_score[i] = Descriptors.MolLogP(mol) # type: ignore
    
    # Making a hist plot of the logp of the generated SMIELS before RL training.
    plt.hist(property_score)
    plt.title("Logp of generated smiles (base)")
    plt.xlabel("Logp")
    plt.ylabel("Count")
    plt.savefig("hist_logp_before.png")
    plt.close()
    
    #  The RL training loop
    smiles, pc = RL_training_generation(file, propertie="logp", target_logp=target, device="cuda", num_loop=50, batch_size=1000, lr=0.000001)

    # getting the logp of the SMILES after RL
    property_score = np.zeros((len(smiles)))
    for i, smile in enumerate(smiles):
        mol = Chem.MolFromSmiles(smile) # type: ignore
        property_score[i] = Descriptors.MolLogP(mol) # type: ignore

    # Saving the first 9 SMILES afte RL as an image
    Visualize_Molecules_from_Smiles(smiles[0:9], "mols_bias_train_logp")

    # Making a hist plot of the logp of the generated SMILES after RL.
    plt.hist(property_score)
    plt.title("Logp of generated smiles (RL)")
    plt.xlabel("Logp")
    plt.ylabel("Count")
    plt.savefig("hist_logp_after.png")
    
    return smiles, pc
