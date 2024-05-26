from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
import pandas as pd
from rdkit.Chem import Descriptors
from rdkit.Chem.Draw import IPythonConsole
import matplotlib.pyplot as plt
from rdkit.Chem import rdAbbreviations # type: ignore
from SmilesGenerate import generate, bias_training_generation
from Smilesfeatures import MACCS_Tanimoto
import heapq

def Visualize_Molecules_from_Smiles(smiles, filename="molgrid"):

    molecules = [Chem.MolFromSmiles(smile) for smile in smiles] # type: ignore

    IPythonConsole.UninstallIPythonRenderer()
    img = Draw.MolsToGridImage(molecules, molsPerRow=3)
    img.save(f'{filename}.png')
    
def epoch_data(data_file="models/SmilesLSTM_CHEMBL_22_50_epoch.pt_epoch_data"):
    epoch_data = np.loadtxt(data_file, delimiter=" ")
    spacing = np.linspace(1, epoch_data.shape[0], epoch_data.shape[0])
    loss = epoch_data[:,0]
    pc = epoch_data[:,1]

    plt.scatter(spacing, loss)
    plt.title("Loss during training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"epoch_data_plots_loss.png")
    plt.close()
    
    plt.scatter(spacing, pc)
    plt.title("Valid smiles during training")
    plt.xlabel("Epoch")
    plt.ylabel("% valid smiles")
    plt.savefig(f"epoch_data_plots_pc.png")
    
def train_data_logp(file="smiles_CHEMBL_22", n=10000):
    data = pd.read_table(file, header=None)
    sub_data = data.sample(n)
    logp = np.zeros(n)
    i = 0
    str_smiles = []
    for _, smile in sub_data.iterrows():
        if i%6 == 0:
            str_smiles.append(str(smile[0]))
        mol = Chem.MolFromSmiles(str(smile[0])) # type: ignore
        logp[i] = Descriptors.MolLogP(mol) # type: ignore
        i += 1
        
    Visualize_Molecules_from_Smiles(str_smiles[0:9], "mols_train_data")
        
    plt.hist(logp)
    plt.title("Logp of the training set (n=10 000)")
    plt.xlabel("logp")
    plt.ylabel("Count")
    plt.savefig(f"logp_train_data.png")
    
def comp_MACCS(file="SmilesLSTM_CHEMBL_22_50_epoch.pt"):
    smiles, pc = bias_training_generation(file, propertie="MACCS", target_mol=r"O=C(\C=C\c1ccc(O)c(OC)c1)CC(=O)\C=C\c2cc(OC)c(O)cc2", device="cuda", num_loop=50, batch_size=1000, lr=0.000001)
    
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
    smiles, pc = bias_training_generation(file, propertie="logp", target_logp=6., device="cuda", num_loop=50, batch_size=1000, lr=0.000001)

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
    
def train_data_MACCS(file="smiles_CHEMBL_22", n=1000):
    data = pd.read_table(file, header=None)
    sub_data = data.sample(n)
    MACCS = np.zeros(n)
    i = 0
    str_smiles = []
    for _, smile in sub_data.iterrows():
        if i%6 == 0:
            str_smiles.append(str(smile[0]))
        MACCS[i] = MACCS_Tanimoto(str(smile[0]), r"O=C(\C=C\c1ccc(O)c(OC)c1)CC(=O)\C=C\c2cc(OC)c(O)cc2") # type: ignore
        i += 1
        
    Visualize_Molecules_from_Smiles(str_smiles[0:9], "mols_train_data_MACCS")
        
        
    plt.hist(MACCS)
    plt.title("MACCS Tanimoto of the training set (n=10 000)")
    plt.xlabel("MACCS Tanimoto")
    plt.ylabel("Count")
    plt.savefig(f"logp_train_data_MACCS.png")
    

    
        

    