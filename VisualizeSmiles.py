"""

This module contains the routines that plot data used to visualy explore the data. This includes the data obtained during the training and exploring the traning data. This module is not vital for the rest of the code.

"""

from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
import pandas as pd
from rdkit.Chem import Descriptors
from rdkit.Chem.Draw import IPythonConsole
import matplotlib.pyplot as plt
from Smilesfeatures import MACCS_Tanimoto
    
    
def Visualize_Molecules_from_Smiles(smiles, filename="molgrid"):
    """
    Saves a list of SMILES as an grid image.

    Args:
        smiles (list): List containing the SMILES
        filename (str, optional): The name of the image to be saved. Defaults to "molgrid".
    """

    molecules = [Chem.MolFromSmiles(smile) for smile in smiles] # type: ignore

    IPythonConsole.UninstallIPythonRenderer()
    img = Draw.MolsToGridImage(molecules, molsPerRow=3)
    img.save(f'{filename}.png')


def train_data_MACCS(file="smiles_CHEMBL_22", n=1000):
    """
    Makes a hist plot of the Tanimoto score of the comapred to Curcumin. this is only used for visualization. it also saves 9 SMILES from the dataset.

    Args:
        file (str, optional): The name of the hist plot to be saved. Defaults to "smiles_CHEMBL_22".
        n (int, optional): The number of SMILES sampled from the train data. Defaults to 1000.
    """
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
    
    
def epoch_data(data_file="models/SmilesLSTM_CHEMBL_22_50_epoch.pt_epoch_data"):
    """
    Makes a scatter plot of the epoch data. One for the % valid SMILES and another for the epoch loss.

    Args:
        data_file (str, optional): the datafile conating the epoch traning data as generated during the training. Defaults to "models/SmilesLSTM_CHEMBL_22_50_epoch.pt_epoch_data".
    """
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
    """
    Makes a hist plot of the logp of the training dataset.

    Args:
        file (str, optional): the filename of the histplot to be saved. Defaults to "smiles_CHEMBL_22".
        n (int, optional): the number of samples taken from the dataset. Defaults to 10000.
    """
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
    

    
        

    