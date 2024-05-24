from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
from rdkit.Chem.Draw import IPythonConsole
import matplotlib.pyplot as plt
from rdkit.Chem import rdAbbreviations # type: ignore

def Visualize_Molecules_from_Smiles(smiles):

    molecules = [Chem.MolFromSmiles(smile) for smile in smiles] # type: ignore

    IPythonConsole.UninstallIPythonRenderer()
    img = Draw.MolsToGridImage(molecules, molsPerRow=8)
    img.save('molgrid.png')
    
def epoch_data(data_file="models/SmilesLSTM_CHEMBL_22_22_epochs.pt_epoch_data"):
    epoch_data = np.loadtxt(data_file, delimiter=" ")
    spacing = np.linspace(1, epoch_data.shape[0], epoch_data.shape[0])
    loss = epoch_data[:,0]
    pc = epoch_data[:,1]

    plt.scatter(spacing, loss)
    plt.savefig(f"epoch_data_plots_loss.png")
    plt.close()
    
    plt.scatter(spacing, pc)
    plt.savefig(f"epoch_data_plots_.png")
    