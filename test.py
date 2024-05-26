from SmilesGenerate import generate_properties, generate, bias_training_generation
from SmilesTrain import train, post_train
from VisualizeSmiles import Visualize_Molecules_from_Smiles
from rdkit.Chem import Descriptors
from rdkit import Chem
import numpy as np
import matplotlib.pyplot as plt

epoch_data = np.loadtxt("models/SmilesLSTM_CHEMBL_22_50_epoch.pt_pc_data")
spacing = np.linspace(1, epoch_data.shape[0], epoch_data.shape[0])

pc = epoch_data

plt.scatter(spacing, pc)
plt.title("Valid smiles during RL")
plt.xlabel("iteration")
plt.ylabel("% valid smiles")
plt.savefig(f"bias_train_pc.png")
plt.close()


