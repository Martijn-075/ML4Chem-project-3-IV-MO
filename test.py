from SmilesGenerate import generate_properties, generate, bias_training_generation
from SmilesTrain import train, post_train
from VisualizeSmiles import Visualize_Molecules_from_Smiles
from rdkit.Chem import Descriptors
from rdkit import Chem
import matplotlib.pyplot as plt

train(save_file="SmilesLSTM_CHEMBL_22_50_epoch.pt", n_epochs=50)


