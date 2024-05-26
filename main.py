from SmilesGenerate import generate_properties, generate, bias_training_generation
from SmilesTrain import train
from VisualizeSmiles import Visualize_Molecules_from_Smiles
from rdkit.Chem import Descriptors
from rdkit import Chem
import matplotlib.pyplot as plt
from rdkit.Chem import Draw

_, pc, _, _= generate(file="SmilesLSTM_CHEMBL_22_50_epoch.pt", batch_size=1000, temp=2)
print(pc)