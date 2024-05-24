from SmilesGenerate import generate_properties, generate, bias_training_generation
from SmilesTrain import train
from VisualizeSmiles import Visualize_Molecules_from_Smiles
from rdkit.Chem import Descriptors
from rdkit import Chem
import matplotlib.pyplot as plt

before_smiles, _, _, _ = generate("SmilesLSTM_CHEMBL_22_22_epochs.pt", 1000)

property_score = []
for j, smile in enumerate(before_smiles):
    mol = Chem.MolFromSmiles(smile) # type: ignore
    property_score.append(Descriptors.MolLogP(mol)) # type: ignore
    
plt.hist(property_score)
plt.savefig("hist_logp_before.png")
plt.close()

smiles = bias_training_generation("SmilesLSTM_CHEMBL_22_22_epochs.pt", logp_target=-6., device="cpu", num_loop=5, batch_size=1000)
Visualize_Molecules_from_Smiles(smiles[0:8])
property_score = []
for j, smile in enumerate(smiles):
    mol = Chem.MolFromSmiles(smile) # type: ignore
    property_score.append(Descriptors.MolLogP(mol)) # type: ignore
    
plt.hist(property_score)
plt.savefig("hist_logp_after.png")