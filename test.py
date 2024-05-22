from SmilesGenerate import generate_properties, generate, bias_training_generation
from SmilesTrain import train
from VisualizeSmiles import Visualize_Molecules_from_Smiles

# smiles, pc, _, _ = generate(file="CHEMBL22_10ep.pt")
# Visualize_Molecules_from_Smiles(smiles)
# # print(smiles)

# train(save_file="test_bias_train2.pt", n_epochs=1)
smiles = bias_training_generation("test_bias_train.pt")
print(smiles)