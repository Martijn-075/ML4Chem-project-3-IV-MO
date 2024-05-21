from SmilesGenerate import generate_properties, generate
from VisualizeSmiles import Visualize_Molecules_from_Smiles

smiles, pc, _, _ = generate(file="CHEMBL22_10ep.pt")
Visualize_Molecules_from_Smiles(smiles)
# print(smiles)