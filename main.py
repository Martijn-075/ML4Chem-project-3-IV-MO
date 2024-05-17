import random
from rdkit import Chem
from rdkit.Chem import Draw
from model import generate

batch_size = 64
smiles, pc = generate(batch_size=batch_size)
n_print = 9
len_smiles = len(smiles)
indicies = []
mols = []
n = 0
while n < n_print:
    index = int(random.random() * len_smiles)
    if index not in indicies:
        indicies.append(index)
        mols.append(Chem.MolFromSmiles(smiles[index]))
        n += 1

Draw.MolsToGridImage(mols)