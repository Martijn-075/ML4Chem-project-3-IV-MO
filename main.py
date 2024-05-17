from rdkit import Chem
from rdkit.Chem import Draw
from SmilesGenerate import generate

# ======== start your code here =================================
batch_size = 64
smiles, pc = generate(batch_size=batch_size, temp=10)
n_print = 9
len_smiles = len(smiles)
indicies = []
mols = []
n = 0
if len_smiles >= n_print:

    while n < len_smiles and len(mols) < n_print:
        index = n-1
        if index not in indicies:
            indicies.append(index)
            mols.append(Chem.MolFromSmiles(smiles[index])) # type: ignore
            n += 1

    Draw.MolsToGridImage(mols)


# ======== end your code here ===================================