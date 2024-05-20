from rdkit import Chem
from rdkit.Chem import Draw
import torch
from torch.nn.functional import one_hot
from SmilesGenerate import generate
from SmilesData import __special__


# box= torch.load('SmilesLSTM5epddropouts.pt')
# model,tokenizer = box['model'],box['tokenizer']
# model.eval()

# bos_token = [k for k,v in __special__.items() if v == "<BOS>"][0]
# x = torch.LongTensor(bos_token)
# x = one_hot(x, 37).float().unsqueeze(1).to('cpu')

# res = model(x)
# print(x.size())
# print(res.size())

# ======== start your code here =================================
batch_size = 100
smiles, pc = generate(file="models/CHEMBL22_10ep.pt", batch_size=batch_size, temp=1)
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

    Draw.MolsToImage(mols)
    
print(f'{pc}% valid strings')
# print(smiles)

# ======== end your code here ===================================