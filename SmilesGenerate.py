import torch
import torch.nn.functional as F
from rdkit import Chem
import numpy as np
from rdkit.Chem import Descriptors
from SmilesData import __special__

def generate(file='SmilesLSTM30ep.pt',batch_size=64, temp=1., h=None, c=None):
    """
    This is the entrypoint for the generator of SMILES
    :param file: A file with pretrained model
    :param batch_size: The number of compounds to generate
    :return: None. It prints a list of generated compounds to stdout
    """
    file = f"models/{file}"
    box= torch.load(file)
    model,tokenizer = box['model'],box['tokenizer']
    model.eval()
    res, h, c = model.sample(batch_size, temp=temp, h=h, c=c)
    correct = 0
    list_smiles = []
    for i in range(res.size(0)):
        smiles = "".join([tokenizer[index] for index in res[i].tolist() if index not in __special__])
        # print(smiles)
# ======== start your code here =================================
        if Chem.MolFromSmiles(smiles) != None: # type: ignore
            correct += 1
            list_smiles.append(smiles)
# ======== end your code here ===================================
    p_valid_smiles = correct/float(batch_size)*100
    # print (f"{p_valid_smiles} % are valid smiles")
    return list_smiles, p_valid_smiles, h, c

def generate_current_model(model, tokenizer, batch_size=100, temp=1.):
    model.eval()
    res = model.sample(batch_size, temp=temp)
    correct = 0
    list_smiles = []
    for i in range(res.size(0)):
        smiles = "".join([tokenizer[index] for index in res[i].tolist() if index not in __special__])
        # print(smiles)

        if Chem.MolFromSmiles(smiles) != None: # type: ignore
            correct += 1
            list_smiles.append(smiles)

    p_valid_smiles = correct/float(batch_size)*100
    
    return list_smiles, p_valid_smiles

def generate_properties(file='SmilesLSTM30ep.pt',batch_size=10, num_loop=5, temp=1.):
    h = None
    c = None
    target = 10.
    for i in range(num_loop):
        smiles, pc, h, c = generate(file=file, batch_size=batch_size, temp=temp, h=h, c=c)
        property_score = np.zeros(len(smiles))
        
        for j, smile in enumerate(smiles):
            mol = Chem.MolFromSmiles(smile) # type: ignore
            property_score[j] = Descriptors.MolLogP(mol) # type: ignore
            best_index = np.absolute(property_score - target).argmin()

        h = h[:, best_index, :].unsqueeze(dim=1).repeat(1, batch_size, 1)
        c = h[:, best_index, :].unsqueeze(dim=1).repeat(1, batch_size, 1)

        
    return smiles