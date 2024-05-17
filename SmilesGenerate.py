import torch
from rdkit import Chem
from SmilesData import __special__

def generate(file='SmilesLSTM30ep.pt',batch_size=64):
    """
    This is the entrypoint for the generator of SMILES
    :param file: A file with pretrained model
    :param batch_size: The number of compounds to generate
    :return: None. It prints a list of generated compounds to stdout
    """
    box= torch.load(file)
    model,tokenizer = box['model'],box['tokenizer']
    model.eval()
    res = model.sample(batch_size)
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
    print (f"{p_valid_smiles} % are valid smiles")
    return list_smiles, p_valid_smiles
