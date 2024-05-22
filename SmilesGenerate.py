import torch
import torch.nn.functional as F
import torch.nn as nn
from rdkit import Chem
import numpy as np
from rdkit.Chem import Descriptors
from SmilesData import __special__, logger

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
    res, _, _ = model.sample(batch_size, temp=temp)
    correct = 0
    list_smiles = []
    indecies = []
    for i in range(res.size(0)):
        smiles = "".join([tokenizer[index] for index in res[i].tolist() if index not in __special__])
        # print(smiles)

        if Chem.MolFromSmiles(smiles) != None: # type: ignore
            correct += 1
            list_smiles.append(smiles)
            indecies.append(i)

    indexed_smiles = res[indecies]
    p_valid_smiles = correct/float(batch_size)*100
    
    return list_smiles, p_valid_smiles, indexed_smiles

# Doesnt work gives empty strings. posible that because normaly the string should be filled with filler tokens to 130 that it just continius when h and c are used
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

def bias_training_generation(file='SmilesLSTM30ep.pt',batch_size=10, num_loop=5, temp=1.):
    file = f"models/{file}"
    box = torch.load(file)
    model, tokenizer = box['model'], box['tokenizer']
    
    target = 1.
    for i in range(1, num_loop+1):
        smiles, cp, indexed_smiles = generate_current_model(model, tokenizer, batch_size=100, temp=1.)
        
        property_score = np.zeros(len(smiles))
        for j, smile in enumerate(smiles):
            mol = Chem.MolFromSmiles(smile) # type: ignore
            property_score[j] = Descriptors.MolLogP(mol) # type: ignore
                
        rewards = np.absolute((property_score-target))
        rewards = rewards / rewards.max()
        
        bias_train(model, indexed_smiles, rewards, lr=0.001, it=i, maxit=num_loop)
    return smiles
        
def bias_train(model, index_smiles, rewards, lr=0.001, n_epochs=10, device="cuda", it=1, maxit=10):
    device = device if torch.cuda.is_available() else 'cpu'
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss(reduction="None")
    model.train()

    batch = F.one_hot(index_smiles, model.vocsize)
    
    for epoch in range(1, n_epochs + 1):
        random_index = torch.randperm(index_smiles.size(0))
        index_smiles = index_smiles[random_index]
        batch = batch[random_index]
        rewards = rewards[random_index]
        
        total_loss = 0.
        
        out = model(batch)
        out = out.transpose(2,1)
        
        loss = loss_function(out[:,:,:-1], index_smiles[:,1:])
        loss = loss * torch.tensor(rewards).unsqueeze(1).to(device)
        loss.mean()
        total_loss += loss

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        message = f"Epoch {epoch} of {n_epochs} of the bias traning it {it} of {maxit} done of the bias training epoch loss: {total_loss}"
        print(message)
        logger(message, "models/bias_training_logs")     
            
        