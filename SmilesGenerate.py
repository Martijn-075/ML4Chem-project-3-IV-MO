import torch
import torch.nn.functional as F
import torch.nn as nn
from rdkit import Chem
import numpy as np
from tqdm import tqdm
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

def generate_current_model(model, tokenizer, batch_size=100, temp=1., device="cuda"):
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

def bias_training_generation(file='SmilesLSTM30ep.pt', logp_target= 6., batch_size=1000, num_loop=5, temp=1., device="cuda"):
    file = f"models/{file}"
    box = torch.load(file)
    model, tokenizer = box['model'], box['tokenizer']
    model.device = device
    model.to(device)


    for i in tqdm(range(1, num_loop+1), "Bias training"):
        smiles, cp, indexed_smiles = generate_current_model(model, tokenizer, batch_size=batch_size, temp=1., device=device)
        indexed_smiles.to(device)
        
        property_score = np.zeros(len(smiles))
        for j, smile in enumerate(smiles):
            mol = Chem.MolFromSmiles(smile) # type: ignore
            property_score[j] = Descriptors.MolLogP(mol) # type: ignore
                
        rewards = (logp_target - property_score)
        rewards = (rewards - rewards.min()) / (rewards.max() - rewards.min())
        rewards = torch.from_numpy(rewards)
        
        total_loss = bias_train(model, indexed_smiles, rewards, lr=0.001, device=device)
        message = f"Bias traning it {i} of {num_loop} done of the bias training epoch loss: {total_loss}, reward sum {rewards.sum()}"
        print(message)
        logger(message, "models/bias_training_logs") 
        
    smiles, cp, indexed_smiles = generate_current_model(model, tokenizer, batch_size=batch_size, temp=1.)
    return smiles
        
def bias_train(model, index_smiles, rewards, lr=0.001, n_epochs=10, device="cuda"):
    device = device if torch.cuda.is_available() else 'cpu'
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss(reduction="none")
    model.train()
    index_smiles = index_smiles.to(torch.int64).to(device)

    batch = F.one_hot(index_smiles, model.vocsize).float().to(device)
    total_loss = 0.
    
    for epoch in range(1, n_epochs + 1):
        random_index = torch.randperm(index_smiles.size(0))
        index_smiles = index_smiles[random_index]
        batch = batch[random_index]
        rewards = rewards[random_index]
        
        out = model(batch)
        out = out.transpose(2,1)
        loss = loss_function(out[:,:,:-1], index_smiles[:,1:])
        loss = loss * rewards.unsqueeze(1).to(device)
        loss = loss.mean()
        total_loss += loss

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
    return total_loss


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

   
            
        