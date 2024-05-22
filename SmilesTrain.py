import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from SmilesData import SmilesProvider, logger
from SmilesModel import SmilesLSTM
from SmilesGenerate import generate_current_model

def train(train_file='smiles_CHEMBL_22', save_file="SmilesLSTM_CHEMBL_22_22_epochs.pt", batch_size=1536,learning_rate=0.001, n_epochs=22, device='cuda'):
    """
    This is the entrypoint for training of the RNN
    :param file: A file with molecules in SMILES notation
    :param batch_size: A batch size for training
    :param learning_rate: A learning rate of the optimizer
    :param n_epochs: A number of epochs
    :param device: "cuda" for GPU training, "cpu" for training on CPU, if there are no CUDA on a computer it uses CPU only
    :return: None. It saves the model to "genmodel.pt" file
    """
    device = device if torch.cuda.is_available() else 'cpu'
    dataset = SmilesProvider(train_file)
    model = SmilesLSTM(dataset.vocsize, device=device).to(device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True) # type: ignore

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    model.train()
    
    epoch_data = np.zeros((n_epochs, 2))
    for epoch in range(1, n_epochs + 1):
        total_loss = 0.
        for iteration,(batch, target) in enumerate(tqdm(dataloader,'Training')):
            batch, target = batch.to(device), target.to(device)
            out = model(batch)
            out = out.transpose(2,1)

            loss = loss_function(out[:,:,:-1], target[:,1:])
            total_loss += loss

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        _, p_validSmiles, _ = generate_current_model(model, dataset.index2token, batch_size=1000, temp=1.)
        model.train()
        message = f"Epoch {epoch} of {n_epochs} done, {p_validSmiles}% valid smiles generated, epoch loss: {total_loss}"
        print(message)
        epoch_data[epoch-1, :] = np.array([float(total_loss), float(p_validSmiles)])
        logger(message, f"models/{save_file}_logs")
        
    model.device = 'cpu'
    torch.save({'tokenizer':dataset.index2token,'model':model.cpu()}, f"models/{save_file}")
    np.savetxt(f"models/{save_file}_epoch_data", epoch_data)
    print("Training done!")
train()
