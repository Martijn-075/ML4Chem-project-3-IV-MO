import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from SmilesData import SmilesProvider, logger
from SmilesModel import SmilesLSTM
from SmilesGenerate import generate_current_model

def train(train_file='smiles_CHEMBL_22', save_file="SmilesLSTM_CHEMBL_22_22_epochs.pt", batch_size=1536, learning_rate=0.001, n_epochs=30, device='cuda'):
    """
    The train function of the SmilesLSTM model. It uses the ADAM optimizer and the crossentropyloss loss function. For the project the smiles_CHEMBL_22 file conains 542 674 (canonical) SMILES from the CHEMBL 22 database. The database was preprocesseed by only keeping SMILES that had an nM (nano molar) ICE/EC50, kb, ki, kd entery, was between 34-74 caracters and dint contain salts.

    Args:
        train_file (str, optional): The path the the txt file conaining the (canonical) SMILES. The SMILES should be in a single columm with no headers.  Defaults to 'smiles_CHEMBL_22'.
        save_file (str, optional): The name of the save file of the model. the name is alos used for the logger and the epoch data logger. Defaults to "SmilesLSTM_CHEMBL_22_22_epochs.pt".
        batch_size (int, optional): The batch size of the itr in each epoch. No default is set to use 5.5 GB of VRAM using CUDA but this is highly dependend on hardware. Defaults to 1536.
        learning_rate (float, optional): The (start) learning rate. this vlaue is passed to the ADAM optimizer and so the lr is adjusted troughout the training as by ADAM. The default of 0.001 is a good trade of between stability (higher) and actualy learning (lower). Defaults to 0.001.
        n_epochs (int, optional): Number of epochs (times all data is seen by model). for the project 50 epochs were used but after analyzing the loss and the % valid SMILES the optimal number is somewhere around 30 epochs. Defaults to 3.
        device (str, optional): The device that should be used CPU or CUDA (GPU). Defaults to 'cuda'.
        
    Returns:
        models/{save_file} (save file (torch)): It creates a (torch) save file containing the model and the tokinizer (index2token). it is saved to the models folder.
        models/{save_file}_logs (save file (txt)): The file that contains the logs of the training containing the loss and % valid SMILES throughout the training
        models/{save_file}_epoch_data (save file (txt)): The file that contaisn the same data as the logger only orginized in a table so it can be read in for easy plotting.
    """
    device = device if torch.cuda.is_available() else 'cpu'
    dataset = SmilesProvider(train_file)
    model = SmilesLSTM(dataset.vocsize, device=device).to(device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True) # type: ignore

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    model.train()
    
    # Array to logg the % valid smiles and the loss
    epoch_data = np.zeros((n_epochs, 2))
    
    # The actual training loop
    for epoch in range(1, n_epochs + 1):
        total_loss = 0.
        # The loop over the itr with a size equal to the batch size.
        for iteration, (batch, target) in enumerate(tqdm(dataloader,'Training')):
            batch, target = batch.to(device), target.to(device)
            out = model(batch)
            
            # The out tensor is (batch size, sequence length, unqiue tokens size) but the loss function expceptcts (batch size, unqiue tokens size, sequence length)
            out = out.transpose(2,1)

            #  the loss is calcualted from the 2 caracter of the target and until the one before last caracter of the prediction (out). This ensures that the loss is calcualted for the actual predicted caracter.
            loss = loss_function(out[:,:,:-1], target[:,1:])
            total_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Check how many valid SMILES are generated with the current model. This is done every epoch
        _, p_validSmiles, _ = generate_current_model(model, dataset.index2token, batch_size=1000, temp=1.)
        model.train()
        
        # Printing and logging a message conaining information such as the loss and % valid SMILES
        message = f"Epoch {epoch} of {n_epochs} done, {p_validSmiles}% valid smiles generated, epoch loss: {total_loss}"
        print(message)
        logger(message, f"models/{save_file}_logs")
        
        # Saving the epoch data, such as the % valid SMILES and the epoch loss
        epoch_data[epoch-1, :] = np.array([float(total_loss), float(p_validSmiles)])
    
    #  Svaving of the trained model. it should be noted that the device of the model is set to cpu so it is compatible with non cuda compiled torch
    model.device = 'cpu'
    torch.save({'tokenizer':dataset.index2token,'model':model.cpu()}, f"models/{save_file}")
    
    # The epoch data is saved to file 
    np.savetxt(f"models/{save_file}_epoch_data", epoch_data)
    print("Training done!")
