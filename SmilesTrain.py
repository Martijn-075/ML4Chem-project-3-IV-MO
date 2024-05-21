import torch
import torch.nn as nn
from tqdm import tqdm
from SmilesData import SmilesProvider, logger
from SmilesModel import SmilesLSTM
from SmilesGenerate import generate_current_model

def train(train_file='smiles_CHEMBL_22', save_file="testCHEMBL22.pt", batch_size=1536,learning_rate=0.001, n_epochs=10, device='cuda'):
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
# ======== TASK 1 start your code here =================================
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    model.train()
    # torch.autograd.set_detect_anomaly(True)
# ======== TASK 1 end your code here ===================================
    for epoch in range(1, n_epochs + 1):
        total_loss = 0.
        for iteration,(batch, target) in enumerate(tqdm(dataloader,'Training')):
            batch, target = batch.to(device), target.to(device)
            out = model(batch)
            out = out.transpose(2,1)
            # what does the slcing do it removes the last from prediction and the first carater from the target
            loss = loss_function(out[:,:,:-1], target[:,1:])
            total_loss += loss
            # print('loss', loss)
            # print('batch', batch)
            # print("prediction", out.size())
            # print("prediction", out[0,:,:])

            # print("target", target.size())
            # print("target", target[0,:])

# ======== TASK 2 start your code here =================================
            optimizer.zero_grad()
# ======== TASK 2 end your code here ===================================
            loss.backward()
# ======== TASK 2 start your code here =================================
            optimizer.step()
# ======== TASK 2 end your code here ===================================

        _, p_validSmiles = generate_current_model(model, dataset.index2token, batch_size=100, temp=1.)
        model.train()
        message = f"Epoch {epoch} of {n_epochs} done, {p_validSmiles}% valid smiles generated, epoch loss: {total_loss}"
        print(message)
        logger(message, f"models/{save_file}_logs")
        
    model.device = 'cpu'
    torch.save({'tokenizer':dataset.index2token,'model':model.cpu()}, f"models/{save_file}")
    print("Training done!")
train() 