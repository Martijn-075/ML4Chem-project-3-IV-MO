import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import one_hot
from SmilesData import __special__

class SmilesLSTM(nn.Module):
    def __init__(self, vocsize, device, max_len=130, hidden_size=256, num_layers=2):
        """
        The init of the SMILES generating LSTM model.
        
        Args:
            vocsize (int): Number of tokens. Number of unique caracters in smiles string + padding, begin of sequence and end of sequence
            device (str): device can be cuda or cpu
            max_len (int, optional): max length of sequence, excess is padded. Defaults to 130.
            hidden_size (int, optional): size of the hidden layers. Defaults to 256.
            num_layers (int, optional): number of recurcion. Defaults to 2.
        """
        super().__init__()
        self.device = device
        self.vocsize = vocsize
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # It should be noted that the dropout is only for every layer except the last so in this implementation after the first layer ther is a 0.5 dropout (as specified in LSTM)  and a drpout of 0.3 for wich a separated layer needed to be used.
        self.lstm = nn.LSTM(vocsize, hidden_size, bidirectional=False, batch_first=True, num_layers=num_layers, dropout=0.5)
        self.dropout_03 = nn.Dropout(0.3)
        self.linear = nn.Linear(hidden_size, vocsize)
        

    def forward(self, x):
        """ 
        The forward pass of the model. this function is only used during the training. For the sampling see the function below. It should be noted that the h and c hidden weights are init by torch and used for the rest of the batch.
        
        Args:
            x (torch.tensor(batch size, sequence length=max_len, input size=vocsize)): input encoded in one hot format.

        Returns:
            x (torch.tensor(batch size, sequence length=max_len, input size=vocsize)): gives a tnesor with the output of the model. note this doesnt use a softmax functions as this is integrated in the Crossentropy loss function used during training
        """
        x, _ = self.lstm(x)
        x = self.dropout_03(x)
        x = self.linear(x)
        return x

    def sample(self, batch_size=128, temp=1., h=None, c=None):
        """
        The main sample function. This function is called from all generation functions. givven a batch size it returns that many SMILES. SMILES are not checked for valididty.
        
        Args:
            batch_size (int, optional): the number of smiles to be sampled. Defaults to 128.
            temp (float, optional): theee temprature to be sampled at. This chnages the 'creativity' of the model. Defaults to 1..
            h (torch.tensor(number of layers = num_layers, batch size = batch_size, hidden size = hidden_size), optional): the hidden short term memeory of the model only used within a sequence. Defaults to None.
            c (torch.tensor(number of layers = num_layers, batch size = batch_size, hidden size = hidden_size), optional): the hidden long term memory of the model only used within a sequence. Defaults to None.

        Returns:
            accumulator torch.tensor(batch size = batch_size, sequence length = max_len): the sampled smiles index encoded.
            h (torch.tensor(number of layers = num_layers, batch size = batch_size, hidden size = hidden_size), optional): the hidden short term memeory of the model only used within a sequence.
            c (torch.tensor(number of layers = num_layers, batch size = batch_size, hidden size = hidden_size), optional): the hidden long term memory of the model only used within a sequence.
        """
        bos_token = [k for k,v in __special__.items() if v == "<BOS>"][0]
        x = torch.LongTensor([bos_token]*batch_size).to(self.device)
        if h == None:
            h = torch.zeros((self.num_layers, batch_size, self.hidden_size)).to(self.device)
        if c == None:
            c = torch.zeros((self.num_layers, batch_size, self.hidden_size)).to(self.device)
        accumulator = torch.zeros(batch_size, self.max_len)
        for i in range(self.max_len):
            x = one_hot(x, self.vocsize).float().unsqueeze(1).to(self.device)
            x, (h, c) = self.lstm(x, (h, c))
            x = self.dropout_03(x)
            x = self.linear(x).squeeze(1)
            # gives temp control over output just like boltzman
            x[1] = x[1] * (1./temp)
            x = F.softmax(x, dim=1)
            x = torch.multinomial(x, num_samples=1,replacement=True).squeeze(1)
            accumulator[:,i] = x
        return accumulator, h, c
    