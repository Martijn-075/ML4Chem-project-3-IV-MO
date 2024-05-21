import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import one_hot
from SmilesData import __special__

class SmilesLSTM(nn.Module):
    def __init__(self, vocsize, device, max_len=130, hidden_size=256, num_layers=2, num_units=3):
        super().__init__()
        self.device = device
        self.vocsize = vocsize
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_units = num_units
        self.lstm = nn.LSTM(vocsize, hidden_size, bidirectional=False, batch_first=True, num_layers=num_layers, dropout=0.5)
        self.dropout_03 = nn.Dropout(0.3)
        self.linear = nn.Linear(hidden_size, vocsize)
        
# h and c are initialized by pytorch and are then used in a batch config
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout_03(x)
        x = self.linear(x)
        return x

    def sample(self, batch_size=128, temp=1., h=None, c=None):
        
        bos_token = [k for k,v in __special__.items() if v == "<BOS>"][0]
        x = torch.LongTensor([bos_token]*batch_size)
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
    