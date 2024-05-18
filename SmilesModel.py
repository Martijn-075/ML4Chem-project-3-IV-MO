import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import one_hot
from SmilesData import __special__

class SmilesLSTM(nn.Module):
    def __init__(self, vocsize, device, max_len=130, hidden_size=256, num_layers=1, num_units=3):
        super().__init__()
        self.device = device
        self.vocsize = vocsize
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_units = num_units
        self.lstm_1 = nn.LSTM(vocsize, hidden_size, bidirectional=False, batch_first=True, num_layers=num_layers)
        self.dropout_05 = nn.Dropout(0.5)
        self.lstm_2 = nn.LSTM(hidden_size, hidden_size, bidirectional=False, batch_first=True, num_layers=num_layers)
        self.dropout_03 = nn.Dropout(0.3)
        self.linear = nn.Linear(hidden_size, vocsize)

# This doesnt do anaything it just selctects the first caracter and uses that. Also not used?
    def forward(self, x):
        # if hc == None:
        #     h_1 = torch.zeros((self.num_layers, x.size(0), self.hidden_size)).to(self.device)
        #     c_1 = torch.zeros((self.num_layers, x.size(0), self.hidden_size)).to(self.device)
        #     h_2 = torch.zeros((self.num_layers, x.size(0), self.hidden_size)).to(self.device)
        #     c_2 = torch.zeros((self.num_layers, x.size(0), self.hidden_size)).to(self.device)
        # else:
        #     h_1, c_1, h_2, c_2 = hc

        # accumulator = torch.zeros(x.size(0), self.max_len)
        # for i in range(self.max_len):
        x, _ = self.lstm_1(x)
        x = self.dropout_05(x)
        x, _ = self.lstm_2(x)
        x = self.dropout_03(x)
        x = self.linear(x)
        # print("before", x.size())
        # ! i think softmax should be in the model to restric the output to be between 0,1 but gives 0.0 valid smiles for 5 ep. Model learns to be between 0,1?
        # ! x = F.softmax(x, dim=1)
        # print('softmax', x.size())
        # x = torch.multinomial(x,num_samples=1,replacement=True).squeeze(1)
        # print("multinomial", x.size())
        # accumulator[:,i] = x

        return x

    def sample(self, batch_size=128, temp=1.):
        bos_token = [k for k,v in __special__.items() if v == "<BOS>"][0]
        x = torch.LongTensor([bos_token]*batch_size)
        h_1 = torch.zeros((self.num_layers, batch_size, self.hidden_size)).to(self.device)
        c_1 = torch.zeros((self.num_layers, batch_size, self.hidden_size)).to(self.device)
        h_2 = torch.zeros((self.num_layers, batch_size, self.hidden_size)).to(self.device)
        c_2 = torch.zeros((self.num_layers, batch_size, self.hidden_size)).to(self.device)
        accumulator = torch.zeros(batch_size, self.max_len)
        for i in range(self.max_len):

            x = one_hot(x, self.vocsize).float().unsqueeze(1).to(self.device)
            x, (h_1, c_1) = self.lstm_1(x, (h_1, c_1))
            x = self.dropout_05(x)
            x, (h_2, c_2) = self.lstm_2(x, (h_2, c_2))
            x = self.dropout_03(x)
            x = self.linear(x).squeeze(1)
            # gives temp control over output just like boltzman
            x[1] = x[1] * (1./temp)
            x = F.softmax(x, dim=1)
            x = torch.multinomial(x, num_samples=1,replacement=True).squeeze(1)
            accumulator[:,i] = x
        return accumulator