import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import one_hot
import functools
from tqdm import tqdm
from rdkit import RDLogger
from rdkit import Chem

__special__ = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>"}
RDLogger.DisableLog('rdApp.*')

class SmilesProvider(torch.utils.data.DataLoader):
    def __init__(self, file, total=130):
        self.total = total
        self.smiles = open(file, 'r').read().split("\n")[:-1]
        tokens = functools.reduce(lambda acc,s: acc.union(set(s)), self.smiles ,set())
        self.vocsize = len(tokens) + len(__special__)
        self.index2token = dict(enumerate(tokens,start=3))
        self.index2token.update(__special__)
        self.token2index = {v:k for k,v in self.index2token.items()}
        self.ints = [torch.LongTensor([self.token2index[s] for s in line]) for line in tqdm(self.smiles,"Preparing of a dataset")]

    def decode(self,indexes):
        return "".join([self.index2token[index] for index in indexes if index not in __special__])

    def __getitem__(self,i):
        special_added = torch.cat((torch.LongTensor([self.token2index['<BOS>']])
                                   ,self.ints[i],torch.LongTensor([self.token2index['<EOS>']]),
                                   torch.LongTensor([self.token2index["<PAD>"]]*(self.total-len(self.ints[i])-2))),dim=0)
        return one_hot(special_added,self.vocsize).float(),special_added

    def __len__(self):
        return len(self.smiles)
    
class SimpleGRU(nn.Module):

    def __init__(self, vocsize, device, hidden_size=512, num_layers=3):
        super().__init__()
        self.device = device
        self.vocsize = vocsize
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(vocsize, hidden_size, bidirectional=False, batch_first=True, num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, vocsize)


    def forward(self, x):
        output = self.gru(x)[0]
        final = self.linear(output)
        return final

    def sample(self,batch_size=128,max_len=130):
        bos_token = [k for k,v in __special__.items() if v == "<BOS>"][0]
        x = torch.LongTensor([bos_token]*batch_size)
        h = torch.zeros((self.num_layers,batch_size,self.hidden_size)).to(self.device)
        accumulator = torch.zeros(batch_size,max_len)
        for i in range(max_len):
            x = one_hot(x, self.vocsize).float().unsqueeze(1).to(self.device)
            output,h = self.gru(x,h)
            next = F.softmax(self.linear(output).squeeze(1),dim=1)
            x = torch.multinomial(next,num_samples=1,replacement=True).squeeze(1)
            accumulator[:,i] = x
        return accumulator
    
def generate(file='genmodel.pt',batch_size=64):
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
        if Chem.MolFromSmiles(smiles) != None:
            correct += 1
            list_smiles.append(smiles)
# ======== end your code here ===================================
    # print ("% of correct molecules is {:4.2f}".format(correct/float(batch_size)*100))
    return list_smiles, correct/float(batch_size)*100

def train(file='250k.smi',batch_size=256,learning_rate=0.001,n_epochs=1,device='cuda'):
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
    dataset = SmilesProvider(file)
    model = SimpleGRU(dataset.vocsize,device=device).to(device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
# ======== TASK 1 start your code here =================================
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    model.train()
# ======== TASK 1 end your code here ===================================
    for epoch in range(1, n_epochs + 1):
        for iteration,(batch,target) in enumerate(tqdm(dataloader,'Training')):
            batch,target = batch.to(device),target.to(device)
            out = model(batch)
            out = out.transpose(2,1)
            loss = loss_function(out[:,:,:-1],target[:,1:])
# ======== TASK 2 start your code here =================================
            optimizer.zero_grad()
# ======== TASK 2 end your code here ===================================
            loss.backward()
# ======== TASK 2 start your code here =================================
            optimizer.step()
# ======== TASK 2 end your code here ===================================


    model.device = 'cpu'
    torch.save({'tokenizer':dataset.index2token,'model':model.cpu()},"genmodelnew.pt")

