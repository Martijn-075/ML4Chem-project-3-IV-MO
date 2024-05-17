import torch
from torch.nn.functional import one_hot
import functools
from tqdm import tqdm
from rdkit import RDLogger

__special__ = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>"}
RDLogger.DisableLog('rdApp.*')

class SmilesProvider(torch.utils.data.DataLoader): # type: ignore
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