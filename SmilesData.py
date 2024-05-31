"""

This module conainst the dataloader class used to load SMILES from file and prepare them for traning.

"""
    
import torch
from torch.nn.functional import one_hot
import functools
import datetime
import numpy as np
from tqdm import tqdm
from rdkit import RDLogger

# Special tokens such as the begin of sequnece (BOS), end of sequence (EOS) and padding token (PAD).
__special__ = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>"}
RDLogger.DisableLog('rdApp.*')


class SmilesProvider(torch.utils.data.DataLoader): # type: ignore
    """
    This is the datalopader class for loading in SMILES from a file and prepair them for the model. This is done by tokinizing the SMILES (wich is later encode in one hot). The inex2token is save with the model and can be used after training and saving.
    """
    def __init__(self, file, total=130):
        """
        This is the routine to init the dataset before traning. it creates a list of unique tokens wich will be com the VOCsize of the model. It converts all SMILES to index.

        Args:
            file (str): The file name with the SMILES for training. The SMILES should be in a single column without index and header.
            total (int, optional): The maximium lenght of the SMILES to be read in and after traning generated. Defaults to 130.
        """
        self.total = total
        self.smiles = open(file, 'r').read().split("\n")[:-1]
        tokens = functools.reduce(lambda acc,s: acc.union(set(s)), self.smiles ,set())
        self.vocsize = len(tokens) + len(__special__)
        self.index2token = dict(enumerate(tokens,start=3))
        self.index2token.update(__special__)
        self.token2index = {v:k for k,v in self.index2token.items()}
        self.ints = [torch.LongTensor([self.token2index[s] for s in line]) for line in tqdm(self.smiles,"Preparing of a dataset")]


    def decode(self, indexes):
        """
        Converts an index tensor to a SMILES string.

        Args:
            indexes (torch.tensor): The tensor with the indexes of the SMILES tokens.

        Returns:
            str: The SMILES string.
        """
        return "".join([self.index2token[index] for index in indexes if index not in __special__])


    def __getitem__(self, i):
        """
        Routine to get a sepcific SMILES from the dataset with inex i. It returns the SMILES already in one hot encoding

        Args:
            i (_type_): The inex of the SMILES from the dataset.

        Returns:
            torhc.tensor: The one hot encoded SMILES
        """
        special_added = torch.cat((torch.LongTensor([self.token2index['<BOS>']])
                                   ,self.ints[i],torch.LongTensor([self.token2index['<EOS>']]),
                                   torch.LongTensor([self.token2index["<PAD>"]]*(self.total-len(self.ints[i])-2))),dim=0)
        return one_hot(special_added, self.vocsize).float(),special_added


    def __len__(self):
        """
        Returns the lenght of the dataset / the number of SMILES in the training file.

        Returns:
            int: The lenght of the dataset
        """
        return len(self.smiles)
    
    
def logger(message, file_path):
    """
    A simple logger function that takes a meassage and puts the date and time infront of it.

    Args:
        message (str): The message that should be logged
        file_path (str): The file path ot a file (doesnt have to exist)
    """
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(file_path, 'a') as file:
        file.write(f"{current_time} - {message} \n")
        
def save_smiles_txt(smiles, file='SMILES.txt'):
    """
    Saves a list of SMILES to a txt file

    Args:
        smiles (list): The list of SMILES to be saved
        file (str, optional): The file name of the to be saved file. Defaults to 'SMILES.txt'.
    """
    array = np.array(smiles)
    np.savetxt(file, array, fmt='%s')