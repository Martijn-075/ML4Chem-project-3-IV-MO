from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdAbbreviations # type: ignore

def Visualize_Molecules_from_Smiles(smiles):

    molecules = [Chem.MolFromSmiles(smile) for smile in smiles] # type: ignore

    IPythonConsole.UninstallIPythonRenderer()
    img = Draw.MolsToGridImage(molecules, molsPerRow=8)
    img.save('molgrid.png')