import pandas as pd
# data = pd.read_table('chembl_22_chemreps_1.txt')
# # print(data)
# smiles = data["canonical_smiles"]
# print(smiles)
# smiles.to_csv("allsmiles.txt", sep='\t', index=False)

data = pd.read_table('allsmiles.txt')

data = data[-data["canonical_smiles"].str.contains("[-+.]", regex=True)]
data = data[(data["canonical_smiles"].str.len() >= 34)]
data = data[(data["canonical_smiles"].str.len() <= 74)]
data.to_csv("smiles.txt", sep='\t', index=False)
print
