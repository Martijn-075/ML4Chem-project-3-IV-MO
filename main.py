"""

Main module to showcase the project and generate SMILES 1) from the base model, 2) with RL training on a logp target and 3) with RL training with Curcumin as the target moleceul

"""

from SmilesGenerate import generate
from SmilesRL import comp_logp, comp_MACCS
from SmilesData import save_smiles_txt

print("Must have CUDA available otherwise options 2 and 3 will be very slow. ")
while True:
    print("Enter 1 to generate 1 000 SMILES from the base trained model")
    print("Enter 2 to generate 1 000 SMILES wtih the RL model with a target of logp=6")
    print("Enter 3 to generate 1 000 SMILES wtih the RL model with Curcumin as a target molecuel")

    
    try:
        ans = int(input("Please enter response: "))
    except:
        pass
    
    if ans == 1:
        smiles, pc, _, _= generate(file="SmilesLSTM_CHEMBL_22_50_epoch.pt", batch_size=1000, temp=1.)
        save_smiles_txt(smiles, "gen_SMILES.txt")
        print(f"Generated {pc}% valid SMILES")
        print("saved SMILES as gen_SMILES.txt")
        break
    elif ans == 2:
        smiles, pc = comp_logp(file="SmilesLSTM_CHEMBL_22_50_epoch.pt", target=6.)
        save_smiles_txt(smiles, "gen_SMILES.txt")
        print(f"Generated {pc}% valid SMILES")
        print("saved SMILES as gen_SMILES.txt")
        break
    elif ans == 3:
        smiles, pc =comp_MACCS(file="SmilesLSTM_CHEMBL_22_50_epoch.pt", target=r"O=C(\C=C\c1ccc(O)c(OC)c1)CC(=O)\C=C\c2cc(OC)c(O)cc2")
        save_smiles_txt(smiles, "gen_SMILES.txt")
        print(f"Generated {pc}% valid SMILES")
        print("saved SMILES as gen_SMILES.txt")
        break
    else:
        print("Pleas enter a valid number")
