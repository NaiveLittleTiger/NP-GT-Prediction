import torch
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem

# fasta
def generate_ecfp(smile):
  mol=Chem.MolFromSmiles(smile)
  ecfp=AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024).ToBitString()
  return ecfp

def save_fasta(str):
    with open("./test.fasta",'w') as f:
        f.write(">")
        f.write(str)

def get_gt_representation(fasta_path):
    model, alphabet = pretrained.load_model_and_alphabet("esm1b_t33_650M_UR50S")
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print("Transferred model-pretrain to GPU")
    else:
        print("model-pretrain train on CPU")
    PATH = "./model-pretrain/model_ESM_binary_A100_epoch_1_similarity.pkl"
    model_dict = torch.load(PATH, map_location=torch.device('cpu'))
    model_dict_V2 = {k.split("model-pretrain.")[-1]: v for k, v in model_dict.items()}

    for key in ["module.fc1.weight", "module.fc1.bias", "module.fc2.weight", "module.fc2.bias", "module.fc3.weight",
                "module.fc3.bias"]:
        del model_dict_V2[key]
    model.load_state_dict(model_dict_V2, False)

    dataset = FastaBatchedDataset.from_file(fasta_path)
    batches = dataset.get_batch_indices(4096, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches)
    print(f"Read with {len(dataset)} sequences")

    # repr_layers_init=33
    repr_layers = [33]

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(labels)
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )

            output = model(toks, repr_layers=repr_layers, return_contacts=False)
            output = output["representations"][33]
            output = output[:, 0, :]
            outupt_array=output.numpy()
    return outupt_array
