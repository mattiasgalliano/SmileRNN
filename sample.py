import argparse # for cli
import numpy as np
import torch
from rdkit.Chem import MolFromSmiles # for cheminformatics
from rdkit.Chem.Draw import MolToFile
from rdkit.Chem.QED import default, qed
import matplotlib.pyplot as plt # for visualization
from model import SmileRNN
import utils
import os

# samples moleclue from trained model
def sample(model, char2idx, idx2char, start_char_list, max_seq_len, device):
    start_char = start_char_list[np.random.randint(0, len(start_char_list))]
    start_char_idx = char2idx[start_char]

    output_smile = None

    with torch.no_grad():
        x = torch.tensor([start_char_idx]).to(device)

        h = model.init_hidden()

        output_smile = idx2char[start_char_idx]

        o = None

        for i in range(max_seq_len):

            o, h = model(x, h)

            o = o[0][0] # TODO sample from softmax out

            o = torch.exp(o)
            
            probs = torch.distributions.Categorical(o)

            char_idx = probs.sample()

            char_idx = char_idx.item()

            if char_idx == 51:
                break
            else:
                char = idx2char[char_idx]
                output_smile += char
            
            x = model.embed(torch.tensor([char_idx]).to(device))

    return output_smile


# checks molecule bracket, paren balance
def check_balance(sample):
    open_list = ['[', '(']
    close_list = [']', ')']

    stack = []
    for e in sample:
        if e in open_list:
            stack.append(e)
        elif e in close_list:
            pos = close_list.index(e)
            if len(stack) > 0 and (open_list[pos] == stack[len(stack) - 1]):
                stack.pop()
            else:
                return False
    if len(stack) == 0:
        return True
    else:
        return False


# checks if valid molecule
def check_mol(sample):
    try:
        mol= MolFromSmiles(sample)
        qed = default(mol)
        return True
    except Exception as e:
        return False


# samples until balanced, valid mol
def rejection_sample(model, char2idx, idx2char, start_char_list, max_seq_len, device):
    s = sample(model, char2idx, idx2char, start_char_list, max_seq_len, device)
    while check_balance(s) == False or check_mol(s) == False:
        s = sample(model, char2idx, idx2char, start_char_list, max_seq_len, device)
    return s


# runs sampling
def run():
    parser = argparse.ArgumentParser(description="parser for model training")
    parser.add_argument("-d", "--data", type=str, default='data/chembl_29_smiles.csv')
    parser.add_argument("-c", "--config", type=str, default="configs/base.json")
    parser.add_argument("-w", "--weights", type=str)
    parser.add_argument("-m", "--max_seq_len", type=int, default=30)
    parser.add_argument("-n", "--num", type=int, default=100)
    parser.add_argument("-r", '--rej', type=bool, default=1)
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("-s", "--save", type=int, default=1)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("-v", "--visualize", type=int, default=1)
    args = parser.parse_args()

    smiles = utils.read_data(args.data) # read smiles data

    config = utils.read_config(args.config) # read model config json

    print("args:", vars(args), "\nconfig:", config)

    unique_chars = utils.get_unique_chars(smiles) # get unique characters from smiles

    char2idx = utils.get_char2idx(unique_chars) # define character, index mappings from smiles

    idx2char = {v: k for k, v in char2idx.items()}

    start_char_list = [x[0] for x in smiles] # possible starting chars

    n_unique_chars = len(unique_chars)
    max_seq_len = args.max_seq_len
    char_embedding_dim = config["char_embedding_dim"] # TODO if plateau change back to 52
    hidden_layer_dim = config["hidden_layer_dim"]
    n_hidden_layers = config["n_hidden_layers"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    model = SmileRNN(
        n_unique_chars,
        char_embedding_dim,
        hidden_layer_dim,
        n_hidden_layers,
        device
    )

    model.load_state_dict(torch.load(args.weights))
    
    print("sampling...")
    samples = []
    for i in range(args.num):
        if args.rej:
            s = rejection_sample(model, char2idx, idx2char, start_char_list, max_seq_len, device)
            samples.append(s)
            if i % args.print_every == 0: print(i, s)
        else:
            s = sample(model, char2idx, idx2char, start_char_list, max_seq_len, device)
            samples.append(s)
            if i % args.print_every == 0: print(i, s)
    
    if args.save:
        for i, s in enumerate(samples):
            if not os.path.exists('output'): os.mkdir('output')
            mol = MolFromSmiles(s)
            qed = default(mol)
            MolToFile(mol, f"output/{i}.png")
    
    count = 0
    for s in set(samples):
        if s in smiles: count += 1
    print("uniqueness", abs(1 - (count / len(set(samples)))))
    
    if args.visualize:
        QEDs = []

        for i, s in enumerate(samples):
            mol = MolFromSmiles(s)
            QED = default(mol)
            QEDs.append(QED)

        QEDs = np.array(QEDs)
        mean = np.mean(QEDs)

        plt.hist(QEDs, bins=int(len(QEDs)/10), density=True)
        plt.title(f"Mean QED: {mean:.2f}")

        plt.show()

run()