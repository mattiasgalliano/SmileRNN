# optimize
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from rdkit.Chem.QED import default, qed
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Draw import MolToFile
from model import SmileRNN
import utils
import argparse
from torch.utils.tensorboard import SummaryWriter # for logging

# optimizes model with policy gradient rl, qed score reward, short smile, invalidity, non-unique penalties
def optimize(model, start_char_list, char2idx, idx2char, max_seq_len, device, optimizer, gamma, iters, save, weights_path, print_every, log_every, writer):
    unique_list = [] # tracks unique smile output samples

    for iter in range(iters):

        start_char = start_char_list[np.random.randint(0, len(start_char_list))]
        start_char_idx = char2idx[start_char]

        output_smile = None

        x = torch.tensor([start_char_idx]).to(device)
        h = model.init_hidden()

        output_smile = idx2char[start_char_idx]

        saved_log_probs = [] # tracks probs

        ### sample to build smile output
        for i in range(max_seq_len):

            out, h = model(x, h)

            # sample from softmax out
            out = out[0][0]

            out = torch.exp(out)

            probs = torch.distributions.Categorical(out)

            char_idx = probs.sample()
            saved_log_probs.append(probs.log_prob(char_idx).unsqueeze(0))

            char_idx = char_idx.item()

            if char_idx == 51: # EOS
                break
            else:
                char = idx2char[char_idx]
                output_smile += char

            x = model.embed(torch.tensor([char_idx]).to(device))
        
        ### calculate total reward based on sampled smile output
        total_reward = 0

        rewards = []

        valid_mol = MolFromSmiles(output_smile) # tests validity

        # result = None # for printing TODO optional?
        if valid_mol is None: # invalid
            total_reward = -5
            if iter % print_every == 0: result = "invalid"
        elif (len(output_smile) < 5): # too short
            total_reward = -10
            if iter % print_every == 0: result = "short"
        elif output_smile in unique_list: # not unique
            total_reward = -1
            if iter % print_every == 0: result = "not unique"
        else:
            q = qed(valid_mol)
            total_reward = q * 10
            unique_list.append(output_smile) # new unique
            if iter % print_every == 0: result = "qed " + str(q)
        
        ### spread total reward across each prediction
        if len(output_smile) == 1:
            rewards.append(-saved_log_probs[0] * (total_reward / len(output_smile) * gamma)) # TODO changed
        else:
            for i in range(len(output_smile)-1):
                rewards.append(-saved_log_probs[i] * (total_reward / len(output_smile) * gamma))

        optimizer.zero_grad()
        policy_loss = torch.cat(rewards).sum() #* -1 # TODO wrong dir???
        if iter % print_every == 0 and i != 0:
            tmp = policy_loss.item()
            print(iter, "result", result, "loss", tmp, "reward", total_reward)
        if iter % log_every == 0 and i != 0:
            tmp = policy_loss.item()
            writer.add_scalar("Loss/iter", tmp, i)
        policy_loss.backward()
        optimizer.step()
    
    if save:
        save_path = f'{weights_path}_optim_iter_{iters}'
        torch.save(model.state_dict(), save_path)


# runs optimization
def run():
    parser = argparse.ArgumentParser(description="parser for model training")
    parser.add_argument("-d", "--data", type=str, default='data/chembl_29_smiles.csv')
    parser.add_argument("-c", "--config", type=str, default="configs/base.json")
    parser.add_argument("-w", "--weights", type=str)
    parser.add_argument("-m", "--max_seq_len", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("-g", "--gamma", type=float, default=1)
    parser.add_argument("-i", "--iters", type=int, default=5000)
    parser.add_argument("-r", "--random_seed", type=int, default=123)
    parser.add_argument("-s", "--save", type=int, default=1)
    parser.add_argument("-p", "--print_every", type=int, default=50)
    parser.add_argument("-l", "--log_every", type=int, default=1)
    args = parser.parse_args()

    smiles = utils.read_data(args.data)

    config = utils.read_config(args.config)

    print("args:", vars(args), "\nconfig:", config)

    unique_chars = utils.get_unique_chars(smiles)

    char2idx = utils.get_char2idx(unique_chars) # define character, index mappings from smiles

    idx2char = {v: k for k, v in char2idx.items()}

    start_char_list = [x[0] for x in smiles] # possible starting chars

    n_unique_chars = len(unique_chars)
    max_seq_len = args.max_seq_len
    char_embedding_dim = config["char_embedding_dim"]
    hidden_layer_dim = config["hidden_layer_dim"]
    n_hidden_layers = config["n_hidden_layers"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    torch.manual_seed(args.random_seed)

    model = SmileRNN(
        n_unique_chars,
        char_embedding_dim,
        hidden_layer_dim,
        n_hidden_layers,
        device
    )

    model.load_state_dict(torch.load(args.weights))
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    writer = SummaryWriter() if args.log_every else None

    optimize(model, 
        start_char_list,
        char2idx, idx2char,
        max_seq_len,
        device, optimizer,
        args.gamma,
        args.iters,
        args.save,
        args.weights,
        args.print_every,
        args.log_every,
        writer
    )

    if args.log_every:
        writer.flush()
        writer.close()

run()