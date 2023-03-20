'''
practicing char-level rnn for SMILES strings from chembl
'''
import torch
import torch.nn as nn
from torch import optim
import argparse # for cli
import random # for train data shuffle
import os

from torch.utils.tensorboard import SummaryWriter # for logging

from model import SmileRNN
import utils

# prepares inputs, targets tensors for training
def prepare_inputs_targets(mols, char2idx, n_unique_chars, max_seq_len, device):
    inputs = []
    targets = []

    for mol in mols:
        chars = []

        for i, char in enumerate(mol):
            if i == max_seq_len:
                break
            else:
                idx = char2idx[char]
                chars.append(idx)

        chars.append(n_unique_chars) # add EOS token

        while len(chars) <= max_seq_len: # pad with 0s
            chars.append(0)

        input_chars = chars[:-1]
        inputs.append(input_chars)

        target_chars = chars[1:]
        targets.append(target_chars)

    inputs = torch.tensor(inputs).to(device)
    targets = torch.tensor(targets).to(device)

    return inputs, targets


# trains model
def train(model, mols, char2idx, n_unique_chars, max_seq_len, device, criterion, optimizer, epochs, save, print_every, log_every, writer):
    for e in range(epochs):
        random.shuffle(mols)

        X, Y = prepare_inputs_targets(mols, char2idx, n_unique_chars, max_seq_len, device)

        total_loss = 0

        for i in range(len(X)):
            x, y = X[i], Y[i]

            h = model.init_hidden() # TODO not?

            model.zero_grad()

            loss = 0

            for j in range(len(x)):
                x_j = x[j] # consolidate steps?
                x_j = x_j.unsqueeze(0)
                y_j = y[j]
                y_j = y_j.unsqueeze(0)

                out, h = model(x_j, h)
                out = out.squeeze(0)

                l = criterion(out, y_j)

                loss += l

            total_loss += loss.item()

            if i % print_every == 0 and i != 0:
                avgloss = total_loss / i
                print("e", e, "i", i, "avg loss was", avgloss, "char avg", avgloss / out.size(1))
            
            if i % log_every == 0 and i != 0:
                avgloss = total_loss / i
                writer.add_scalar("Loss/iter", avgloss, i)

            loss.backward()

            optimizer.step()
        
        if save:
            if not os.path.exists('weights'): os.mkdir('weights')
            checkpoint = f'weights/train_e_{e}'
            torch.save(model.state_dict(), checkpoint)


# runs training
def run():
    parser = argparse.ArgumentParser(description="parser for model training")
    parser.add_argument("-d", "--data", type=str, default='data/chembl_29_smiles.csv')
    parser.add_argument("-c", "--config", type=str, default="configs/base.json")
    parser.add_argument("-m", "--max_seq_len", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("-e", "--epochs", type=int, default=1)
    parser.add_argument("-r", "--random_seed", type=int, default=123)
    parser.add_argument("-s", "--save", type=int, default=1)
    parser.add_argument("-p", "--print_every", type=int, default=100)
    parser.add_argument("-l", "--log_every", type=int, default=1000)
    args = parser.parse_args()

    smiles = utils.read_data(args.data) # read train input smiles data

    config = utils.read_config(args.config) # read model config json

    print("args:", vars(args), "\nconfig:", config)

    unique_chars = utils.get_unique_chars(smiles) # get unique characters from smiles

    char2idx = utils.get_char2idx(unique_chars) # get character, index mappings from smiles

    n_unique_chars = len(unique_chars)
    max_seq_len = args.max_seq_len
    char_embedding_dim = config["char_embedding_dim"] # TODO if plateau change back to 52
    hidden_layer_dim = config["hidden_layer_dim"]
    n_hidden_layers = config["n_hidden_layers"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)
    
    torch.manual_seed(args.random_seed)

    smile_rnn = SmileRNN(
        n_unique_chars,
        char_embedding_dim,
        hidden_layer_dim,
        n_hidden_layers,
        device
    )

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(smile_rnn.parameters(), lr=args.lr)

    writer = SummaryWriter() if args.log_every else None

    print("training...")
    train(smile_rnn,
        smiles,
        char2idx,
        n_unique_chars,
        max_seq_len,
        device,
        criterion,
        optimizer,
        args.epochs,
        args.save,
        args.print_every,
        args.log_every,
        writer      
    )

    if args.log_every:
        writer.flush()
        writer.close()

run()