import json

# reads model config as json from filepath, returns json obj
def read_config(filepath):

    with open(filepath, 'r') as f:
        config = json.load(f)

    return config


# reads, parses train input smiles from filepath, return smiles list
def read_data(filepath):

    with open(filepath, 'r') as f:
        s = f.read()
        smiles = s.split('\n')

    return smiles


# inputs smiles list, return list of unique chars
def get_unique_chars(smiles):

    unique_chars = []

    for smile in smiles:
        for char in smile:
            if char not in unique_chars:
                unique_chars.append(char)

    return unique_chars


# inputs list of unique chars, return character to index mapping
def get_char2idx(unique_chars):

    char2idx = {}

    for i, char in enumerate(unique_chars):
        char2idx[char] = i

    return char2idx
