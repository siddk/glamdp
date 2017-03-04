"""
train_seq2seq.py

trains a sequence to sequence RNN on the given parallel corpus.

"""

from argparse import ArgumentParser
from sys import argv, path
path.append('../data')
path.append('../models')
from data_utils import load_pc
from seq2seq_lifted import seq2seq_lifted

def parse(args):
    parser = ArgumentParser()
    parser.add_argument("--natural_language", help="file path to natural language list")
    parser.add_argument("--machine_language", help="file path to machine language list")
    return parser.parse_args(args)


def run(args):
    #load data from file
    nl, ml = load_pc(args.natural_language, args.machine_language)

    #create RNN model with given parallel corpus
    model = seq2seq_lifted((nl, ml))

    model.fit(45)




if __name__ == "__main__":
    run(parse(argv[1:]))
