"""
train_seq2seq.py

trains a sequence to sequence RNN on the given parallel corpus.

"""
from argparse import ArgumentParser
from sys import argv, path
path.append('../data')
path.append('../models')
from data_utils import load_pc
from seq2seq_lifted import Seq2Seq_Lifted

levels = ['L0', 'L1', 'L2']

def parse(args):
    parser = ArgumentParser()
    parser.add_argument("--natural_language", help="file path to natural language list")
    parser.add_argument("--machine_language", help="file path to machine language list")
    return parser.parse_args(args)


def run(args):
    # Load data from file
    data_nl, data_ml = [], []
    for level in levels:
        nl, ml = load_pc(args.natural_language, args.machine_language, level)
        data_nl += nl
        data_ml += ml

    # Create RNN model with given parallel corpus
    model = Seq2Seq_Lifted((data_nl, data_ml))
    model.fit()
   


if __name__ == "__main__":
    run(parse(argv[1:]))
