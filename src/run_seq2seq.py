"""
train_seq2seq.py

trains a sequence to sequence RNN on the given parallel corpus.

"""
from argparse import ArgumentParser
from sys import argv, path
from data_processing.data_utils import load_pc
from models.seq2seq_lifted import Seq2Seq_Lifted

levels = ['L0', 'L1', 'L2']

def parse(args):
    parser = ArgumentParser()
    parser.add_argument("--nl_train", help="file path to natural language for testing", required=True)
    parser.add_argument("--ml_train", help="file path to natural language for testing", required=True)
    parser.add_argument("--nl_test", help="file path to natural language for testing", required=True)
    parser.add_argument("--ml_test", help="file path to natural language for testing", required=True)
    parser.add_argument("--load", help="Model to load", required=False)
    parser.add_argument("--save", help="directory to save model", required=False)
    return parser.parse_args(args)

def run(args):
    # Load data from file

    pc_train = load_pc(args.nl_train, args.ml_train)
    pc_test = load_pc(args.nl_test, args.ml_test)

    #Create RNN model with given parallel corpus
    if args.save:
        model = Seq2Seq_Lifted(pc_train, pc_test, epochs=10, verbose=0, save_to=args.save)
        model.fit()
        model.test(print_example=True)
    if args.load:
        model = Seq2Seq_Lifted(pc_train, pc_test, restore_from=args.load)
        model.test(print_example=True)



if __name__ == "__main__":
    run(parse(argv[1:]))
