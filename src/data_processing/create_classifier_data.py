#create training and test data for the RNN classifier

from argparse import ArgumentParser
import random as r
from sys import argv

def parse(args):
    parser = ArgumentParser()
    parser.add_argument("--lo", help="data files for low-level language", nargs="*")
    parser.add_argument("--hi", help="data files for high-level language", nargs="*")
    parser.add_argument("--split", help="pct of training data", type=float)
    parser.add_argument("--out", help="root path for output files")
    return parser.parse_args(args)

def run(args):
    #load all low-level
    low_level = []
    for dat in args.lo:
        with open(dat, 'r') as f:
            [low_level.append(line.strip() + ":0") for line in f.readlines()] #appending label

    high_level = []
    for dat in args.hi:
        with open(dat, 'r') as f:
            [high_level.append(line.strip() + ":1") for line in f.readlines()]

    both = low_level + high_level

    r.shuffle(both)

    train_ind = int(len(both)*args.split)

    train = both[:train_ind]
    test = both[train_ind:]

    #saving data to output file
    with open(args.out + "_train.txt", 'w') as trainfile:
        trainfile.write("\n".join(train))

    with open(args.out + "_test.txt", 'w') as testfile:
        testfile.write("\n".join(test))


if __name__=="__main__":
    run(parse(argv[1:]))
