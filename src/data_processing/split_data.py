#creates a train/test split from a set of .ml and .en files

from argparse import ArgumentParser
from sys import argv
import random as r

r.seed(1)
nl_ext = '.en'
ml_ext = '.ml'

def parse(args):
    parser = ArgumentParser()
    parser.add_argument("--grounded_ml", help="grounded machine language")
    parser.add_argument("--lifted_ml", help = "lifted machine language")
    parser.add_argument("--en", help="corresponding english data")
    parser.add_argument("--pct", help="percent test data", type=float)
    parser.add_argument("--out", help="output location ")
    return parser.parse_args(args)

#save strings to file
def save_strings(outfile, data):
    with open(outfile, 'w') as textfile:
        outstr = "\n".join(data)
        textfile.write(outstr)
        print "saved to {}".format(outfile)

def run(args):

    with open(args.grounded_ml, 'r') as f:
        grounded_ml = [line.strip() for line in f]

    with open(args.lifted_ml, 'r') as f:
        lifted_ml = [line.strip() for line in f]

    with open(args.en, 'r') as f:
        en = [line.strip() for line in f]

    combined = zip(en, grounded_ml, lifted_ml)

    r.shuffle(combined)

    num_train = int(len(combined)*args.pct)

    train_en, train_grounded, train_lifted = zip(*combined[:num_train])
    test_en, test_grounded, test_lifted = zip(*combined[num_train:])

    save_strings(args.out + "train.en", train_en)
    save_strings(args.out + "train_npi_lifted.ml", train_lifted)
    save_strings(args.out + "train_rnn_grounded.ml", train_grounded)

    save_strings(args.out + "test.en", train_en)
    save_strings(args.out + "test_npi_lifted.ml", train_lifted)
    save_strings(args.out + "test_rnn_grounded.ml", train_grounded)

if __name__=="__main__":
    run(parse(argv[1:]))
