#creates a train/test split from a set of .ml and .en files

from argparse import ArgumentParser
from sys import argv
import random as r

r.seed(1)
nl_ext = '.en'
ml_ext = '.ml'

def parse(args):
    parser = ArgumentParser()
    parser.add_argument("--ml", help="machine language")
    parser.add_argument("--en", help="corresponding english data")
    parser.add_argument("--pct", help="percent training data", type=float)
    parser.add_argument("--out", help="output location string")
    return parser.parse_args(args)

#save strings to file
def save_strings(outfile, data):
    with open(outfile, 'w') as textfile:
        outstr = "\n".join(data)
        textfile.write(outstr)
        print "saved to {}".format(outfile)


def run(args):

    with open(args.ml, 'r') as f:
        ml = [line.strip() for line in f]

    with open(args.en, 'r') as f:
        en = [line.strip() for line in f]

    combined = zip(ml, en)

    r.shuffle(combined)

    num_train = int(len(combined)*args.pct)

    combined_train = combined[:num_train]
    combined_test = combined[num_train:]

    train_ml, train_en = zip(*combined_train)
    test_ml, test_en = zip(*combined_test)

    #Create output locations
    out_train_nl = args.out + "_train" + nl_ext
    out_train_ml = args.out + "_train" + ml_ext

    out_test_nl = args.out + "_test" + nl_ext
    out_test_ml = args.out + "_test" + ml_ext

    save_strings(out_train_nl, train_en)
    save_strings(out_train_ml, train_ml)
    save_strings(out_test_nl, test_en)
    save_strings(out_test_ml, test_ml)

if __name__=="__main__":
    run(parse(argv[1:]))
