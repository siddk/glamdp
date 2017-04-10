#Create training and test split, save data, for repeatability

import random as r
from sys import argv
from argparse import ArgumentParser

r.seed(1)
levels = ['L0', 'L1', 'L2']
nl_ext = '.en'
ml_ext = '.ml'

def parse(args):
    parser = ArgumentParser()
    parser.add_argument('--in_folder', help="folder where command pairs are stored.")
    parser.add_argument('--pct_train', help="percentage of data for training", type=float)
    parser.add_argument('--out_folder', help="location for output training and test")
    return parser.parse_args(args)

#loading strings, prepending level, go, and eos if needed
def load_strings_level(filepath, level=None):
    with open(filepath, 'r') as textfile:
        if level:
            return [['<<GO>>', level] + line.strip().split() + ['<<EOS>>'] for line in textfile]
        else:
            return [line.strip().split() for line in textfile]

#save strings to file
def save_strings(outfile, data):
    with open(outfile, 'w') as textfile:
        outstr = "\n".join([" ".join(line) for line in data])
        textfile.write(outstr)
        print "saved to {}".format(outfile)


def run(args):
    #Load all data points from all levels

    all_data = []
    for level in levels:
        loc = "{}/{}".format(args.in_folder, level)
        nl = load_strings_level(loc + nl_ext)
        ml = load_strings_level(loc + ml_ext, level=level)
        all_data.extend(zip(nl, ml))

    #Shuffle data
    r.shuffle(all_data)

    #Create train test split
    num_train = int(len(all_data)*args.pct_train)

    all_train = all_data[:num_train]
    all_test = all_data[num_train:]

    #Create output locations
    out_train_nl = args.out_folder + "_train" + nl_ext
    out_train_ml = args.out_folder + "_train" + ml_ext

    out_test_nl = args.out_folder + "_test" + nl_ext
    out_test_ml = args.out_folder + "_test" + ml_ext

    nl_train, ml_train = zip(*all_train)
    nl_test, ml_test = zip(*all_test)

    save_strings(out_train_nl, nl_train)
    save_strings(out_train_ml, ml_train)
    save_strings(out_test_nl, nl_test)
    save_strings(out_test_ml, ml_test)

if __name__=="__main__":
    run(parse(argv[1:]))
