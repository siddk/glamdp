#creating combined means/ends dataset
#means and ends data with corresponding level, RF, and action sequence data

from argparse import ArgumentParser
from sys import argv
import random as r
from split_data import save_strings

r.seed(1)
nl_ext = '.en'
ml_ext = '.ml'

def parse(args):
    parser = ArgumentParser()
    parser.add_argument("--means", help="location of means data")
    parser.add_argument("--ends", help="location of ends data")
    parser.add_argument("--out", help="location of combined dataset")
    return parser.parse_args(args)

def run(args):

    with open(args.means + ml_ext, 'r') as means_ml:
        means_ml = [line.strip() for line in means_ml]

    with open(args.means + nl_ext, 'r') as means_en:
        means_en = [line.strip() for line in means_en]

    with open(args.ends + ml_ext, 'r') as ends_ml:
        ends_ml = [line.strip() for line in ends_ml]

    with open(args.ends + nl_ext, 'r') as ends_en:
        ends_en = [line.strip() for line in ends_en]

    both_ml = means_ml + ends_ml
    both_en = means_en + ends_en

    #zip and shuffle
    combined = zip(both_ml, both_en)
    r.shuffle(combined)

    shuffled_ml, shuffled_en = zip(*combined) #unzipping

    save_strings(args.out + ml_ext, shuffled_ml)
    save_strings(args.out + nl_ext, shuffled_en)


if __name__=="__main__":
    run(parse(argv[1:]))
