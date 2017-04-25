#given a list of lifted groundings, picks random domains for those groundings to map to

from argparse import ArgumentParser
from sys import argv
from domains import id2domain
import random as r
from ground_rf import ground_rf #grounder

r.seed(0) #setting random seed

def parse(args):
    parser = ArgumentParser()
    parser.add_argument("--lifted", help="list of lifted RFs")
    parser.add_argument("--out", help="output path for saving ground truth groundings + list of domain numbers")
    return parser.parse_args(args)

def load_strings(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f]

def save_strings(filename, strings):
    with open(filename, 'w') as f:
        f.write("\n".join(strings))

def run(args):
    lifted_rfs = load_strings(args.lifted)

    domain_list = [str(r.randint(1, len(id2domain.keys()))) for _ in range(len(lifted_rfs))]

    grounded_rf = []

    for lifted, domain_id in zip(lifted_rfs, domain_list):
        domain = id2domain[domain_id]
        grounded_rf.append(ground_rf(lifted,domain))

    save_strings(args.out + "_grounded_gt.ml", grounded_rf)
    save_strings(args.out + "_domain_numbers.txt", domain_list)


if __name__=="__main__":
    run(parse(argv[1:]))
