#generates action sequences from a reward function in a specific domain

from sys import argv
from argparse import ArgumentParser
from ground_rf import ground_rf
import plans

def parse(args):
    parser = ArgumentParser()
    parser.parse_args("--rf", help="list of reward functions")
    parser.parse_args("--domain", help="domain ID") #TODO: multiple domains in same file?
    parser.parse_args("--lifted", help="if reward functions are lifted", action="store_true")
    parser.parse_args("--out", help="list of action sequences")
    return parser.parse_args(args)

def run(args):
    domain = domains.id2domain[args.domain]
    plan_dict = plans.id2plans[args.domain]

    #load reward functions
    with open(args.rf, 'r') as f:
        rfs = [rf.strip() for rf in f]

    #if lifted, produce grounded rfs

    if args.lifted:
        grounded_rfs = [ground_rf(lifted, domain) for lifted in rfs]
    else:
        grounded_rfs = rfs

    #get pre-computed actions from dictionary
    actions = [" ".join(plan_dict[grounded_rf]) for grounded_rfs in grounded_rfs]

    with open(args.out, 'w') as f:
        f.write("\n".join(actions))

if __name__=="__main__":
    run(parse(argv[1:]))
