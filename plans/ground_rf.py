#grounding lifted reward functions

import domains
from sys import argv
from argparse import ArgumentParser

#produces a grounded reward function for the given domain.
def ground_rf(lifted, domain):
    #lifted reward function consists of a sequence of prop function, binding constraint pairs
    rf_list = lifted.split()
    assert len(rf_list)% 2 == 0, "reward function {} is invalid".format(lifted)
    grounded_list = []
    for i in range(0, len(rf_list) - 1, 2):
        obj = "agent0" if rf_list[i] == 'agentInRegion' else 'block0'
        grounded_rf = "{} {} {}".format(rf_list[i], obj, domain[rf_list[i + 1]])
        grounded_list.append(grounded_rf)

    rf = " ".join(grounded_list)

    return rf

def parse(args):
    parser = ArgumentParser()
    parser.add_argument("--rf", help="list of reward functions")
    parser.add_argument("--domain", help="domain ID")
    parser.add_argument("--out", help="list of grounded reward function")
    return parser.parse_args(args)

def run(args):
    domain = domains.id2domain[args.domain]
    with open(args.rf, 'r') as rfs:
        grounded_rfs = [ground_rf(lifted.strip(), domain) for lifted in rfs]

    with open(args.out, 'w') as f:
        f.write("\n".join(grounded_rfs))


if __name__=="__main__":
    run(parse(argv[1:]))
