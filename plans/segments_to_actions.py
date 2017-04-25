#produces action sequences from NPI segments, essentially expanding the run-length-encoding

from argparse import ArgumentParser
from sys import argv
from randomize_grounding import save_strings, load_strings

def parse(args):
    parser = ArgumentParser()
    parser.add_argument("--npi_segments", help="NPI segmented actions")
    parser.add_argument("--action_trajectories", help="output action trajectories")
    return parser.parse_args(args)

def segments_to_trajectory(segmented_str):
    commands = [cmd.strip() for cmd in segmented_str.split('|')]

    trajectory = []

    for cmd in commands:
        direction, arg = tuple(cmd.split())
        for _ in range(int(arg)): #repeat command n times
            trajectory.append(direction)

    return trajectory

def run(args):
    npi_segments = load_strings(args.npi_segments)

    trajectories = [segments_to_trajectory(segmented) for segmented in npi_segments]

    outstr = [" ".join(t) for t in trajectories]

    save_strings(args.action_trajectories, outstr)

if __name__=="__main__":
    run(parse(argv[1:]))
