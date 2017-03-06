"""
parse_lifted.py

Parsing grounded commands from the test data to lifted AMDP reward functions.

"""

#defining mappings from grounded tasks to lifted tasks

L0_lifting = {'agentInRoom':'agentInRoom', 'agent0': '', 'blockInRoom': 'blockInRoom', 'block0': '', 'room1': 'roomIsGreen', 'room0': 'roomIsRed', 'room2': 'roomIsBlue', 'goNorth': 'goNorth', 'goSouth': 'goSouth', 'goEast':'goEast', 'goWest':'goWest'}
L1_lifting = {'agentInRegion':'agentInRegion', 'agent0': '', 'block0': '', 'blockInRegion': 'blockInRegion', 'room1': 'roomIsGreen', 'room0': 'roomIsRed', 'room2': 'roomIsBlue'}
L2_lifting = {'agentInRegion':'agentInRegion', 'agent0': '', 'block0': '', 'blockInRegion': 'blockInRegion', 'room1': 'roomIsGreen', 'room0': 'roomIsRed', 'room2': 'roomIsBlue'}


#input: string with filename of commands file
#output: list of strings, each string is a list of commands.
def load_commands(filename):
	with open(filename, 'r') as cmd_file:
		return [line.strip() for line in cmd_file]

#input: list of strings, filename
def save_commands(commands_list, outfile):
	with open(outfile, 'w') as out:
		out.write("\n".join(commands_list))


#input: string, dictionary specifying mapping between grounded groundings and lifted commands
def lift_command(grounded_command, lifting_dict):
	return " ".join([lifting_dict[elem] for elem in grounded_command.split()])

#input: list of strings, dictionary specifying mappings from grounded to lifted commands.
#output: list of strings
def lift_all(commands_list, lifted_dict):
	return [lift_command(cmd, lifted_dict) for cmd in commands_list]


if __name__=="__main__":
	cmd_file = "../data/grounded/L2.ml"
	outfile = "../data/lifted/L2.ml"
	cmds = load_commands(cmd_file)
	lifted = lift_all(cmds, L2_lifting)
	save_commands(lifted, outfile)
