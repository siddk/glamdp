"""
validator.py
"""
import re

CONSTRAINTS = {"inRedRoom", "inBlueRoom", "inGreenRoom", "nextToRedChair", "nextToBlueChair",
               "nextToGreenChair", "nextToChair"}

# Boundaries [(left, right), (bottom, top)] (0, 1, 2)
ROOM_BOUNDARIES = [[range(0, 8), range(0, 4)], [range(0, 4), range(4, 8)],  [range(4, 8), range(4, 8)]]

# Coordinate Regex
COORD_RE = re.compile('x: {\d} y: {\d}')
COLOR_RE = re.compile('colour: {(red|blue|green)}')


def parse(lines):
    examples, stride, counter = [], 13, 0
    for idx in range(0, len(lines), stride):
        command = lines[idx]

        f1, f2, f3 = lines[idx+2], lines[idx+6], lines[idx+10]
        c1, c2, c3 = parse_state(f1), parse_state(f2), parse_state(f3)
        intersection = c1 & c2 & c3

        examples.append((command, intersection))

    return examples


def parse_state(state):
    """Parse a state string, return set of valid constraints."""
    ents, constraints = filter(lambda x: 'door' not in x, state.split(',')[:-1]), set()
    agent_pos, blocks, rooms = ents[0], ents[1:-3], ents[-3:]

    # Asserts for Safe-Keeping
    assert('agent' in agent_pos)
    assert(reduce(lambda x, y: x and y, map(lambda z: 'room' in z, rooms)))
    assert(reduce(lambda x, y: x and y, map(lambda z: 'block' in z, blocks)))

    # Filter out Non-Chairs
    blocks = filter(lambda x: 'chair' in x, blocks)

    # Compute Agent X, Y Position
    agent_pos = re.findall(COORD_RE, agent_pos)
    assert(len(agent_pos) == 1)
    agent_pos = (int(agent_pos[0][4]), int(agent_pos[0][11]))

    # Find Agent Room (Assume only in 1 room)
    agent_room = None
    for i, bounds in enumerate(ROOM_BOUNDARIES):
        if agent_pos[0] in bounds[0] and agent_pos[1] in bounds[1]:
            agent_room = i

    # Get Room Color
    room_color = re.findall(COLOR_RE, rooms[agent_room])
    if room_color[0] == 'red':
        constraints.add("inRedRoom")
    elif room_color[0] == 'blue':
        constraints.add("inBlueRoom")
    elif room_color[0] == 'green':
        constraints.add("inGreenRoom")
    else:
        raise EnvironmentError

    # Block Constraints - Compute Position, Color for each
    for b in blocks:
        block_pos = re.findall(COORD_RE, b)
        block_pos = (int(block_pos[0][4]), int(block_pos[0][11]))

        # Check if next to block
        if next_to(agent_pos, block_pos):
            # Get color, add to constraints
            block_color = re.findall(COLOR_RE, b)
            if block_color[0] == 'red':
                constraints.add("nextToRedChair")
            elif block_color[0] == 'blue':
                constraints.add("nextToBlueChair")
            elif block_color[0] == 'green':
                constraints.add("nextToGreenChair")
            else:
                raise EnvironmentError

            # Add general nextToChair constraint
            constraints.add("nextToChair")

    # Return Constraints
    return constraints


def next_to(agent_pos, block_pos):
    return abs((agent_pos[0] - block_pos[0]) + (agent_pos[1] - block_pos[1])) == 1

if __name__ == "__main__":
    # Load Train File
    with open('data/train_shuffled.bdm', 'r') as f:
        train_l = map(lambda x: x.strip(), f.readlines())

    # Parse Lines
    train_data = parse(train_l)

    # Load Test File
    with open('data/test_shuffled.bdm', 'r') as f:
        test_l = map(lambda x: x.strip(), f.readlines())

    # Parse Lines
    test_data = parse(test_l)

    # Write to File
    with open('data/validation_function.pik', 'w') as f:
        import pickle
        pickle.dump((train_data, test_data), f)