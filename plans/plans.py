#plans precomputed from AMDP planner for each domain


south = 'SOUTH'
north = 'NORTH'
east = 'EAST'
west = 'WEST'
pull = 'PULL'

plans_cd1 = {"agentInRegion agent0 room0": [south, south, south]
            "agentInRegion agent0 room1": [south, south, south, west, west, west, west, north, north],
            "agentInRegion agent0 room2": [north],
            "blockInRegion block0 room0":[east],
            "blockInRegion block0 room1": [south, south, south, west, west, south, south, west, west, north, north, north],
            "blockInRegion block0 room2": [south, south, south, south, west, west, west, pull, east, east, east, south, east, north, north, north],
            'agentInRegion agent0 room0 blockInRegion block0 room0': [south, south, south],
            'agentInRegion agent0 room0 blockInRegion block0 room1': [south, south, south, west, west, west, west, north, south, pull, north, north, south],
            'agentInRegion agent0 room0 blockInRegion block0 room2': [south, south, south, west, south, west, west, pull, east, east, east, south, east, north, north, north, south],
            'agentInRegion agent0 room1 blockInRegion block0 room0': [south, south, south, west, west, west, west, north, north],
            'agentInRegion agent0 room1 blockInRegion block0 room1': [south, south, south, west, west, south, west, south, west, north, north, north, north],
            'agentInRegion agent0 room1 blockInRegion block0 room2': [south, south, south, south, west, west, west, pull, east, east, east, south, east, north, north, north, south, west, west, west, west, north, north],
            'agentInRegion agent0 room2 blockInRegion block0 room0': [west],
            'agentInRegion agent0 room2 blockInRegion block0 room1': [south, south, south, west, south, south, west, west, west, north, north, north, south, east, east, east, east, north, north],
            'agentInRegion agent0 room2 blockInRegion block0 room2': [south, south, south, west, west, south, west, pull, east, east, east, south, east, north, north, north, north]}


plans_cd2 = {'agentInRegion agent0 room0 ': [south, south, south],
            'agentInRegion agent0 room1 ': [west],
            'agentInRegion agent0 room2 ': [south, south, south, east, east, east, east, north, north],
            'agentInRegion block0 room0 ': [south, south, south, east, east, east, east, north, north, pull, south, south],
            'agentInRegion block0 room1 ': [south, south, south, east, east, east, east, north, north, pull, south, south, south, pull, east, north, west, west, west, west, south, west, north, north],
            'agentInRegion block0 room2 ': [south],
            'agentInRegion agent0 room0 blockInRegion block0 room0': [south, south, south, east, east, east, east, north, north, pull, south, south, south],
            'agentInRegion agent0 room0 blockInRegion block0 room1': [south, south, south, east, east, east, east, north, north, pull, south, south, south, pull, east, north, west, west, west, west, south, west, north, north, south],
            'agentInRegion agent0 room0 blockInRegion block0 room2': [south, south, south],
            'agentInRegion agent0 room1 blockInRegion block0 room0': [south, south, south, east, east, east, east, north, north, pull, south, south, south, west, west, west, west, north, north],
            'agentInRegion agent0 room1 blockInRegion block0 room1': [south, south, south, east, east, east, east, north, north, pull, south, south, south, east, south, west, west, west, west, south, west, north, north, north, north],
            'agentInRegion agent0 room1 blockInRegion block0 room2': [west],
            'agentInRegion agent0 room2 blockInRegion block0 room0': [south, south, south, east, east, east, east, north, north, pull, south, south, north],
            'agentInRegion agent0 room2 blockInRegion block0 room1': [north, north, west, east, south, north, west, south, south, east, north, west, west, north, east, north, south, west, west, north, north, south, east, north, south, north, south, east, east, north, east, east, west, east, north, south, north, east, south, north, east, south, north, south, west, south, south, south, south, north, west, east, south, south, east, south, east, south, north, east, south, west, west, north, north, north, east, east, west, south, north, south, west, east, south, east, south, east, east, west, west, west, south, south, east, west, west, north, west, south, north, east, south, east, south, west, east, south, east],
            'agentInRegion agent0 room2 blockInRegion block0 room2': [south, south, south, east, east, east, east, north, north]}

#mapping from domain ID to dictionary
id2plans = {'1':plans_cd1, '2':plans_cd2}
