#plans precomputed from AMDP planner for each domain


south = 'SOUTH'
north = 'NORTH'
east = 'EAST'
west = 'WEST'
pull = 'PULL'

#TODO: FILL IN
plans_cd1 = {"agentInRegion agent0 room0": [south, south, south]
            "agentInRegion agent0 room1": [south, south, south, west, west, west, west, north, north],
            "agentInRegion agent0 room2": [north],
            "blockInRegion block0 room0":[east],
            "blockInRegion block0 room1": [south, south, south, west, west, south, south, west, west, north, north, north],
            "blockInRegion block0 room2": [south, south, south, south, west, west, west, pull, east, east, east, south, east, north, north, north],
            "agentInRegion agent0 room0 blockInRegion block0 room0":[],
            "agentInRegion agent0 room1 blockInRegion block0 room0":[],
            "agentInRegion agent0 room2 blockInRegion block0 room0":[],
            "agentInRegion agent0 room0 blockInRegion block0 room1":[],
            "agentInRegion agent0 room1 blockInRegion block0 room1":[],
            "agentInRegion agent0 room2 blockInRegion block0 room1":[],
            "agentInRegion agent0 room0 blockInRegion block0 room2":[],
            "agentInRegion agent0 room1 blockInRegion block0 room2":[],
            "agentInRegion agent0 room2 blockInRegion block0 room2": []}
#TODO: FILL IN
plans_cd2 = {"agentInRegion agent0 room0": [south, south, south]
            "agentInRegion agent0 room1": [south, south, south, west, west, west, west, north, north],
            "agentInRegion agent0 room2": [north],
            "blockInRegion block0 room0":[east],
            "blockInRegion block0 room1": [south, south, south, west, west, south, south, west, west, north, north, north],
            "blockInRegion block0 room2": [south, south, south, south, west, west, west, pull, east, east, east, south, east, north, north, north],
            "agentInRegion agent0 room0 blockInRegion block0 room0":[],
            "agentInRegion agent0 room1 blockInRegion block0 room0":[],
            "agentInRegion agent0 room2 blockInRegion block0 room0":[],
            "agentInRegion agent0 room0 blockInRegion block0 room1":[],
            "agentInRegion agent0 room1 blockInRegion block0 room1":[],
            "agentInRegion agent0 room2 blockInRegion block0 room1":[],
            "agentInRegion agent0 room0 blockInRegion block0 room2":[],
            "agentInRegion agent0 room1 blockInRegion block0 room2":[],
            "agentInRegion agent0 room2 blockInRegion block0 room2":[]}



#mapping from domain ID to dictionary
id2plans = {'1':plans_cd1, '2':plans_cd2}
