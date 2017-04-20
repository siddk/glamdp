#contains domain information for the CleanupDomain configurations used.
#domain is defined as a dictionary, with prop functions as keys and room IDs as values

#used for grounding lifted RFs


#CleanupDomain from RSS submission/arXiv paper
cd_1 = {'roomIsGreen': 'room1', 'roomIsRed':'room0', 'roomIsBlue':'room2', 'start':'room2', 'block':'room0'}

#second CleanupDomain variant
#room configs the same, block and start position in differnt rooms
cd_2 = {'roomIsGreen': 'room1', 'roomIsRed':'room0', 'roomIsBlue':'room2', 'start':'room1', 'block':'room2'}

#mapping from domain ID to dictionary
id2domain = {'1':cd_1, '2':cd_2}
