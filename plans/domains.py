#contains domain information for the CleanupDomain configurations used.
#domain is defined as a dictionary, with prop functions as keys and room IDs as values

#used for grounding lifted RFs


#CleanupDomain from RSS submission/arXiv paper
cd_1 = {'roomIsGreen': 'room1', 'roomIsRed':'room0', 'roomIsBlue':'room2', 'start':'room2', 'block':'room0'}

#second CleanupDomain variant
#room configs the same, block and start position in differnt rooms
cd_new_start = {'roomIsGreen': 'room1', 'roomIsRed':'room0', 'roomIsBlue':'room2', 'start':'room1', 'block':'room2'}

#new domains for random sampling
#TODO: fix the start conditions?
cd_2 = {'roomIsGreen': 'room1', 'roomIsRed':'room2', 'roomIsBlue':'room0', 'start':'room1', 'block':'room2'}

cd_3 = {'roomIsGreen': 'room2', 'roomIsRed':'room0', 'roomIsBlue':'room1', 'start':'room1', 'block':'room2'}
cd_4 = {'roomIsGreen': 'room2', 'roomIsRed':'room1', 'roomIsBlue':'room0', 'start':'room1', 'block':'room2'}

cd_5 = {'roomIsGreen': 'room0', 'roomIsRed':'room1', 'roomIsBlue':'room2', 'start':'room1', 'block':'room2'}
cd_6 = {'roomIsGreen': 'room0', 'roomIsRed':'room2', 'roomIsBlue':'room1', 'start':'room1', 'block':'room2'}

#mapping from domain ID to dictionary
id2domain = {'1':cd_1, '2':cd_2, '3':cd_3, '4':cd_4, '5':cd_5, '6':cd_6}
