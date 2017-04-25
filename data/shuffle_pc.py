#shuffles parallel corpus

import random as r

pc_file = 'rss_data/means/L0_205-NSEW'


with open(pc_file + ".en", 'r') as f:
    pc_en = [line.strip() for line in f]

with open(pc_file + ".ml", 'r') as f:
    pc_ml = [line.strip() for line in f]

pc = zip(pc_en, pc_ml)

print len(pc)

r.shuffle(pc)

pc_en, pc_ml = zip(*pc)

with open(pc_file + "_shuffled.en", 'w') as f:
    f.write("\n".join(pc_en))

with open(pc_file + "_shuffled.ml", 'w') as f:
    f.write("\n".join(pc_ml))
