#removes lines from the document

# if the line is marked as DELETED in the segmented corpus, remove it from the parallel corpus

# segmented = 'cd_config_2/means/segmented.en'
# pc_en = 'cd_config_2/means/cd2_L0_unsegmented.en'
# pc_ml = 'cd_config_2/means/cd2_lifted.ml'
#
# pc_out = 'cd_config_2/means/cd2_L0_synced'

# segmented = 'rss_data/means/L0_segmented_nomarks.en'
#
# pc_en = 'rss_data/means/L0_unsegmented.en'
# pc_ml = 'rss_data/means/L0_unsegmented.ml'
#
# pc_out ='rss_data/means/L0_rf_synced'

all_en = 'rss_data/means/segmenting/L0_205-NSEW_shuffled.en'
actions = 'rss_data/means/segmenting/ml_segmented_NSEW.txt'
all_rfs = 'rss_data/means/segmenting/L0_205-NSEW_shuffled.ml'

out = 'rss_data/means/segmenting/NSEW_'


# all_en = 'rss_data/means/segmenting/MERGED.en'
# actions = 'rss_data/means/segmenting/MERGED_actions.ml'
# all_rfs = 'rss_data/means/segmenting/MERGED_rfs.ml'
#
# out = 'rss_data/means/merged/MERGED_'

with open(all_en, 'r') as f:
    en = [line.strip() for line in f]

with open(actions, 'r') as f:
    actions = [line.strip() for line in f]

with open(all_rfs, 'r') as f:
    rfs = [line.strip() for line in f]


num_segmented = len(actions)

#separate out unsegmented and segmented language

print num_segmented

segmented_en = en[:num_segmented]
segmented_rfs = rfs[:num_segmented]
segmented_pc = zip(segmented_en, segmented_rfs, actions)

segmented_pc_filtered = [(en, rf, action) for en, rf, action in segmented_pc if en != "DELETED"]

print len(segmented_pc_filtered)

#save segmented synced PC

en_filt, rf_filt, action_filt = zip(*segmented_pc_filtered)

with open(out + "segmented.en", 'w') as f:
    f.write("\n".join(en_filt))

with open(out + "segmented_actions.ml", 'w') as f:
    f.write("\n".join(action_filt))

with open(out + "segmented_rf.ml", 'w') as f:
    f.write("\n".join(rf_filt))


#save remaining unsegmented PC separately

if (num_segmented < len(en)):
    unsegmented_en = en[num_segmented:]
    unsegmented_rfs = rfs[num_segmented:]

    with open(out + "unsegmented.en", 'w') as f:
        f.write("\n".join(unsegmented_en))

    with open(out + "unsegmented.ml", 'w') as f:
        f.write("\n".join(unsegmented_rfs))
