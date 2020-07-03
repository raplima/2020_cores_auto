# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 07:05:37 2020

@author: Rafael
"""


"""
prepare the data for upcoming processes
"""

# import necessary libraries:
import os
import json
from sklearn.model_selection import KFold

# setup files
data_dir = 'cores'
dataset_tag = 'cores'

# read the classes dictionary
json_file = os.path.join(data_dir, "classes.json")
with open(json_file) as f:
    classes = json.load(f)

# read the colors dictionary
json_file = os.path.join(data_dir, "colors_dict.json")
with open(json_file) as f:
    patch_dict = json.load(f)

# save a list with same format as function and metadata:
thing_classes = sorted([it for _, it in classes.items()])

# read main json file
json_file = os.path.join(data_dir, "vanhorn_hawkins_payne_musselman_final_json.json")

# lump classes:
# Read in the file
with open(json_file, 'r') as file :
  filedata = file.read()
# Replace the target string
for c_old, c_replace in [('glauconitic_siltstone/sandsstone', 'laminated_siltstone'), 
                         ('crosslaminated_siltstone', 'laminated_siltstone')]:
    filedata = filedata.replace(c_old, c_replace)

# Write the file out again
with open(json_file, 'w') as file:
  file.write(filedata)

# finally read in as dictionary:
with open(json_file) as f:
    main_json = json.load(f)

# save the dictionary keys:
k = list(main_json.keys())
# create kfold
kf = KFold(n_splits=5, random_state=0, shuffle=True)

# split the dictionary into n_splits train/test:
for fold_idx, [train_index, test_index] in enumerate(kf.split(k)):
    print(f'Saving fold {fold_idx} json dictionary')
    for d, indexes in zip(["train", "val"], [train_index, test_index]):
        tag = f'{dataset_tag}_fold_{fold_idx}_{d}'
        split_dict = {}
        for idx in indexes:
            split_dict[str(idx)] = main_json[k[idx]]
        # save to file:
        with open(os.path.join(data_dir, f"{tag}.json"), 'w') as f:
            json.dump(split_dict, f)