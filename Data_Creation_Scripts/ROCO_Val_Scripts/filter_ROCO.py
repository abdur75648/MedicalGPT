import os
import json
import pandas as pd

# Load the CSV files
roco_df = pd.read_csv('radiologyvaldata.csv')
# Drop the id column
roco_df = roco_df.drop(columns=['id'])

print("Number of samples in the dataset: ", len(roco_df))

# print(roco_df.head(3))
#                                      name                                            caption
# 0    PMC3970251_CRIONM2014-931546.003.jpg   Axial computed tomography scan of the pelvis ...
# 1          PMC2766744_cios-1-176-g005.jpg   Postoperative anteroposterior radiograph of t...
# 2  PMC3789931_poljradiol-78-3-35-g001.jpg   Angiography of the internal carotid artery, l...

# Filter the data with caption length > 100
roco_df = roco_df[roco_df['caption'].str.len() > 100]

# Remove the data for which the 'images/{name}' does not exist in the path
roco_df = roco_df[roco_df['name'].apply(lambda x: os.path.exists('images/'+x))]

print("Number of samples in the dataset after filtering: ", len(roco_df))

# Randomly sample 40 samples
roco_df = roco_df.sample(n=100, random_state=42)

# Save the filtered data in a json format of the form [{"filename": "images/"+{name}, "Report": "None available"}, ...]

data = []

for index, row in roco_df.iterrows():
    data.append({"filename": "images/"+row['name'], "Report": row['caption']})

data_json = json.dumps(data, indent=2)

with open('roco_filtered.json', 'w') as f:
    f.write(data_json)