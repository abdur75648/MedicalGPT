import os
import json
import pandas as pd

# Load the CSV files
files_df = pd.read_csv('indiana_files.csv')
reports_df = pd.read_csv('indiana_reports.csv')


# Join these 2 using uid column
combined_df = pd.merge(files_df, reports_df, on='uid')
# Drop the uid column
combined_df = combined_df.drop(columns=['uid'])
combined_df.head(5)

# For the column "filename" replace the any occurence of ".dcm" with an empty string
combined_df['filename'] = combined_df['filename'].str.replace('.dcm', '')
# For the column "filename" add 'CXR' to beginning of each value
combined_df['filename'] = 'images/CXR' + combined_df['filename']

# Drop NaN values
combined_df = combined_df.dropna()

# Convert all the rows into a list of dictionaries
data = combined_df.to_dict(orient='records')

# Print total number of records
print(f'Total number of records: {len(data)}')

# Remove the records for which the filename does not exist in the path
data = [record for record in data if os.path.exists(record['filename'])]

# Filter the records to with length of "findings" and "impression" greater than 50
data = [record for record in data if len(record['findings']) > 50 and len(record['impression']) > 50]

# print the total number of records
print(f'Total number of records after filtering: {len(data)}')

## Wrong - Choose first 10 records
# data = data[:10]

## Right - Randomly select 10 records
import random
data = random.sample(data, 10)

# Convert the data into a JSON string
data_json = json.dumps(data, indent=2)

# Write the JSON string to a file
with open('indiana_filtered.json', 'w') as f:
    f.write(data_json)
    
print('Done!')