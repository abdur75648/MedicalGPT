# Script to copy required images in new folder
import json, os, shutil
from tqdm import tqdm

json_file = 'roco_vqa.json'
images_folder_original = "images"
images_folder_new = "images_new"

# Create new folder if not exists
if not os.path.exists(images_folder_new):
    os.makedirs(images_folder_new)
    
with open(json_file) as f:
    data = json.load(f)

# Copy images to new folder
for item in tqdm(data):
    image = item['image']
    image_name = image.split('/')[-1]
    shutil.copyfile(os.path.join(images_folder_original, image_name), os.path.join(images_folder_new, image_name))
