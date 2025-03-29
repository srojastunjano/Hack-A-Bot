from PIL import Image
import numpy as np
import os

input_folder = "asl_alphabet_train/asl_alphabet_train"
output_folder = "processed_Train"

os.makedirs(output_folder, exist_ok=True)

# for img_name in os.listdir(input_folder):
#     img = Image.open(os.path.join(input_folder, img_name))
#     img = img.resize((224, 224))
#     img_array = np.array(img) / 255.0 # normalized range from 255 pixels to [0,1] makes faster...
#     img.save(os.path.join(output_folder, img_name))

for letter_folder in os.listdir(input_folder):
    # check for .DSfile
    if letter_folder.startswith('.'):
        continue
        
    letter_output_path = os.path.join(output_folder, letter_folder)
    os.makedirs(letter_output_path, exist_ok=True)
    
    letter_folder_path = os.path.join(input_folder, letter_folder)
    
    for img_name in os.listdir(letter_folder_path):
        if img_name.startswith('.'):
            continue
            
        input_img_path = os.path.join(letter_folder_path, img_name)
        output_img_path = os.path.join(letter_output_path, img_name)
        
        try:
            with Image.open(input_img_path) as img:
                img = img.resize((224, 224))
                img_array = np.array(img) / 255.0  # Normalize to [0,1]... use later when training the model
                img.save(output_img_path)
        except Exception as e:
            print(f"Error processing {input_img_path}: {e}")