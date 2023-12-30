import csv

import clip
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# load semart data and join it in a single file
data_path = "../data/"
semart_path = data_path+"/SemArt/"
image_path = semart_path+"/Images/"

semart_train = pd.read_csv(
    semart_path+"semart_train.csv", encoding='latin1', sep="\t")
semart_val = pd.read_csv(semart_path+"semart_val.csv",
                         encoding='latin1', sep="\t")
semart_test = pd.read_csv(semart_path+"semart_test.csv",
                          encoding='latin1', sep="\t")
semart = pd.concat([semart_train, semart_val, semart_test])
semart = semart.rename(
    columns={"IMAGE_FILE": "image", "DESCRIPTION": "context"})

# load clip model
clip_architecture = "ViT-B/32"
clip_model, clip_processor = clip.load(clip_architecture, device=device)

# Open the CSV file for writing
csv_filename = data_path + "semart_cache.csv"
with open(csv_filename, 'w', newline='') as csv_file:
    # Create a CSV writer
    csv_writer = csv.writer(csv_file)

    # Write the header
    csv_writer.writerow(["image", "context", "img_emb"])

    # Process and write each image
    for image_name in tqdm(semart["image"].tolist()):
        image = Image.open(image_path + image_name)
        with torch.no_grad():
            inputs = clip_processor(image).unsqueeze(0).to(device)
            img_emb = clip_model.encode_image(inputs).to("cpu").squeeze()

        # Write the row to the CSV file
        csv_writer.writerow([image_name, semart.loc[semart["image"]
                            == image_name, "context"].values[0], img_emb.tolist()])

print("Cache saved to "+data_path+"semart_cache.csv")
