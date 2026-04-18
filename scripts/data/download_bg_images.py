"""
scripts/data/download_bg_images.py
-----------------------------------
Download 2,000 COCO val2017 background images (no people or clothing).
These are used as negative examples for edna_1.4m training.

Usage:
    python scripts/data/download_bg_images.py
"""

import requests, json, os, shutil
from pathlib import Path

print("Downloading COCO annotations...")
os.system("wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -q")
os.system("unzip -q annotations_trainval2017.zip annotations/instances_val2017.json")

with open("annotations/instances_val2017.json") as f:
    coco = json.load(f)

# Exclude: person + clothing-adjacent categories
EXCLUDE = {1, 27, 28, 31, 32}  # person, backpack, umbrella, handbag, tie

ann_by_img = {}
for ann in coco['annotations']:
    ann_by_img.setdefault(ann['image_id'], set()).add(ann['category_id'])

clean_imgs = [
    img for img in coco['images']
    if not ann_by_img.get(img['id'], set()) & EXCLUDE
][:2000]

print(f"Found {len(clean_imgs)} clean background images")

out = Path("bg_images")
out.mkdir(exist_ok=True)
for i, img in enumerate(clean_imgs):
    url = img['coco_url']
    fname = out / f"bg_{img['id']}.jpg"
    r = requests.get(url, stream=True)
    with open(fname, 'wb') as f:
        shutil.copyfileobj(r.raw, f)
    if i % 50 == 0:
        print(f"  {i}/{len(clean_imgs)}")

print("Done.")
