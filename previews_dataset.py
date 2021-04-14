import io


import torchvision.transforms
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
from PIL import Image

import captions_db


def end_sentence(text):
    if text == "":
        return text
    
    if text[-1] not in ".!?":
        text += "."
    
    return text + " "

def tags_text(tags):
    tags = ", ".join(t.strip() for t in tags.split(","))
    return "Tags: " + tags

def simple_extract_caption_plaintext(src, cc):
    if src.startswith("yfcc100m"):
        title, desc, tags = cc.split("\t")
        pt = title
        if desc != "":
            pt = end_sentence(pt) + desc
        
        if tags != "":
            pt = end_sentence(pt) + tags_text(tags)
    elif src.startswith("wikicaps") or src.startswith("src2") or src.startswith("conceptual_captions") \
        or src.startswith("wit") or src.startswith("sbu_captions") or src.startswith("localized_narratives"):
        pt = cc
    elif src.startswith("src1"):
        topic, desc = cc.split("\t")
        pt = end_sentence(topic) + desc
    elif src.startswith("src3"):
        pt = tags_text(cc)
    else:
        assert False, f"Unknown source: {src}"
        
    return pt

fit_center_crop_transform = torchvision.transforms.RandomResizedCrop(256, scale=(1.0, 1.0), ratio=(1.0, 1.0))


def fit_center_crop_image(img):
    img = Image.open(io.BytesIO(img)).convert("RGB")
    img = fit_center_crop_transform(img)
    return torchvision.transforms.functional.to_tensor(img)
       

class PreviewsDataset(IterableDataset):
    def __init__(self, db_file, start_idx=0):
        self.db = captions_db.DiskBackedListReader(db_file)
        self.start_idx = start_idx
        
    def __iter__(self):
        while True:
            start = self.start_idx
            for i in range(start, len(self.db)):
                img, caps = captions_db.deserialize_image_captions(self.db[i])
                src, cc = caps[np.random.choice(len(caps))]
                yield fit_center_crop_image(img), simple_extract_caption_plaintext(src, cc)
                
            start = 0
