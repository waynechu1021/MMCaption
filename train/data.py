import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
from torch.utils.data import Dataset
import json
from PIL import Image
import torch
import os
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from torchvision.transforms import InterpolationMode

class COCOCaptionDataset(Dataset):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer,
                 data_args: Dict, split: str = "train", max_length: int = 72):
        annotation_folder= data_args.data_path
        annotation_path = os.path.join(annotation_folder, f"annotations/{split}_caption.json")
        self.split = split
        self.tokenizer = tokenizer
        self.image_folder = data_args.image_folder
        self.num_image_tokens = data_args.num_image_tokens
        self.max_length = max_length
        with open(annotation_path, "r") as f:
            self.annotations = json.load(f)
        # If it's not a train dataset, remove sample with duplicate image_id
        if split != "train":
            image_ids_set = set()
            filtered_annotations = []
            for ann in self.annotations:
                if ann["image_id"] not in image_ids_set:
                    image_ids_set.add(ann["image_id"])
                    filtered_annotations.append(ann)
            self.annotations = filtered_annotations
        self.transform = build_transform()
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_id = ann['image_id']
        caption = ann['caption']

        filename = os.path.join(self.image_folder, f"{self.split}2014/COCO_{self.split}2014_{int(image_id):012d}.jpg")
        image = Image.open(filename).convert("RGB")

        image, input_ids, labels = process(image, caption, self.tokenizer, self.transform, self.num_image_tokens, self.max_length)

        if self.split != "train":
            input_ids = input_ids[: 1]
        data_dict = dict(
            image_ids=image_id,
            images=image,
            input_ids=input_ids,
            label_ids=labels
        )
        return data_dict

@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer


    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, label_ids = tuple([instance[key] for instance in instances] for key in ("input_ids", "label_ids"))
        image_ids = [instance["image_ids"] for instance in instances]
        images = torch.stack([instance["images"] for instance in instances], 0)
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        label_ids = torch.nn.utils.rnn.pad_sequence(label_ids, batch_first=True, padding_value=-100)

        # ignore image tokens

        return dict(
            image_ids=image_ids,
            images=images,
            input_ids=input_ids,
            label_ids=label_ids,
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args, max_length) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = COCOCaptionDataset(tokenizer=tokenizer, data_args=data_args, split='train', max_length=max_length)
    eval_dataset = COCOCaptionDataset(tokenizer=tokenizer, data_args=data_args, split="val", max_length=max_length)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)


def process(image, caption, tokenizer, transform, num_image_tokens=196, max_length=72):
    # TODO add image process
    image = transform(image)
    input_ids = tokenizer.encode(caption, return_tensors="pt", max_length=max_length, truncation=True)
    sep_id = torch.empty([1, 1], dtype=torch.long).fill_(tokenizer.eos_token_id)
    # Add eos token to the input_ids
    # Add eos token between image and text
    input_ids = torch.cat((sep_id, input_ids, sep_id), dim=-1)
    empty_targets = (
            torch.ones([1, num_image_tokens + 1], dtype=torch.long).to(image.device).fill_(-100)
        )
    labels = torch.cat((empty_targets, input_ids[:, 1:]), dim=-1)
    input_ids = input_ids.squeeze(0)
    labels = labels.squeeze(0)
    return image, input_ids, labels


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize((448, 448), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ]
    )