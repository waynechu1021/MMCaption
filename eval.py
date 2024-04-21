from models.multimodal_mamba import MultiModalMambaLMHeadModel
from train.coco_eval import COCOEvalCap
from pycocotools.coco import COCO
from tqdm import tqdm
import transformers
from train.data import make_supervised_data_module
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
from train.mm_mamba_trainer import MultiModalMambaTrainer

@dataclass
class ModelArguments:
    vmamba_path: str = field(default='configs/videomamba_middle.json') 
    mamba_lm_path: Optional[str] = field(default="pretrained_model/language_model/mamba-130m")
    ckpt_path:str = field(default=None) 

@dataclass
class DataArguments:
    tokenizer: str = field(default="pretrained_model/gpt-neox-20b",
                           metadata={"help": "Tokenizer to use."})
    data_path: str = field(default='data/coco2014',
                           metadata={"help": "Path to the training data."})
    image_folder: Optional[str] = field(default='data/coco2014/images')
    num_image_tokens: Optional[int] = field(default=49)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    evaluate_file_path: Optional[str] = field(default='data/coco2014/annotations/coco_karpathy_val_gt.json',
                                              metadata={"help": "Path to the evaluation file."})
    max_length: Optional[int] = field(default=12,
                                      metadata={"help": "Max length of the input."})
    
    frozen_vision_backbone: Optional[bool] = field(default=False,
                                                   metadata={"help": "Whether to freeze the vision backbone."})
    frozen_language_model:Optional[bool] = field(default=False,
                                                   metadata={"help": "Whether to freeze the language model."})
def eval():
    parser = transformers.HfArgumentParser(
            (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    tokenizer = transformers.AutoTokenizer.from_pretrained('pretrained_model/gpt-neox-20b')
    #tokenizer = transformers.AutoTokenizer.from_pretrained('pretrained_model/language_model/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e')
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token
    data_module = make_supervised_data_module(tokenizer, data_args, training_args.max_length)
    # print(len(data_module["train_dataset"]))  #414113 annotations  82783 images
    # print(len(data_module["eval_dataset"]))   #25010 annotations  5000 images
    model = MultiModalMambaLMHeadModel.from_pretrained(model_args.ckpt_path)
    trainer = MultiModalMambaTrainer(
            evaluate_file_path=training_args.evaluate_file_path,
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            **data_module
        )
    trainer.evaluate(data_module['eval_dataset'])

if __name__ == '__main__':
    eval()