import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
from torch.utils.data import Dataset
from train.data import make_supervised_data_module
from train.mm_mamba_trainer import MultiModalMambaTrainer
import pathlib
import logging
from transformers.trainer import TrainerCallback
import os
from models.multimodal_mamba import init_multimodal_mamba_model
import torch
from typing import Union
import random
import numpy as np

@dataclass
class ModelArguments:
    vmamba_path: str = field(default=None) 
    mamba_lm_path: Optional[str] = field(default="pretrained_model/language_model/mamba-130m")

@dataclass
class DataArguments:
    tokenizer: str = field(default="pretrained_model/gpt-neox-20b",
                           metadata={"help": "Tokenizer to use."})
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    image_folder: Optional[str] = field(default=None)
    num_image_tokens: Optional[int] = field(default=196)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    evaluate_file_path: Optional[str] = field(default=None,
                                              metadata={"help": "Path to the evaluation file."})
    max_length: Optional[int] = field(default=72,
                                      metadata={"help": "Max length of the input."})
    
    frozen_vision_backbone: Optional[bool] = field(default=False,
                                                   metadata={"help": "Whether to freeze the vision backbone."})
    frozen_language_model:Optional[bool] = field(default=False,
                                                   metadata={"help": "Whether to freeze the language model."})
class LogCallback(TrainerCallback):
    def __init__(self, logging):
        self.logging = logging

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            self.logging.info(logs)

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    # torch.use_deterministic_algorithms(True, warn_only=False)
    # # Enable CUDNN deterministic mode
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def train():

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    file_formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                                       datefmt="%m/%d/%Y %H:%M:%S", )
    set_seed(training_args.seed)
    if training_args.local_rank == 0 or training_args.local_rank == -1:
        os.makedirs(training_args.output_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(training_args.output_dir, f"train.log"))
        file_handler.setFormatter(file_formatter)
        logging.root.addHandler(file_handler)
        logging.root.setLevel(logging.INFO)

        logging.info("Training/evaluation parameters %s", training_args)
        logging.info("Model parameters %s", model_args)
        logging.info("Data parameters %s", data_args)
    #tokenizer = transformers.AutoTokenizer.from_pretrained(data_args.tokenizer)
    tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2',cache_dir='./pretrained_model/language_model')
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token
    data_module = make_supervised_data_module(tokenizer, data_args, training_args.max_length)
    dtype = torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
    # TODO build model
    model = init_multimodal_mamba_model(
        vision_encoder_config_path=model_args.vmamba_path,
        mamba_lm_model_path=model_args.mamba_lm_path,
        device=training_args.device,
        dtype=dtype
    )
    logging.info('Vision Projection Layer %s',model.vision_projection_layer)
    if training_args.frozen_vision_backbone:
        # frozen parameters in model.vision_encoder
        for name, p in model.vision_encoder.named_parameters():
            p.requires_grad = False
            # if "glu" not in name:
            #     p.requires_grad = False
            # else:
            #     logging.info('Unfeeze parameter %s',name)
    if training_args.frozen_language_model:
        # frozen parameters in model.backbone
        for name, p in model.backbone.named_parameters():
            if "ffn" not in name:
                p.requires_grad = False
            else:
                logging.info('Unfeeze parameter %s',name)
            #p.requires_grad = False
    logging.info('Trainable parameters: %s',sum(p.numel() for p in model.parameters() if p.requires_grad))
    log_callback = LogCallback(logging)
    trainer = MultiModalMambaTrainer(
        evaluate_file_path=training_args.evaluate_file_path,
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=[log_callback],
        **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

if __name__ == '__main__':
    train()