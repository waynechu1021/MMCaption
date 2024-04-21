from transformers import Trainer
import torch
import os
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset
import time
from transformers.trainer_utils import speed_metrics, has_length
from transformers.trainer_pt_utils import find_batch_size, IterableDatasetShard
import math
from transformers.integrations.deepspeed import deepspeed_init
import json
from .coco_eval import COCOEvalCap
from pycocotools.coco import COCO
from tqdm import tqdm

class MultiModalMambaTrainer(Trainer):

    def __init__(self, evaluate_file_path, **kwargs):
        self.evaluate_file_path = evaluate_file_path
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("label_ids")
        output = model(**inputs)
        lm_logits = output.logits

        labels = labels.to(lm_logits.device)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss()
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

        return lm_loss
    
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()
        args = self.args
        metrics = {}


        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=eval_dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        model.eval()

        self.callback_handler.eval_dataloader = eval_dataloader
        # Do this before wrapping.
        eval_dataset = getattr(eval_dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        current_predictions = {'image_ids': [], 'captions': []}
        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in tqdm(enumerate(eval_dataloader)):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            out = model.generate(**inputs, eos_token_id=self.tokenizer.eos_token_id, max_length=self.args.max_length, top_p=0.7)
            decoded = self.tokenizer.batch_decode(out)
            for image_id, sentence in zip(inputs["image_ids"], decoded):
                caption = sentence.split(self.tokenizer.eos_token)[1]
                current_predictions['image_ids'].append(image_id)
                current_predictions['captions'].append(caption.strip())

            current_predictions = self.accelerator.gather_for_metrics(current_predictions)
        
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")
        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(eval_dataloader):
                num_samples = self.num_examples(eval_dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples


        # TODO: Save file and calculate metrics
        # Save the predictions
        if self.is_world_process_zero():
            # reformt the predictions as a list of dictionaries
            current_predictions = [{'image_id': image_id, 'caption': caption} for image_id, caption in zip(current_predictions['image_ids'], current_predictions['captions'])] 
            output_dir = self.args.output_dir
            prediction_dir = os.path.join(output_dir, "predictions")
            if not os.path.exists(prediction_dir):
                os.makedirs(prediction_dir)
            # save json file, named with epoch number
            results_file = os.path.join(prediction_dir, f"prediction_{self.state.global_step}.json")
            with open(results_file, "w") as f:
                json.dump(current_predictions, f)

            coco = COCO(self.evaluate_file_path)
            coco_result = coco.loadRes(results_file)
            coco_eval = COCOEvalCap(coco, coco_result)
            coco_eval.evaluate()
            for metric, score in coco_eval.eval.items():
                metrics[f"{metric_key_prefix}_{metric}"] = score

        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=num_samples,
                num_steps=math.ceil(num_samples / total_batch_size),
            )
        )

        self.log(metrics)

        self._memory_tracker.stop_and_update_metrics(metrics)

        return metrics

    def save_model(self, output_dir, _internal_call):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)