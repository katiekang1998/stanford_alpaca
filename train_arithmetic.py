import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
import utils
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
import os
from datasets import load_dataset
import numpy as np
import json
import argparse


IGNORE_INDEX = -100

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
        
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    sources_lens = sources_tokenized["input_ids_lens"]
    if tokenizer.add_eos_token:
        sources_lens = [source_len-1 for source_len in sources_lens]
    for label, source_len in zip(labels, sources_lens):
        label[:source_len] = IGNORE_INDEX
    
    return dict(input_ids=input_ids, labels=labels)


def get_target(train_steps):
    target_str = train_steps[0]
    for step in train_steps[1:]:
        target_str += " = " + step
    return target_str

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, train_examples, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        

        sources = [ 
            train_example[0] + " = " for train_example in train_examples
        ]
        
        
        train_examples_solution_steps = [train_example[1:] for train_example in train_examples]
        targets = list(map(get_target,train_examples_solution_steps))
                
        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(data_type, num_train_points, tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    if data_type == "easy_full":
        train_2steps = np.load("arithmetic/2step_train.npy")
        subsample_idxs = np.random.choice(np.arange(0, len(train_2steps)), size=num_train_points//3, replace=False)
        train_2steps = train_2steps[subsample_idxs]

        train_3steps = np.load("arithmetic/3step_train.npy")
        subsample_idxs = np.random.choice(np.arange(0, len(train_3steps)), size=num_train_points//3, replace=False)
        train_3steps = train_3steps[subsample_idxs]
        
        train_4steps = np.load("arithmetic/4step_train.npy")
        subsample_idxs = np.random.choice(np.arange(0, len(train_4steps)), size=num_train_points//3, replace=False)
        train_4steps = train_4steps[subsample_idxs]
                
        train_steps = list(train_2steps) + list(train_3steps) + list(train_4steps)
    # elif data_type == "hard_full":
    #     train_2steps = np.load("arithmetic/5step_train.npy")
    #     subsample_idxs = np.random.choice(np.arange(0, len(train_2steps)), size=num_train_points//3, replace=False)
    #     train_2steps = train_2steps[subsample_idxs]

    #     train_3steps = np.load("arithmetic/6step_train.npy")
    #     subsample_idxs = np.random.choice(np.arange(0, len(train_3steps)), size=num_train_points//3, replace=False)
    #     train_3steps = train_3steps[subsample_idxs]
        
    #     train_4steps = np.load("arithmetic/7step_train.npy")
    #     subsample_idxs = np.random.choice(np.arange(0, len(train_4steps)), size=num_train_points//3, replace=False)
    #     train_4steps = train_4steps[subsample_idxs]
                
    #     train_steps = list(train_2steps) + list(train_3steps) + list(train_4steps)
    elif data_type == "hard_skip":
        
        train_5steps = np.load("arithmetic/5step_train.npy")
        subsample_idxs = np.random.choice(np.arange(0, len(train_5steps)), size=int(num_train_points/3), replace=False)
        train_5steps = train_5steps[subsample_idxs]
        train_5steps_skip = np.dstack([train_5steps[:, 0], train_5steps[:, 3], train_5steps[:, -1]])[0]


        train_6steps = np.load("arithmetic/6step_train.npy")
        subsample_idxs = np.random.choice(np.arange(0, len(train_6steps)), size=num_train_points//3, replace=False)
        train_6steps = train_6steps[subsample_idxs]
        train_6steps_skip = np.dstack([train_6steps[:, 0], train_6steps[:, 3], train_6steps[:, -1]])[0]

        
        train_7steps = np.load("arithmetic/7step_train.npy")
        subsample_idxs = np.random.choice(np.arange(0, len(train_7steps)), size=num_train_points//3, replace=False)
        train_7steps = train_7steps[subsample_idxs]
        train_7steps_skip = np.dstack([train_7steps[:, 0], train_7steps[:, 4], train_7steps[:, -1]])[0]
        
        ratio_full = 0.2
        num_full = int(num_train_points/3*ratio_full)
                        
        train_steps = list(train_5steps[:num_full])+ list(train_5steps_skip[num_full:]) + list(train_6steps[:num_full])+ list(train_6steps_skip[num_full:])+ list(train_7steps[:num_full])+ list(train_7steps_skip[num_full:])
    elif data_type == "mixed":
        train_2steps = np.load("arithmetic/2step_train.npy")
        subsample_idxs = np.random.choice(np.arange(0, len(train_2steps)), size=num_train_points//6, replace=False)
        train_2steps = train_2steps[subsample_idxs]

        train_3steps = np.load("arithmetic/3step_train.npy")
        subsample_idxs = np.random.choice(np.arange(0, len(train_3steps)), size=num_train_points//6, replace=False)
        train_3steps = train_3steps[subsample_idxs]
        
        train_4steps = np.load("arithmetic/4step_train.npy")
        subsample_idxs = np.random.choice(np.arange(0, len(train_4steps)), size=num_train_points//6, replace=False)
        train_4steps = train_4steps[subsample_idxs]
        
        train_steps_easy = list(train_2steps) + list(train_3steps) + list(train_4steps)
        

        train_5steps = np.load("arithmetic/5step_train.npy")
        subsample_idxs = np.random.choice(np.arange(0, len(train_5steps)), size=int(num_train_points/6), replace=False)
        train_5steps = train_5steps[subsample_idxs]
        train_5steps_skip = np.dstack([train_5steps[:, 0], train_5steps[:, 3], train_5steps[:, -1]])[0]


        train_6steps = np.load("arithmetic/6step_train.npy")
        subsample_idxs = np.random.choice(np.arange(0, len(train_6steps)), size=num_train_points//6, replace=False)
        train_6steps = train_6steps[subsample_idxs]
        train_6steps_skip = np.dstack([train_6steps[:, 0], train_6steps[:, 3], train_6steps[:, -1]])[0]

        
        train_7steps = np.load("arithmetic/7step_train.npy")
        subsample_idxs = np.random.choice(np.arange(0, len(train_7steps)), size=num_train_points//6, replace=False)
        train_7steps = train_7steps[subsample_idxs]
        train_7steps_skip = np.dstack([train_7steps[:, 0], train_7steps[:, 4], train_7steps[:, -1]])[0]
        
        ratio_full = 0.2
        num_full = int(num_train_points/3*ratio_full)
                        
        train_steps_hard = list(train_5steps[:num_full])+ list(train_5steps_skip[num_full:]) + list(train_6steps[:num_full])+ list(train_6steps_skip[num_full:])+ list(train_7steps[:num_full])+ list(train_7steps_skip[num_full:])
        train_steps = train_steps_easy + train_steps_hard
    else:
        raise Exception("Invalid data type")
    
    
    # assert(num_train_points <= len(train_steps))
    # subsample_idxs = np.random.choice(np.arange(0, len(train_steps)), size=num_train_points, replace=False)

    train_dataset = SupervisedDataset(train_steps, tokenizer=tokenizer)
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, data_collator=data_collator)

def train():
    model_name_or_path="NousResearch/Llama-2-7b-hf" 
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str)
    parser.add_argument("--num_train_points", type=int)
    args = parser.parse_args()

    data_type = args.data_type
    num_train_points = args.num_train_points
    
    project_name = "arit_fft"
    run_name = f"{data_type}{str(num_train_points)}"
    os.environ["WANDB_PROJECT"]=project_name

    
    training_args = TrainingArguments(
        num_train_epochs = 2, 
        # num_train_epochs = 1, 
        
        
        per_device_train_batch_size = 6,
        per_device_eval_batch_size = 6,
        gradient_accumulation_steps = 1,
        
        # per_device_train_batch_size = 2,
        # per_device_eval_batch_size = 2,
        # gradient_accumulation_steps = 6,
        # lr_scheduler_type = "cosine",
        # warmup_ratio = 0.03,
        # lr_scheduler_type = "constant",
        lr_scheduler_type = "linear",
        warmup_steps = 20,
        learning_rate = 5e-5,
        max_grad_norm = 2,
        optim = "adamw_torch",
        output_dir = f"ckpts/{project_name}_{run_name}",
        evaluation_strategy = "no",
        # eval_steps = 25,
        logging_strategy = "steps",
        logging_steps = 25,
        save_strategy = "no",
        # save_strategy = "epoch",
        # save_strategy = "steps",
        # save_steps = 50,
        save_only_model = True,
        report_to = "wandb",
        run_name=run_name,
        bf16 = True,
        fsdp= "full_shard auto_wrap",
        fsdp_transformer_layer_cls_to_wrap= 'LlamaDecoderLayer',
        tf32 =True,
    )
        

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=1024,
        padding_side="right",
        use_fast=False,
        add_eos_token=True,
    )

    data_module = make_supervised_data_module(data_type, num_train_points, tokenizer=tokenizer)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    
    if training_args.save_strategy == "no":
        trainer.save_state()
        trainer.save_model(output_dir=f"ckpts/{project_name}_{run_name}")


if __name__ == "__main__":
    train()
