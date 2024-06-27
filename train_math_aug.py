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


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, train_questions, train_answers, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        sources = [ 
            question+ "\nAnswer:"
            for question in train_questions
        ]
        targets =[train_answers[i] for i in range(len(train_answers))] 
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


def make_supervised_data_module(data_type, num_train_points, output_dir, tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    
    with open('MATH_aug/AugMATH_part1.jsonl', 'r') as json_file:
        json_list = list(json_file)

    with open('MATH_aug/AugMATH_part2.jsonl', 'r') as json_file:
        json_list += list(json_file)

    train_questions = []
    train_answers = []
    for json_str in json_list:
        result = json.loads(json_str)
        train_questions.append(result["query"])
        train_answers.append(result["response"])
        
    train_questions = np.array(train_questions)
    train_answers = np.array(train_answers)
    
    # train_dataset = SupervisedDataset(train_questions, train_answers, tokenizer=tokenizer)
    
    # subsample_idxs = np.random.choice(np.arange(0, len(train_questions)), size=len(train_questions)//2)
    
    train_part1_accs = (np.load("MATH_aug/train_part1_answer_types5.npy")==0).sum(axis=-1)
    train_part2_accs = (np.load("MATH_aug/train_part2_answer_types5.npy")==0).sum(axis=-1)
    train_accs = np.concatenate([train_part1_accs, train_part2_accs])
    
    if data_type == "hard": 
        subsample_idxs = np.where(train_accs==0)[0]
        # subsample_idxs = np.load("MATH_aug/hard_50000idxs.npy")
        print(len(subsample_idxs))
    elif data_type == "easy":
        subsample_idxs = np.where(train_accs>=3)[0]
        print(len(subsample_idxs))
    elif data_type == "rand":
        subsample_idxs = np.where(train_accs>=0)[0]
        # subsample_idxs = np.load("MATH_aug/rand_50000idxs.npy")
        print(len(subsample_idxs))
    elif data_type == "medium":
        subsample_idxs = np.where((train_accs>0)*(train_accs<3))[0]
        print(len(subsample_idxs))
    elif data_type == "easymedium":
        subsample_idxs = np.where(train_accs>0)[0]
        print(len(subsample_idxs))
    else:
        raise Exception("Invalid data type")
    
    
    assert(num_train_points <= len(subsample_idxs))
    subsample_idxs = np.random.choice(subsample_idxs, size=num_train_points, replace=False)
    
    
    # make directory for outputdir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    np.save(output_dir+"subsample_idxs.npy", subsample_idxs)

    train_dataset = SupervisedDataset(train_questions[subsample_idxs], train_answers[subsample_idxs], tokenizer=tokenizer)
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, data_collator=data_collator)

def train():
    model_name_or_path="NousResearch/Llama-2-7b-hf" 
    
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str)
    parser.add_argument("--num_train_points", type=int)
    parser.add_argument("--num_epochs", type=int)
    args = parser.parse_args()

    data_type = args.data_type
    num_train_points = args.num_train_points
    num_epochs = args.num_epochs
    
    project_name = "math_aug_fft"
    run_name = f"{data_type}{str(num_train_points)}_{str(num_epochs)}epochs"
    os.environ["WANDB_PROJECT"]=project_name
    
    
    output_dir = f"ckpts/{project_name}_{run_name}/"

    
    training_args = TrainingArguments(
        num_train_epochs = num_epochs, 
        # num_train_epochs = 2,        
        
        per_device_train_batch_size = 3,
        per_device_eval_batch_size = 3,
        gradient_accumulation_steps = 2,
        
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
        output_dir = output_dir,
        evaluation_strategy = "no",
        # eval_steps = 25,
        logging_strategy = "steps",
        logging_steps = 25,
        save_strategy = "no",
        # save_strategy = "epoch",
        # save_strategy = "steps",
        # save_steps = 1000,
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

    data_module = make_supervised_data_module(data_type, num_train_points, output_dir, tokenizer=tokenizer)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    
    # if training_args.save_strategy == "no":
    trainer.save_state()
    trainer.save_model(output_dir=output_dir)


if __name__ == "__main__":
    train()
