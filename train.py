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


        num_steps = [answer.count('\n') for answer in train_answers]

        sources = [ 
            question+ "\nAnswer:"
            for question in train_questions
        ]
        # targets = [" Num steps: "+str(num_steps[i])+"\n"+train_answers[i] for i in range(len(train_answers))]
        targets =[" "+train_answers[i] for i in range(len(train_answers))] 
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



def make_supervised_data_module(subsample_type, subsample_numtraincorrect, tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset = load_dataset("gsm8k", "main")
    
    # checkpoints = ["00500", "05000", "10000", "15000", "20000", "45000", "50000"]
    # num_correct_all = []
    # for checkpoint in checkpoints:
    #     model_path = f"ckpts/sft_gsm8k_llama7B_full3/checkpoint_{checkpoint}/hf_model/"        
    #     num_correct = (np.load(os.path.join(model_path, "train_answer_types_16.npy"))==0).sum(axis=-1)/16
    #     num_correct_all.append(num_correct)
    # num_correct_all = np.array(num_correct_all)
    # num_correct_all = np.mean(num_correct_all, axis=0)
    # sorted_idxs = np.argsort(num_correct_all)
    # # low acc to high acc
    
    
    # full_rouges = []
    # for checkpoint in [311, 934, 1557, 3110]:
    #     rouge_dict = np.load(f"ckpts/gsm8k_fft_full_constantlr/checkpoint-{str(checkpoint)}/train_rouge.npy", allow_pickle=True).item()
    #     full_rouges.append(rouge_dict["rougeL"])
    # full_rouges = np.array(full_rouges)
    # rouge_score = full_rouges.mean(axis=-1).mean(axis=0)
    # sorted_idxs = np.argsort(rouge_score)
    

    
    full_correct = []
    for checkpoint in [311, 1557, 3110]:
        model_path = f"ckpts/gsm8k_fft_full_constantlr/checkpoint-{str(checkpoint)}/"        
        correct = (np.load(os.path.join(model_path, "train_answer_types.npy"))==0)
        full_correct.append(correct)
    full_correct = np.array(full_correct)
    # 4, 7473, 100
    
    # sorted_idxs = np.argsort(full_correct.mean(axis=-1).mean(axis=0))
    
    
    # correct_no_memorize = np.logical_and(full_correct, full_rouges<0.9).mean(axis=-1).mean(axis=0)    
    # sorted_idxs = np.argsort(correct_no_memorize)
    
    # subsample_idxs = sorted_idxs[len(sorted_idxs)//2:]
    # subsample_idxs1 = np.random.choice(sorted_idxs, len(sorted_idxs)//4, replace=False)
    # subsample_idxs2 = np.random.choice(sorted_idxs[len(sorted_idxs)//2:], len(sorted_idxs)//4, replace=False)
    # subsample_idxs = np.concatenate([subsample_idxs1, subsample_idxs2])
    
    subsample_idxs = np.where(full_correct.sum(axis=-1).mean(axis=0)>= subsample_numtraincorrect)[0]
    
    if subsample_type == "rand":
        subsample_idxs = np.random.choice(np.arange(0, len(dataset["train"]["question"])), size=len(subsample_idxs))
    elif subsample_type == "hard":
        print("hard mode")
        sorted_idxs = np.argsort(full_correct.sum(axis=-1).mean(axis=0))
        subsample_idxs = sorted_idxs[:len(subsample_idxs)]
    np.random.shuffle(subsample_idxs)
    
    train_questions = np.array(dataset["train"]["question"])[subsample_idxs]
    train_answers = np.array(dataset["train"]['answer'])[subsample_idxs]
    
    
    
    
    
    # path = "ckpts/gsm8k_fft_full_constantlr/checkpoint-311"
    # train_generated_answers = np.load(path+"/train_answers.npy")
    # train_answers_correct = np.load(path+"/train_answer_types.npy")==0

    # train_subsample_idxs = []
    # train_generated_answers_subsampled = []
    # for example_idx in range(len(train_generated_answers)):
    #     if train_answers_correct[example_idx].sum() > 0:
    #         subsample_idx = np.random.choice(np.where(train_answers_correct[example_idx])[0])
    #         train_subsample_idxs.append(example_idx)
    #         train_generated_answers_subsampled.append(train_generated_answers[example_idx][subsample_idx])
    
    # train_questions = np.array(dataset["train"]["question"])[train_subsample_idxs]
    # train_answers = np.array(dataset["train"]['answer'])[train_subsample_idxs]
    # # train_answers = train_generated_answers_subsampled
    
    train_dataset = SupervisedDataset(train_questions, train_answers, tokenizer=tokenizer)
    
    test_questions = dataset["test"]["question"][:500]
    test_answers = dataset["test"]['answer'][:500]
    test_dataset = SupervisedDataset(test_questions, test_answers, tokenizer=tokenizer)
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=test_dataset, data_collator=data_collator)

def train():
    model_name_or_path="NousResearch/Llama-2-7b-hf"
    # model_name_or_path = "ckpts/gsm8k_fft_easy3_50_1epoch"
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample_type", type=str)
    parser.add_argument("--subsample_numtraincorrect", type=int)
    args = parser.parse_args()

    subsample_type = args.subsample_type
    subsample_numtraincorrect = args.subsample_numtraincorrect
    
    project_name = "gsm8k_fft"
    # run_name = "rouge_high50"
    # run_name = "subsampled_ground_truth"
    run_name  = f"{subsample_type}{str(subsample_numtraincorrect)}+"
    os.environ["WANDB_PROJECT"]=project_name

    
    training_args = TrainingArguments(
        num_train_epochs = 2, 
        # num_train_epochs = 1, 
        # per_device_train_batch_size = 4,
        # per_device_eval_batch_size = 4,
        # gradient_accumulation_steps = 3,
        per_device_train_batch_size = 3,
        per_device_eval_batch_size = 3,
        gradient_accumulation_steps = 2,
        # lr_scheduler_type = "cosine",
        # warmup_ratio = 0.03,
        # lr_scheduler_type = "constant",
        lr_scheduler_type = "linear",
        warmup_steps = 20,
        learning_rate = 5e-5,
        max_grad_norm = 2,
        optim = "adamw_torch",
        output_dir = f"ckpts/{project_name}_{run_name}",
        evaluation_strategy = "steps",
        eval_steps = 25,
        logging_strategy = "steps",
        logging_steps = 25,
        save_strategy = "no",
        # save_strategy = "epoch",
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

    data_module = make_supervised_data_module(subsample_type, subsample_numtraincorrect, tokenizer=tokenizer)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    
    if training_args.save_strategy == "no":
        trainer.save_state()
        trainer.save_model(output_dir=f"ckpts/{project_name}_{run_name}")


if __name__ == "__main__":
    train()
