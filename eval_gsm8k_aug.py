from vllm import LLM, SamplingParams
from datasets import load_dataset
import numpy as np
import os
import argparse
import re 
from pylatexenc.latex2text import LatexNodes2Text
from analyze.math_equivalence import is_equiv
import json

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", type=str)
parser.add_argument("--seed", type=int, default=2)
parser.add_argument("--eval_type", type=str, default="test")
parser.add_argument("--num_devices", type=int, default=4)


args = parser.parse_args()

ckpt_dir = args.ckpt_dir

llm = LLM(model=ckpt_dir, tokenizer = "NousResearch/Llama-2-7b-hf", tensor_parallel_size=args.num_devices)  # Name or path of your model

sampling_params = SamplingParams(
    n = 5,
    temperature=0.8,
    max_tokens=512,
    top_p=0.95,
    seed=args.seed
)

if args.eval_type == "test":
    dataset = load_dataset("gsm8k", "main")

    test_questions = dataset["test"]["question"]
    test_answers = dataset["test"]['answer']
    eval_questions = test_questions
    eval_questions = [question + "\nAnswer:" for question in eval_questions]
    eval_answers = test_answers
elif args.eval_type == "train1000":
    with open('GSM8K_AUG/AugGSM8K_part1.jsonl', 'r') as json_file:
        json_list = list(json_file)

    # with open('MATH_aug/AugMATH_part2.jsonl', 'r') as json_file:
    #     json_list += list(json_file)

    train_questions = []
    train_answers = []
    for json_str in json_list:
        result = json.loads(json_str)
        train_questions.append(result["query"])
        train_answers.append(result["response"])
        
    train_questions = np.array(train_questions)
    train_answers = np.array(train_answers)
    eval_questions = train_questions[:1000]
    eval_questions = [question + "\nAnswer:" for question in eval_questions]
    eval_answers = train_answers[:1000]
elif args.eval_type == "train_part1":
    with open('GSM8K_AUG/AugGSM8K_part1.jsonl', 'r') as json_file:
        json_list = list(json_file)
    train_questions = []
    train_answers = []
    for json_str in json_list:
        result = json.loads(json_str)
        train_questions.append(result["query"])
        train_answers.append(result["response"])
        
    train_questions = np.array(train_questions)
    train_answers = np.array(train_answers)
    eval_questions = train_questions
    eval_questions = [question + "\nAnswer:" for question in eval_questions]
    eval_answers = train_answers
elif args.eval_type == "train_part2":
    with open('GSM8K_AUG/AugGSM8K_part2.jsonl', 'r') as json_file:
        json_list = list(json_file)

    train_questions = []
    train_answers = []
    for json_str in json_list:
        result = json.loads(json_str)
        train_questions.append(result["query"])
        train_answers.append(result["response"])
    train_questions = np.array(train_questions)
    train_answers = np.array(train_answers)
    eval_questions = train_questions
    eval_questions = [question + "\nAnswer:" for question in eval_questions]
    eval_answers = train_answers

output = llm.generate(eval_questions, sampling_params)
    

def get_aug_answer(full_answer):
    idx = full_answer.rfind("The answer is")
    if idx == -1:
        return None
    else:
        answer = full_answer[idx + len("The answer is: "):]
        answer = answer.replace(":", "").replace("$", "").strip()
        if len(answer)> 0:
            if answer[-1] == ".":
                answer = answer[:-1]
            left = "\\boxed{"
            if answer[:len(left)] == left and answer[-1] == "}":
                answer = answer[len(left):-1]
        return answer

def extract_latex(text):
    start = text.find("#### ") + len("#### ")
    return text[start:]

def answer_type_individial(output , answer):
    if output[-len(" </s>"):] == " </s>":
        output = output[: -len(" </s>")]
    if output[-len("</s>"):] == "</s>":
        output = output[: -len("</s>")]
    
    if args.eval_type == "train1000" or args.eval_type == "train_part1" or args.eval_type == "train_part2":
        answer = get_aug_answer(answer)
    else:
        answer = extract_latex(answer)
    output_answer = get_aug_answer(output)

    if output_answer is not None and answer is not None:
        
        eqiv = is_equiv(answer, output_answer, verbose=False)

        if eqiv:
            answer_type = 0
        else:
            answer_type = 1
    else:
        answer_type = 2
    return answer_type


answer_types_all = []
# answers_all = []
for i in range(len(output)):
    answer_types = []
    answers = []
    for item in output[i].outputs:
        answers.append(item.text)
        answer_type = answer_type_individial(item.text, eval_answers[i])
        answer_types.append(answer_type)
    answer_types_all.append(answer_types)
    # answers_all.append(answers)

answer_types_all = np.array(answer_types_all)
# answers_all = np.array(answers_all)
print((answer_types_all==0).mean(axis=-1).mean())
print((answer_types_all==1).mean(axis=-1).mean())
print((answer_types_all==2).mean(axis=-1).mean())


# np.save(os.path.join(ckpt_dir, f"{args.eval_type}_answers5.npy"), answers_all)
np.save(os.path.join(ckpt_dir, f"{args.eval_type}_answer_types5_seed{args.seed}.npy"), answer_types_all)
# import IPython; IPython.embed()