from vllm import LLM, SamplingParams
from datasets import load_dataset
import numpy as np
import os
import argparse
import re 
from pylatexenc.latex2text import LatexNodes2Text
from analyze.math_equivalence import is_equiv

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", type=str)
parser.add_argument("--seed", type=int, default=2)
parser.add_argument("--eval_type", type=str, default="test")

args = parser.parse_args()

ckpt_dir = args.ckpt_dir

llm = LLM(model=ckpt_dir, tokenizer = "NousResearch/Llama-2-7b-hf", tensor_parallel_size=4)  # Name or path of your model

dataset = load_dataset("hendrycks/competition_math")
train_questions = np.array(dataset["train"]["problem"])
train_answers = np.array(dataset["train"]['solution'])

test_questions = dataset["test"]["problem"]
test_answers = dataset["test"]['solution']


sampling_params = SamplingParams(
    n = 5,
    temperature=0.8,
    max_tokens=512,
    top_p=0.95,
    seed=args.seed
)

if args.eval_type == "test":
    eval_questions = test_questions
    eval_questions = [question + "\nAnswer:" for question in eval_questions]
    eval_answers = test_answers
elif args.eval_type == "train1000":
    eval_questions = train_questions[:1000]
    eval_questions = [question + "\nAnswer:" for question in eval_questions]
    eval_answers = train_answers[:1000]

output = llm.generate(eval_questions, sampling_params)

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def answer_type_individial(output , answer):
    if output[-len(" </s>"):] == " </s>":
        output = output[: -len(" </s>")]
    if output[-len("</s>"):] == "</s>":
        output = output[: -len("</s>")]
    
    answer = remove_boxed(last_boxed_only_string(answer))
    output_answer = remove_boxed(last_boxed_only_string(output))

    if output_answer is not None:
        
        eqiv = is_equiv(answer, output_answer, verbose=False)

        if eqiv:
            answer_type = 0
        else:
            answer_type = 1
    else:
        answer_type = 2
    return answer_type


answer_types_all = []
answers_all = []
for i in range(len(output)):
    answer_types = []
    answers = []
    for item in output[i].outputs:
        answers.append(item.text)
        answer_type = answer_type_individial(item.text, eval_answers[i])
        answer_types.append(answer_type)
    answer_types_all.append(answer_types)
    answers_all.append(answers)

answer_types_all = np.array(answer_types_all)
answers_all = np.array(answers_all)
print((answer_types_all==0).mean(axis=-1).mean())
print((answer_types_all==1).mean(axis=-1).mean())
print((answer_types_all==2).mean(axis=-1).mean())


np.save(os.path.join(ckpt_dir, f"{args.eval_type}_answers5.npy"), answers_all)
np.save(os.path.join(ckpt_dir, f"{args.eval_type}_answer_types5.npy"), answer_types_all)
# import IPython; IPython.embed()