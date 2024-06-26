from vllm import LLM, SamplingParams
from datasets import load_dataset
import numpy as np
import os
import argparse
import re 
from pylatexenc.latex2text import LatexNodes2Text
from analyze.math_equivalence import is_equiv
import re

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", type=str)
parser.add_argument("--seed", type=int, default=2)
args = parser.parse_args()

ckpt_dir = args.ckpt_dir

llm = LLM(model=ckpt_dir, tokenizer = "NousResearch/Llama-2-7b-hf", tensor_parallel_size=2)  # Name or path of your model

dataset = load_dataset("gsm8k", "main")
train_questions = dataset["train"]["question"] 
train_answers = dataset["train"]['answer']

test_questions = dataset["test"]["question"]
test_answers = dataset["test"]['answer']


sampling_params = SamplingParams(
    n = 5,
    temperature=0.8,
    max_tokens=512,
    top_p=0.95,
    seed=args.seed
)

def convert_dollar_strings(text):
    dollar_pattern = re.compile(r'\$([0-9]+)')
    
    def replace_with_dollars(match):
        return f"{match.group(1)} dollars"
    
    converted_text = re.sub(dollar_pattern, replace_with_dollars, text)
    
    return converted_text

def convert_percent_strings(text):
    percent_pattern = re.compile(r'([0-9]+)\%')
    
    def replace_with_percents(match):
        return f"${match.group(1)}\\%$"
    
    converted_text = re.sub(percent_pattern, replace_with_percents, text)
    
    return converted_text

def convert_fractions_to_latex(text):
    fraction_pattern = re.compile(r'(\d+)/(\d+)')
    
    def replace_with_latex(match):
        return f"$\\frac{{{match.group(1)}}}{{{match.group(2)}}}$"
    
    converted_text = re.sub(fraction_pattern, replace_with_latex, text)
    
    return converted_text


def convert_to_MATH_style(text):
    return convert_fractions_to_latex(convert_percent_strings(convert_dollar_strings(text)))



eval_questions = [convert_to_MATH_style(question) for question in test_questions]
eval_questions = [question + "\nAnswer:" for question in eval_questions]
eval_answers = test_answers


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
    
def extract_latex(text):
    start = text.find("#### ") + len("#### ")
    return text[start:]

def answer_type_individial(output , answer):
    if output[-len(" </s>"):] == " </s>":
        output = output[: -len(" </s>")]
    if output[-len("</s>"):] == "</s>":
        output = output[: -len("</s>")]
    
    answer = extract_latex(answer) #remove_boxed(last_boxed_only_string(answer))
    output_answer = remove_boxed(last_boxed_only_string(output))

    if output_answer is not None:
        
        eqiv = is_equiv(answer, output_answer, verbose=True)

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

np.save(os.path.join(ckpt_dir, "gsm8k_test_answers5.npy"), answers_all)
np.save(os.path.join(ckpt_dir, "gsm8k_test_answer_types5.npy"), answer_types_all)
# import IPython; IPython.embed()