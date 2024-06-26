from vllm import LLM, SamplingParams
from datasets import load_dataset
import numpy as np
import os
import argparse

# This Python script seems to be evaluating a language model on a dataset of questions and answers.
# Here's a breakdown of what the script is doing:

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", type=str)
parser.add_argument("--seed", type=int, default=2)
args = parser.parse_args()



ckpt_dir = args.ckpt_dir


llm = LLM(model=ckpt_dir, tokenizer = "NousResearch/Llama-2-7b-hf", tensor_parallel_size=4)  # Name or path of your model

dataset = load_dataset("gsm8k", "main")
train_questions = dataset["train"]["question"] 
train_answers = dataset["train"]['answer']

test_questions = dataset["test"]["question"]
test_answers = dataset["test"]['answer']


sampling_params = SamplingParams(
    n = 10,
    temperature=0.8,
    max_tokens=512,
    top_p=0.95,
    seed=args.seed
)


eval_questions = test_questions
eval_questions = [question + "\nAnswer:" for question in eval_questions]
eval_answers = test_answers


output = llm.generate(eval_questions, sampling_params)

def extract_latex(text):
    start = text.find("#### ") + len("#### ")
    return text[start:]
    
def answer_type_individial(output , answer):
    if output[-len(" </s>"):] == " </s>":
        output = output[: -len(" </s>")]
    if output[-len("</s>"):] == "</s>":
        output = output[: -len("</s>")]
        
    answer = extract_latex(answer)
    
    output_answer_start_idx = output.find("#### ")
    if output_answer_start_idx != -1:
        output = output[output_answer_start_idx+len("#### "):]
        if output.replace(",", "") == answer.replace(",", ""):
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


np.save(os.path.join(ckpt_dir, "test_answers_gsm8k.npy"), answers_all)
np.save(os.path.join(ckpt_dir, "test_answer_types_gsm8k.npy"), answer_types_all)
# import IPython; IPython.embed()