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
parser.add_argument("--num_devices", type=int, default=4)


args = parser.parse_args()

ckpt_dir = args.ckpt_dir

llm = LLM(model=ckpt_dir, tokenizer = "NousResearch/Llama-2-7b-hf", tensor_parallel_size=args.num_devices)  # Name or path of your model


def get_aug_answer(full_answer):
    idx = full_answer.rfind("= ")
    if idx == -1:
        return None
    else:
        answer = full_answer[idx + len("= "):]
        return answer

def answer_type_individial(output , answer):
    if output[-len(" </s>"):] == " </s>":
        output = output[: -len(" </s>")]
    if output[-len("</s>"):] == "</s>":
        output = output[: -len("</s>")]
    
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



sampling_params = SamplingParams(
    n = 5,
    temperature=0.8,
    max_tokens=512,
    top_p=0.95,
    seed=args.seed
)



def generate_output(train_examples, save_name):
    eval_questions = [train_example[0] + " = " for train_example in train_examples]
    eval_answers = train_examples[:, -1]

    output = llm.generate(eval_questions, sampling_params)



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
    print(save_name)
    print((answer_types_all==0).mean(axis=-1).mean())
    print((answer_types_all==1).mean(axis=-1).mean())
    print((answer_types_all==2).mean(axis=-1).mean())


    np.save(os.path.join(ckpt_dir, f"{save_name}_answers5.npy"), answers_all)
    np.save(os.path.join(ckpt_dir, f"{save_name}_answer_types5.npy"), answer_types_all)
    
    
train_examples = np.load("arithmetic/2step_eval.npy")[:500]
generate_output(train_examples, "2step_eval")
train_examples = np.load("arithmetic/3step_eval.npy")[:500]
generate_output(train_examples, "3step_eval")
train_examples = np.load("arithmetic/4step_eval.npy")[:500]
generate_output(train_examples, "4step_eval")
train_examples = np.load("arithmetic/5step_eval.npy")[:500]
generate_output(train_examples, "5step_eval")
train_examples = np.load("arithmetic/6step_eval.npy")[:500]
generate_output(train_examples, "6step_eval")
train_examples = np.load("arithmetic/7step_eval.npy")[:500]
generate_output(train_examples, "7step_eval")



