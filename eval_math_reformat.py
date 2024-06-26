from vllm import LLM, SamplingParams
from datasets import load_dataset
import numpy as np
import os
import argparse
import re 
from pylatexenc.latex2text import LatexNodes2Text

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", type=str)
parser.add_argument("--seed", type=int, default=2)
args = parser.parse_args()

ckpt_dir = args.ckpt_dir

llm = LLM(model=ckpt_dir, tokenizer = "NousResearch/Llama-2-7b-hf", tensor_parallel_size=2)  # Name or path of your model

dataset = load_dataset("hendrycks/competition_math")
train_questions = np.array(dataset["train"]["problem"])
train_answers = np.array(dataset["train"]['solution'])

test_questions = dataset["test"]["problem"]
test_answers = dataset["test"]['solution']


sampling_params = SamplingParams(
    n = 1,
    temperature=0.8,
    max_tokens=512,
    top_p=0.95,
    seed=args.seed
)



def reformat_final_answer(response):
    try:
        start1 = response.rindex("\\boxed{")
        start2 = start1 + len("\\boxed{")
        end = response.rindex("}")
    except:
        print(response)
        start1 = response.rindex("\\boxed")
        start2 = start1 + len("\\boxed")
        end = response.rindex("$")
    new_response = response[:end] + "####" + response[end+1:]
    new_response = new_response[:start1] + "####" + new_response[start2:]
    return new_response

def reformat_question(question):
    return (re.sub(' +', ' ', LatexNodes2Text().latex_to_text((question))))

def reformat_response(response):
    return (re.sub(' +', ' ', LatexNodes2Text().latex_to_text(reformat_final_answer(response))))


# eval_questions = list(map(reformat_question, test_questions))
eval_questions = test_questions
eval_questions = [question + "\nAnswer:" for question in eval_questions]
# eval_answers = list(map(reformat_response, test_answers))
eval_answers = test_answers


output = llm.generate(eval_questions, sampling_params)


def extract_latex(text):
    start = text.rindex("\\boxed{")+len("\\boxed{")
    end = text.rindex("}")
    return text[start:end]

# def extract_latex(text):
#     start = text.rindex("####")+len("####")
#     end = text.rindex("####")
#     return text[start:end]
    
def answer_type_individial(output , answer):
    if output[-len(" </s>"):] == " </s>":
        output = output[: -len(" </s>")]
    if output[-len("</s>"):] == "</s>":
        output = output[: -len("</s>")]
    
    answer = extract_latex(answer)
    output_answer_start_idx = output.find("\\boxed{")
    output_answer_end_idx = output.find("}", output_answer_start_idx+len("\\boxed{"))
    # output_answer_start_idx = output.find("####")
    # output_answer_end_idx = output.find("####", output_answer_start_idx+len("####"))
    if output_answer_start_idx != -1 and output_answer_end_idx != -1:
        output = output[output_answer_start_idx+len("\\boxed{"):output_answer_end_idx]
        # output = output[output_answer_start_idx+len("####"):output_answer_end_idx]

        if output == answer:
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


np.save(os.path.join(ckpt_dir, "test_answers.npy"), answers_all)
np.save(os.path.join(ckpt_dir, "test_answer_types.npy"), answer_types_all)
# import IPython; IPython.embed()