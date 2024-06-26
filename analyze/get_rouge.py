import numpy as np
from datasets import load_dataset
import evaluate
import tqdm

name = "gsm8k_fft_full_constantlr"
ckpt = 3110


dataset = load_dataset("gsm8k", "main")
train_answers = np.array(dataset["train"]['answer'])


predicted_answers_all = np.load(f"../ckpts/{name}/checkpoint-{ckpt}/train_answers.npy")



# Load the ROUGE evaluation metric
rouge = evaluate.load('rouge')


rouge1_all = np.ones((len(train_answers), 100))*-1
rouge2_all = np.ones((len(train_answers), 100))*-1
rougeL_all = np.ones((len(train_answers), 100))*-1
rougeLsum_all = np.ones((len(train_answers), 100))*-1
for idx in tqdm.tqdm(range(len(train_answers))):
    predictions =  predicted_answers_all[idx]
    references =[train_answers[idx]]
    references = np.repeat(references, 100, axis=0)

    # Compute the ROUGE score
    results = rouge.compute(predictions=predictions, references=references, use_aggregator=False)
    
    rouge1_all[idx] = results['rouge1']
    rouge2_all[idx] = results['rouge2']
    rougeL_all[idx] = results['rougeL']
    rougeLsum_all[idx] = results['rougeLsum']
import IPython; IPython.embed()


save_dict = {}
save_dict['rouge1'] = rouge1_all
save_dict['rouge2'] = rouge2_all
save_dict['rougeL'] = rougeL_all
save_dict['rougeLsum'] = rougeLsum_all

np.save(f"../ckpts/{name}/checkpoint-{ckpt}/train_rouge.npy", save_dict)


# np.load(f"../ckpts/{name}/checkpoint-{ckpt}/train_rouge.npy", allow_pickle=True).item()