import datasets, json, ray, pandas, math_verify, vllm, argparse, os
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed
from transformers import AutoTokenizer
import numpy as np
from eval_utils import *
from sae_utils import add_hooks, get_intervention_hook, get_clamp_hook
from sae_lens import SAE

def load_llm(args):

    model = args.model
    args.tokenizer=model
    LLM = vllm.LLM(model=model, 
                tensor_parallel_size=args.num_gpu, 
                seed=42, 
                gpu_memory_utilization=0.8,
                tokenizer=args.tokenizer,
                max_num_seqs=32,
                trust_remote_code=False,
                enforce_eager = True)
    
    

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    sampling_params = vllm.SamplingParams(temperature=args.temperature, 
                                        max_tokens=args.max_length, 
                                        top_p=0.95,
                                        top_k=30,
                                        n=args.num_samples)    
    return LLM, tokenizer, sampling_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluating GraphArena")
    parser.add_argument('--model', type=str, default='notsaev')
    parser.add_argument('--num-gpu', type=int, default=1)
    parser.add_argument('--save-path', type=str, default='/test/eval_result')
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--max-length', type=int, default=4096)
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument('--few-shot', action='store_true', default=False)
    args = parser.parse_args()

    if args.tokenizer is None:
        args.tokenizer = args.model
    
    LLM, tokenizer, sampling_params = load_llm(args)

    dataset_path_to_eval = [
        "limo",
    ]                                                    

    #data path
    base_path = "/data"

    result = {}
    meta_data = {}

    for dataset_path_name in dataset_path_to_eval:
        dataset_path = os.path.join(base_path, dataset_path_name, "train.parquet")
        dataset = pandas.read_parquet(dataset_path)



        prompt = [d[0] for d in dataset["prompt"]]
        answer = [d["ground_truth"] for d in dataset["reward_model"]]
        cot = [d["answer"] for d in dataset["extra_info"]]


        

        log_dict = eval_single_dataset(LLM, tokenizer, sampling_params, prompt, answer,cot, args)


        ## Calculate avg and std
        acc_trials = [0 for i in range(0, args.num_samples)]
        for sample in log_dict:
            for i in range(args.num_samples):
                acc_trials[i] += sample["label"][i]

        acc_trials = [acc/len(log_dict) for acc in acc_trials]
        acc = np.mean(acc_trials).item()
        std = np.std(acc_trials).item()


        result[dataset_path_name] = log_dict
        meta_data[dataset_path_name] = {
            "total": len(prompt),
            "acc_trials": acc_trials,
            "acc": acc,
            "std": std
        }
    
    result["meta_data"] = meta_data

    save_path = os.path.join(args.save_path, args.model.replace('/', '-'))
    os.makedirs(save_path, exist_ok=True)

    if args.few_shot:
        save_path = os.path.join(save_path, "eval_math_result_few_shot.json")
    else:
        save_path = os.path.join(save_path, "eval_math_result.json")

    with open(save_path, 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    print(meta_data)
