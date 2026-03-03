import datasets, json, ray, pandas, math_verify, vllm, argparse, os
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed
from transformers import AutoTokenizer
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from sae_utils import add_hooks, get_intervention_hook
from sae_lens import SAE

def extract_response_and_answer(json_file_path):
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    extracted_data = []
    
    for i, item in enumerate(data['math'], 1):
        extracted_item = {
            'question_id': i,
            'prompt': item['prompt'],
            'answer': item['answer'],
            'response': item['response']
        }
        extracted_data.append(extracted_item)
    
    return extracted_data






def verify_with_model(extracted_ans, target_answer, model, tokenizer):
    model.eval()
    
    prompt=f"<|start_header_id|>system<|end_header_id|>\n\nYou are a math expert.<|eot_id|><|start_header_id|>user<|end_header_id|>I have an answer to a problem that needs verification. The answer may involve complexity, and you should disregard the loss of units (if any) in both answers. My answer is {extracted_ans}, and the correct answer is {target_answer}. Please tell me whether my answer is correct or not in one word: 'Correct' or 'Incorrect'.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    source_text_res = tokenizer.encode_plus(
            prompt, max_length=4096, truncation=True, add_special_tokens=False
        )
    with torch.no_grad():
        outputs = model.generate(
                    torch.tensor([source_text_res['input_ids']]).to(model.device),
                    attention_mask=torch.tensor([source_text_res["attention_mask"]]).to(model.device),
                    max_new_tokens=128,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False
                )
    decoded_output = tokenizer.batch_decode(
                    outputs,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
    result = decoded_output[0].split("<|end_header_id|>\n\n")[-1]
    if "incorrect" in result.lower():
        print(result, "extracted_ans", extracted_ans, "target_answer", target_answer)
        return False
    elif "correct" in result.lower():
        print(result, "extracted_ans", extracted_ans, "target_answer", target_answer)
        return True
    else:
        print("Strange Verify Result", result, "extracted_ans", extracted_ans, "target_answer", target_answer)
        return False

def compare_parse(predict, gold):
    if len(predict) == 0 or len(gold) == 0:
        return False
    
    label = math_verify.verify(gold[0], predict[0], strict=False)

    if len(predict) == 2:
        label = label or math_verify.verify(gold[0], predict[1], strict=False)

        if len(gold) == 2:
            label = label or math_verify.verify(gold[1], predict[1], strict=False)
    else:
        if len(gold) == 2:
            label = label or math_verify.verify(gold[1], predict[0], strict=False)
    
    return label

def math_equal(predict, gold):
    if not predict:
        return False, predict, gold
    try:

        predict_parse= math_verify.parse(f'${predict}$')
        gold_parse = math_verify.parse(f'${str(gold)}$')

        label = compare_parse(predict_parse, gold_parse)

        return label, predict_parse, gold_parse
    except:
        return False, predict, gold

def extract_answer(solution_str):
    try:
        return remove_boxed(last_boxed_only_string(solution_str))
    except:
        try:
            return solution_str.split("#### ")[1]
        except:
            return ""

def check_output_correct(response, gold,model,tokenizer):
    predict = extract_answer(response)
    #label, predict, gold = math_equal(predict, gold)
    label = verify_with_model(response, gold, model, tokenizer)
    return label, predict, gold


def eval_single_dataset(LLM:vllm.LLM, tokenizer, sampling_params, prompts, answers,cots, args):
    outputs = []
    #extract features with/without In-context learning
    icl = False
    if args.few_shot:
        test_cases = [
            tokenizer.apply_chat_template(few_shot_prompt + [prompt], tokenize=False,
            add_generation_prompt=True, 
            ) for prompt in prompts
        ]
    elif icl==True:
 
       
        few_shot_prompt = [prompts[30]]+[{"content":cots[30],"role":"assistant"}]+[prompts[31]]+[{"content":cots[31],"role":"assistant"}]




        test_cases = [
            tokenizer.apply_chat_template(few_shot_prompt + [prompt], tokenize=False,
            add_generation_prompt=True, 
            ) for prompt in prompts
        ]       
    else:
        test_cases = [
            tokenizer.apply_chat_template([prompt], tokenize=False,
            add_generation_prompt=True, 
            ) for prompt in prompts
        ]
    


    sae = SAE.load_from_disk("llama-3-8b-it-res/blocks.25.hook_resid_post")
    
    lm_model = LLM.llm_engine.model_executor.driver_worker.model_runner.model
    sae_hooks = []
    feature_idx = 1
    strength = 1
    max_activation = 1.0
    sae_hooks.append(
        (
            lm_model.model.layers[sae.cfg.hook_layer],
            get_intervention_hook(sae,feature_idx,max_activation,strength)
        )
    )

    with add_hooks([], sae_hooks):
         outputs = LLM.generate(test_cases, sampling_params)
    
    # outputs = LLM.generate(test_cases,sampling_params)

    model_name = '/model/meta-llama/Llama-3.1-8B-Instruct'
    judger_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    judger_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )





    log_dict = []



    for i, (output, answer) in enumerate(zip(outputs, answers)): 
        if True:
            log_dict.append({
                "prompt": prompts[i]["content"],
                "answer": str(answer),
                "label": [],
                "response": [],
                "predict": []
            })

        for j in range(0, args.num_samples):
            response_text = output.outputs[j].text
            label, predict_parse, answer_parse = check_output_correct(response_text, answer, judger_model, judger_tokenizer)

            log_dict[i]["response"].append(response_text)
            log_dict[i]["label"].append(float(label))
            log_dict[i]["predict"].append(str(predict_parse))
    return log_dict


if __name__ == '__main__':
    print("test")
