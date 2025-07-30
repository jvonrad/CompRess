import argparse
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

tasks = ['sst2', 'qqp', 'mnli', 'qnli', 'mnli-mm', 'rte']

task_to_keys = {
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
}


def main(args):
    with open(args.data_file) as f:
        dataset = json.load(f)
    model, tokenizer = load_model_tokenizer(args)
    eval(model, tokenizer, dataset, args)

def load_model_tokenizer(args):
    # Debug print to see what's being loaded
    print(f"Loading model and tokenizer from: {args.path}")
    
    try:
        # For LLaMA models, you might need to specify the tokenizer explicitly
        # Try loading with padding token setup
        tokenizer = AutoTokenizer.from_pretrained(
            args.path,
            use_fast=False,
            trust_remote_code=True,
            padding_side='left'  # LLaMA models typically use left padding
        )
        
        # Set pad token if not set (common issue with LLaMA tokenizers)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print(f"Tokenizer loaded successfully: {type(tokenizer)}")
        
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        # Try alternative loading method
        try:
            # If the tokenizer isn't in the model directory, try loading from the base model
            tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Meta-Llama-3-8B",  # Base model name
                use_fast=False,
                trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print("Loaded tokenizer from base model")
        except:
            raise ValueError(f"Could not load tokenizer from {args.path}")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16  # Use float16 for efficiency
    )
    
    print(f"Model loaded successfully: {type(model)}")
    
    return model, tokenizer

def format_example(task_name, question, origin=False, k=5):
    prompt = ""
    for i in range(k):
        if task_name in ["mnli","mnli-mm"]:
            prompt += gen_prompt(task_name, question[i], origin=origin)
            if question[i]['label'] == 0:
                prompt += " A. yes\n\n"
            elif question[i]['label'] == 1:
                prompt += " B. maybe\n\n"
            else:
                prompt += " C. no\n\n"
        elif task_name in  ['qnli','rte','qqp']:
            prompt += gen_prompt(task_name, question[i], origin=origin)
            
            if question[i]['label'] == 0:
                prompt += " A. yes\n\n"
            else:
                prompt += " B. no\n\n"
        elif task_name =='sst2':
            prompt += gen_prompt(task_name, question[i], origin=origin)
            
            if question[i]['label'] == 0:
                prompt += " A. positive\n\n"
            else:
                prompt += " B. negative\n\n"
        else:
            raise ValueError("Unsupported task:", task_name)
    return prompt

def gen_prompt(task_name, question, origin=False):
    if task_name == "mnli":
        prompt = "Please identify whether the premise entails the hypothesis. The answer should be exactly 'A. yes', 'B. maybe' or 'C. no'\n"
        if origin and 'original_premise' in question.keys():
            prompt += "Premise: " + question['original_premise']
        else:
            prompt += "Premise: " + question['premise']
        prompt += "\nHypothesis: " + question['hypothesis']
        prompt += "\nAnswer:"
    elif task_name == "mnli-mm":
        prompt = "Please identify whether the premise entails the hypothesis. The answer should be exactly 'A. yes', 'B. maybe' or 'C. no'\n"
        prompt += "Premise: " + question['premise']
        if origin and 'original_hypothesis' in question.keys():
            prompt += "\nHypothesis: " + question['original_hypothesis']
        else:
            prompt += "\nHypothesis: " + question['hypothesis']
        prompt += "\nAnswer:"
    elif task_name == 'qnli':
        prompt = "Please identify whether the sentence answers the question. The answer should be exactly 'A. yes' or 'B. no'\n"
        if origin and 'original_question' in question.keys():
            prompt += "Question: " + question['original_question']
        else:
            prompt += "Question: " + question['question']
        prompt += "\nSentence: " + question['sentence']
        prompt += "\nAnswer:"
    elif task_name == 'rte':
        prompt = "Please identify whether the sentence1 entails the sentence2. The answer should be exactly 'A. yes' or 'B. no'\n"
        if origin and 'original_sentence1' in question.keys():
            prompt += "Sentence 1: " + question['original_sentence1']
        else:
            prompt += "Sentence 1: " + question['sentence1']
        prompt += "\nSentence 2: " + question['sentence2']
        prompt += "\nAnswer:"
    elif task_name == 'qqp':
        prompt = "Please identify whether Question 1 has the same meaning as Question 2. The answer should be exactly 'A. yes' or 'B. no'\n"
        if origin and 'original_question1' in question.keys():
            prompt += "Question 1: " + question['original_question1']
        else:
            prompt += "Question 1: " + question['question1']
        prompt += "\nQuestion 2: " + question['question2']
        prompt += "\nAnswer:"
    elif task_name =='sst2':
        prompt = "For each snippet of text, label the sentiment of the text as positive or negative. The answer should be exactly 'A. positive' or 'B. negative'\n"
        if origin and 'original_sentence' in question.keys():
            prompt += "Sentence: " + question['original_sentence']
        else:
            prompt += "Sentence: " + question['sentence']
        prompt += "\nAnswer:"
    else:
        raise ValueError("Unsupported task:", task_name)
    
    return prompt


def eval(model, tokenizer, dataset, args):
    cors = []
    for task_name in tasks:
        print(f"\nEvaluating task: {task_name}")
        task_cors = []
        test = dataset[task_name]
        
        # Determine the range based on ntrain
        start_idx = args.ntrain
        end_idx = len(test)
        
        for i in range(start_idx, end_idx):
            prompt_end = gen_prompt(task_name, test[i], origin=args.test_origin)
            
            # Handle zero-shot case
            if args.ntrain == 0:
                prompt = prompt_end
            else:
                example = format_example(task_name, test, origin=args.test_origin, k=args.ntrain)
                prompt = example + prompt_end
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs["input_ids"].to(model.device)
            
            # Get model predictions
            with torch.no_grad():
                outputs = model(input_ids=input_ids)
                logits = outputs.logits[:, -1, :].flatten()
            
            label = test[i]["label"]
            
            # Get token IDs for answer options
            token_A = tokenizer.encode("A", add_special_tokens=False)[-1]
            token_B = tokenizer.encode("B", add_special_tokens=False)[-1]
            token_C = tokenizer.encode("C", add_special_tokens=False)[-1] if task_name in ["mnli", "mnli-mm"] else None
            
            if task_name in ["mnli", "mnli-mm"]:
                # 3-way classification
                probs = torch.nn.functional.softmax(
                    torch.tensor([
                        logits[token_A],
                        logits[token_B],
                        logits[token_C]
                    ]).float(),
                    dim=0
                ).detach().cpu().numpy()
                pred = np.argmax(probs)
            else:
                # Binary classification
                probs = torch.nn.functional.softmax(
                    torch.tensor([
                        logits[token_A],
                        logits[token_B]
                    ]).float(),
                    dim=0
                ).detach().cpu().numpy()
                
                # Apply task-specific mappings
                task_mappings = {
                    'qqp': {0: 1, 1: 0},   # A->no(1), B->yes(0)
                    'sst2': {0: 1, 1: 0},  # A->positive(1), B->negative(0)
                    'qnli': {0: 0, 1: 1},  # A->yes(0), B->no(1)
                    'rte': {0: 1, 1: 0}    # A->yes(1), B->no(0)
                }
                task_map = task_mappings.get(task_name, {0: 0, 1: 1})
                pred = task_map[np.argmax(probs)]
            
            cor = pred == label
            task_cors.append(cor)
            cors.append(cor)
            
            # Print progress every 100 examples
            if (i - start_idx + 1) % 100 == 0:
                print(f"  Processed {i - start_idx + 1}/{end_idx - start_idx} examples")
        
        task_acc = np.mean(task_cors)
        print(f"Accuracy {task_acc:.4f} - Task {task_name}")
    
    acc = np.mean(cors)
    print(f"\nAverage accuracy {acc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5, help='number of shots')
    parser.add_argument("--path", type=str, required=True, help='model checkpoint location')
    parser.add_argument("--data_file", type=str, default='data/adv_glue/dev_ann.json', help='Input data JSON file.')
    parser.add_argument("--test_origin", action='store_true', help='Whether to test on the original GLUE data.')
    args = parser.parse_args()
    main(args)