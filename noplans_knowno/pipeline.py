import os
import gc
import sys
import glob
import random
import numpy as np
import pandas as pd
import re
import spacy
import re
import tqdm

#print(glob.glob("*"))
sys.path.append(".")
sys.path.append("./utils")
from llm import LLM
from parse_config import parse_args, parse_config
from metrics import _calculate_metrics, aggreate, batch_metric_calculation, ambiguity_differentiation

from original_knowno.prompts import examples_generation, answer_generation

CP = 0.9999967157065831
#0.26287592743744476 #SET YOUR CP VALUE DEFINED THROUGH RUNNING calibration.py SCRIPT

class KnowNoConfig():
    def __init__(self, config):
        config_mapping = {
            'examples_generation': ['model', 'generation_kwargs'],
            'answering': ['model', 'generation_kwargs'],
        }
        self.config = config
        self.examples_generation = config['examples_generation']
        self.answering = config['answering']

def format_examples(examples): 
    #checking whether the answer prompt is in the correct format: A) <option A>\nB) <option B> ... D) <option D>. If something is wrong, replacing the variant with 'do nothing'
    lines = examples.split("\n")
    if "\n" in examples:
        lines = examples.split("\n")
        if len(lines) < 4:
            lines = [x for x in re.split(r'(\d. [\w\s]*.)', examples) if len(x) > 2]         
    else:
        lines = [x for x in re.split(r'(\d. [\w\s]*.)', examples) if len(x) > 2]

    options = ""
    mapping = {"A": "1", "B":"2", "C": "3", "D": "4"}
    variants = {"A":[], "B":[], "C":[], "D":[]}
    for line in lines:
        for key in variants.keys():    
            if line.startswith(f"{key})") or line.startswith(f"{mapping[key]}.") or line.startswith(f"{mapping[key]})"):
                variants[key].append(line)

    for key in variants.keys():
        variants[key] = list(set(variants[key]))
        if len(variants[key]) > 0:
            options+=variants[key][0] +"\n"
            variants[key] = variants[key][0]
        else:
            options += 'do nothing' +"\n"
            variants[key] = 'do nothing'
    return variants, options
            
    
class KnowNoPipe():
    def __init__(self, config=None):
        global CP
        self.config = config
        self.cp = CP #You can use calibrate.py to recalculate 0.9 - llama 0.7 gemma
        self.mapping_1 = ['A', 'B', 'C', 'D']

    def options_prompt(self, description, task):
        #creating prompt for generating options (base prompt is taken from knowno/prompts/generation.txt)
        prompt = examples_generation.replace('<DESCRIPTION>', description)
        prompt = prompt.replace('<TASK>', task)
        return prompt

    def format_options(self, prompt, options):
        examples = options.replace(prompt, "")
        options, options_str = format_examples(examples)
        return options
        
    def predict_examples(self,  description, task, prefix, action):
        llm = LLM(self.config.examples_generation['model'],
                  self.config.examples_generation['generation_kwargs'])
        prompt = self.options_prompt(description, task, prefix, action)
        options = llm.generate(prompt)
        llm = None
        
        return self.format_options(prompt, options)

    def predict_examples_batch(self, prompts, batch_size=2): #generate examples for batch
        llm = LLM(self.config.examples_generation['model'],
                  self.config.examples_generation['generation_kwargs'])
        options_full = []
        for i in tqdm.tqdm(range(0, len(prompts), batch_size)):  
            options = llm.generate_batch(prompts[i:i+batch_size])
            options_full += options

        formated_options = []
        for i in range(len(prompts)):
            prompt = prompts[i]
            options = options_full[i]
            formated_options.append(self.format_options(prompt, options))
        llm = None
        gc.collect()
        return formated_options
        
    def answer_with_cp(self, tokens_logits): #take in CP set only options whose logits are greater then CP value. CP value is defined through running calibration.py script
        possible_options = []
        for key in tokens_logits.keys():
            if tokens_logits[key] > self.cp:
                possible_options.append(key) 
      #  print(possible_options)
        formated_options = []
        for option in possible_options:
            if option.isdigit():  
                option_formated = self.mapping_1[int(option)-1]
            else:
                option_formated = option.upper()
            if option_formated not in formated_options:
                formated_options.append(option_formated)
        return possible_options

    def answer_prompt(self, prompts_di, description, task):
        #prompt for getting logprobs of A, B, C, D variants. base prompt is in knowno/prompts/choising.txt
        #prompt = prompt.replace("You", "Options")
        prompt = ''
        for key, value in prompts_di.items():
            prompt += key
            prompt += ') '
            prompt += value[3:]
            prompt += '\n'
        prompt = answer_generation.replace('<OPTIONS>', prompt)
        prompt = prompt.replace('<TASK>', task)
        prompt = prompt.replace('<DESCRIPTION>', description)
        return prompt
        
    def generate_answer_batch(self, prompts, batch_size=2): #choosing CP set for batch
        llm = LLM(self.config.answering['model'],
                  self.config.answering['generation_kwargs'])
        full_texts = []
        full_logits = []
        for i in tqdm.tqdm(range(0, len(prompts), batch_size)):
            texts, logits = llm.generate_batch(prompts[i:i+batch_size], return_logits=True)
            full_texts+=texts
            full_logits+=logits

        filtered_logits_batch = []
        answers = []
        for i in range(len(full_texts)):
            filtered_logits = llm.filter_logits(full_logits[i][0], words=["A", "B", "C", "D", 'a', 'b', 'c', 'd', '1', '2', '3', '4'])
            #print(filtered_logits)
            filtered_logits_batch.append(filtered_logits)
            answer = self.answer_with_cp(filtered_logits)
            answers.append(answer)
        llm = None
        gc.collect()
        return filtered_logits_batch, answers
        
    def generate_answer(self, prompt, description, task, prefix, action): #choosing CP set for single example
        llm = LLM(self.config.answering['model'],
                  self.config.answering['generation_kwargs'])
        
        prompt = self.answer_prompt(prompt, description, task, prefix, action)
        text, logits = llm.generate(prompt, return_logits=True)
        filtered_logits = llm.filter_logits(logits[-1][0], words=["A", "B", "C", "D", 'a', 'b', 'c', 'd', '1', '2', '3', '4'])
        llm = None
        return filtered_logits, self.answer_with_cp(filtered_logits)

    def run(self, description, task, prefix, action):
        #generating multiple iptions
        options, task = self.predict_examples(description, task, prefix, action) 
        #getting logits of options and choosing the options with logits greater than CP value. constricting a CP set.
        answers_letter = self.generate_answer(task, task[1]
        answers = [options[letter] for letter in answers_letter]
          
        if len(answers)==0:
            return [] #if no options are left in the set, it is impossible to answer
        return answers #else there 1 or many answers

    def run_batch(self, option_prompts, tasks_for_ans): #run, but for batch
        options = self.predict_examples_batch(option_prompts)
        answer_prompts = []
        for i in range(len(options)):
            answer_prompts.append(self.answer_prompt(options[i], tasks_for_ans[i]['description'], tasks_for_ans[i]['task'], tasks_for_ans[i]['prefix'], tasks_for_ans[i]['action']))
        logits, answers = self.generate_answer_batch(answer_prompts)
        right_answers = []
    
        for i in range(len(answers)):
            option = options[i]
            answers_letter = answers[i]
            if len(answers_letter) > 0:
                answers_ = [option[letter] for letter in answers_letter]
            else: 
                answers_ = []
            right_answers.append(answers_)
        return options, right_answers

if __name__ == "__main__":

    configs = parse_config("./configs/knowno.yaml" , use_args=True)
    gen_model = configs['examples_generation']['model']
    if "/" in gen_model:
        gen_model = gen_model.split("/")[1]
    answ_model = configs['answering']['model']
    if "/" in answ_model:
        answ_model = answ_model.split("/")[1]
    exp_res_dir = f"./{CP}_{gen_model}_{answ_model}"
    os.makedirs(exp_res_dir, exist_ok=True)

    print()
    print(" Start experiment !", exp_res_dir)
    print()
    knowno_config = KnowNoConfig(configs)
    knowno = KnowNoPipe(knowno_config)

    dataset = pd.read_csv("./ambik_dataset/ambik_test_400.csv") #ambik_test_400.csv #ambik_test_for_testing.csv
    amb = dataset[['id', 'environment_short', 'environment_full',  'ambiguity_type', 'amb_shortlist', 'ambiguous_task', 'question', 'answer', 'plan_for_amb_task', 'end_of_ambiguity', 'user_intent']]
    dataset.ambiguity_type = ['unambiguous_direct']*len(dataset)
    dataset = pd.concat([dataset, amb])
    dataset['plan'] = dataset['plan_for_clear_task']
    dataset['plan'] = dataset['plan'].fillna(dataset['plan_for_amb_task'])
    dataset['task'] = dataset['unambiguous_direct']
    dataset['task'] = dataset['task'].fillna(dataset['ambiguous_task'])
    dataset = dataset.drop(columns=['Unnamed: 0', 'unambiguous_direct', 'unambiguous_indirect', 'ambiguous_task', 'plan_for_clear_task', 'plan_for_amb_task', 'variants'])
    dataset = dataset.reset_index()
   
   #Data to metrics
    amb_type = dataset['ambiguity_type'].values
    intents = dataset['user_intent'].values
    amb_shortlist = dataset['amb_shortlist'].values
    
    calibration_data = []
    metrics_batch = {'llm_answers':[], 'y_amb_type':[], 'y_amb_intents':[], 'y_amb_shortlist':[],
                     'SR':[], 'help_rate': [], 'correct_help_rate': [], 'SSC': []}
    option_prompts = []
                                  
    tasks_for_ans = []
    for i in range(len(dataset)): #len(dataset) 
        description = dataset.loc[i, 'environment_full']
        task = dataset.loc[i, 'task']
        plan = dataset.loc[i, 'plan'].split('\n')
        point = dataset.loc[i, 'end_of_ambiguity']
        if point == 0:
            prefix = 'Your first action is:'
        else:
            prefix = 'Your previous actions were:\n'
            for act in plan[:point]:
                prefix += act
                prefix += '\n'
        action = plan[point]
        
        tasks_for_ans.append({'description':description, 'task':task, 'prefix':prefix, 'action':action})
        option_prompt = knowno.options_prompt(description, task, prefix, action)
        option_prompts.append(option_prompt)
    
    options, right_answers = knowno.run_batch(option_prompts, tasks_for_ans)
    batch_size = 2
    metrics_batch = batch_metric_calculation(llm_answers_batch=right_answers, scores=options, y_amb_type_batch=amb_type, y_amb_intents_batch=intents, y_amb_shortlist_batch = amb_shortlist)
    
    agg_metrics = aggreate(metrics_batch)
    #agg_metrics = {key:[agg_metrics[key]] for key in agg_metrics}
    agg_metrics_df = pd.DataFrame(agg_metrics)

    agg_metrics_df.to_csv(f"{exp_res_dir}/original_knowno_agg_metrics_{i}.csv") #поправить записььь

    metrics = pd.DataFrame(metrics_batch)

    metrics, amb_dif = ambiguity_differentiation(metrics)
    print(amb_dif)
    with open (f"{exp_res_dir}/original_knowno_ambdif_{i}.txt", 'a') as file:
        file.write(str(amb_dif))

    metrics.to_csv(f"{exp_res_dir}/original_knowno_metrics_{i}.csv")