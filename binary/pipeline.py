import os
import gc
import sys
import numpy as np
import pandas as pd
import re
import tqdm

#print(glob.glob("*"))
sys.path.append(".")
sys.path.append("./utils")
from llm import LLM
from parse_config import parse_args, parse_config
from metrics import ambiguity_differentiation, batch_binary_metric_calculation, binary_aggreate

from binary.prompts import examples_generation, uncertainty_detection


class BinaryConfig():
    def __init__(self, config):
        config_mapping = {
            'examples_generation': ['model', 'generation_kwargs'],
            'uncertainty_detection': ['model', 'generation_kwargs'],
        }
        self.config = config
        self.examples_generation = config['examples_generation']
        self.uncertainty_detection = config['uncertainty_detection']
    
class BinaryPipe():
    def __init__(self, config=None):
        self.config = config

    def options_prompt(self, description, task, prefix, action):
        #creating prompt for generating options (base prompt is taken from knowno/prompts/generation.txt)
        prompt = examples_generation.replace('<DESCRIPTION>', description)
        prompt = prompt.replace('<TASK>', task)
        prompt = prompt.replace('<PREFIX>', prefix)
        prompt = prompt.replace('<ACT>', action)
        return prompt

    def format_options(self, prompt, option):
        examples = option.replace(prompt, "")
        return examples

    def predict_examples_batch(self, prompts, batch_size=2): #generate examples for batch
        llm = LLM(self.config.examples_generation['model'],
                  self.config.examples_generation['generation_kwargs'])
        options_full = []
        for i in tqdm.tqdm(range(0, len(prompts), batch_size)):  
            options = llm.generate_batch(prompts[i:i+batch_size])
            options_clean = []
            for opt in options:
                opt = opt.strip()
                options_clean.append(opt)
            options_full += options

        formated_options = []
        for i in range(len(prompts)):
            prompt = prompts[i]
            option = options_full[i]
            opt = self.format_options(prompt, option)
            opt = opt.strip()
            formated_options.append([opt])

        llm = None
        gc.collect()
        return formated_options
        
    def unc_prompt(self, option, description, task, prefix, action):
        #prompt for getting logprobs of certain/uncertain variants.
        prompt = uncertainty_detection.replace('<OPTIONS>', option)
        prompt = prompt.replace('<TASK>', task)
        prompt = prompt.replace('<PREFIX>', prefix)
        prompt = prompt.replace('<ACT>', action)
        prompt = prompt.replace('<DESCRIPTION>', description)
        return prompt
        
    def generate_uncertainty_batch(self, prompts, batch_size=2): #choosing CP set for batch
        llm = LLM(self.config.uncertainty_detection['model'],
                  self.config.uncertainty_detection['generation_kwargs'])
        full_texts = []
        for i in tqdm.tqdm(range(0, len(prompts), batch_size)):
            texts = llm.generate_batch(prompts[i:i+batch_size], return_logits=False)
            full_texts+=texts

        formated_texts = []
        for i in range(len(prompts)):
            prompt = prompts[i]
            text = full_texts[i]
            certainty = self.format_options(prompt, text)
            certainty = certainty.strip()
            formated_texts.append(certainty)

        return formated_texts

    def run_batch(self, option_prompts, tasks_for_ans): #run, but for batch
        options = self.predict_examples_batch(option_prompts)
        unc_prompts = []
        for i in range(len(options)):
            unc_prompts.append(self.unc_prompt(options[i][0], tasks_for_ans[i]['description'], tasks_for_ans[i]['task'], tasks_for_ans[i]['prefix'], tasks_for_ans[i]['action']))
        answers = self.generate_uncertainty_batch(unc_prompts) #здесь внутри сертайн/ансертайн

        return options, answers

if __name__ == "__main__":

    configs = parse_config("./configs/binary.yaml" , use_args=True)
    gen_model = configs['examples_generation']['model']
    if "/" in gen_model:
        gen_model = gen_model.split("/")[1]
    unc_model = configs['uncertainty_detection']['model']
    if "/" in unc_model:
        answ_model = unc_model.split("/")[1]
    exp_res_dir = f"./binary_{gen_model}_{unc_model}"
    os.makedirs(exp_res_dir, exist_ok=True)

    print()
    print(" Start experiment !", exp_res_dir)
    print()
    binary_config = BinaryConfig(configs)
    binary = BinaryPipe(binary_config)

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
                     'SR':[], 'help_rate': [], 'correct_help_rate': []}
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
        option_prompt = binary.options_prompt(description, task, prefix, action)
        option_prompts.append(option_prompt)
    
    options, right_answers = binary.run_batch(option_prompts, tasks_for_ans)
    batch_size = 2
    metrics_batch = batch_binary_metric_calculation(llm_answers_batch=options, scores=right_answers, y_amb_type_batch=amb_type, y_amb_intents_batch=intents, y_amb_shortlist_batch = amb_shortlist)
    
    agg_metrics = binary_aggreate(metrics_batch)
    #agg_metrics = {key:[agg_metrics[key]] for key in agg_metrics}
    agg_metrics_df = pd.DataFrame(agg_metrics)

    agg_metrics_df.to_csv(f"{exp_res_dir}/binary_agg_metrics_{i}.csv") #поправить записььь

    metrics = pd.DataFrame(metrics_batch)
    metrics.to_csv(f"{exp_res_dir}/binary_metrics_{i}.csv")


    metrics, amb_dif = ambiguity_differentiation(metrics)
    print(amb_dif)
    with open (f"{exp_res_dir}/binary_ambdif_{i}.txt", 'a') as file:
        file.write(str(amb_dif))

    #переписать хелп рейт, мб вообще отдельную функцию для метрик