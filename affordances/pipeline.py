import gc
import sys
import numpy as np
import pandas as pd
import os
import re

#print(glob.glob("*"))
sys.path.append(".")
sys.path.append("./utils")
from llm import LLM
from parse_config import parse_args, parse_config
from metrics import _calculate_metrics, aggreate, ambiguity_differentiation

from affordances.prompts import examples_generation, affordance_generation, answer_generation
from affordances.environment_objects import environment_objects

CP = 2.718281828459045

class AffordancesConfig():
    def __init__(self, config):
        config_mapping = {
            'examples_generation': ['model', 'generation_kwargs'],
            'answering': ['model', 'generation_kwargs'],
            'affordance_generation': ['model', 'generation_kwargs'],
        }
        self.config = config
        self.examples_generation = config['examples_generation']
        self.answering = config['answering']
        self.affordance_generation = config['affordance_generation']

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

class AffordancesPipe():
    def __init__(self, config=None):
        self.config = config
        self.cp = CP #YOUR CP VALUE
        self.mapping_1 = ['A', 'B', 'C', 'D']
        self.affordance_prompt = affordance_generation
        self.environment_objects = environment_objects #список всех объектов в среде

    def context_based_affordance(self, objects, scene):
        for obj in scene:
            if obj not in scene:
                return 0
        return 1

    def prompt_based_affordance(self, prompt):
        llm = LLM(self.config.affordance_generation['model'],
                  self.config.affordance_generation['generation_kwargs'])
        generated_text, logits = llm.generate(prompt, return_logits = True)

        generated_text = generated_text.replace(prompt + ' ', '')
        generated_text = generated_text.split('\n')[0]

        if generated_text == 'True' or generated_text == 'Yes':
            return llm.filter_logits(logits[-1][0], words=[generated_text])[generated_text] # первая буква ответа?
        elif generated_text== 'False' or generated_text== 'No':
            return llm.filter_logits(logits[-1][0], words=[generated_text])[generated_text]
        return generated_text + 'IS NOT TRUE OR FALSE'

    def predict_examples(self, sample):
        llm = LLM(self.config.examples_generation['model'],
                  self.config.examples_generation['generation_kwargs'])
        description = sample['description']
        task = sample['task']
        action = sample['action']
        prefix = sample['prefix']
        prompt = examples_generation.replace('<DESCRIPTION>', description)
        prompt = prompt.replace('<TASK>', task)
        prompt = prompt.replace('<PREFIX>', prefix)
        prompt = prompt.replace('<ACT>', action)
        
        examples = llm.generate(prompt)
        llm = None
        examples = examples.replace(prompt, "")
        
        options, options_str = format_examples(examples)
        sample['options_str'] = options_str
        sample['options'] = options
        return sample  


    def answer_with_cp(self, sample):
        #take in CP set only options whose logits are greater then CP value. CP value is defined through running calibration.py script
        tokens_logits = sample['affordance_scores']
        possible_options = []
        for key in tokens_logits.keys():
            if tokens_logits[key] >= self.cp:
                possible_options.append(key) 

        formated_options = []
        for option in possible_options:
            if option.isdigit():  
                option_formated = self.mapping_1[int(option)-1]
            else:
                option_formated = option.upper()
            if option_formated not in formated_options:
                formated_options.append(option_formated)
        return possible_options

    def generate_answer(self, sample):  #для выбора логов
        llm = LLM(self.config.answering['model'],
                  self.config.answering['generation_kwargs'])
        
        prompt = prompt.replace("You", "Options")
        prompt = answer_generation.replace('<PROMPT>', prompt)
        
        text, logits = llm.generate(prompt, return_logits=True)
        filtered_logits = llm.filter_logits(logits[-1][0], words=["A", "B", "C", "D", 'a', 'b', 'c', 'd', '1', '2', '3', '4'])
        sample['filtered_logits'] = filtered_logits
        llm = None
        return sample, self.answer_with_cp(filtered_logits)
    
    def generate_affordances(self, sample): #calculate affordance scores
        description = sample['description']
        task = sample['task']
        context_scores = {}
        promt_scores = {}
        affordance_scores = {}
        li = ['A', 'B', 'C', 'D']
        for j in range(len(li)):
            option = sample['options'][li[j]][0] #sample['options'] это словарь {'A': [], 'B': [], 'C': [], 'D': []}
            affordance_prompt = self.affordance_prompt.replace('<DESCRIPTION>', description)
            affordance_prompt = self.affordance_prompt.replace('<TASK>', task)
            promt_score = self.prompt_based_affordance(affordance_prompt)
            promt_scores[li[j]] = promt_score

            cur_objects = []
            for obj in self.environment_objects:
                if obj in option:
                    cur_objects.append(object)

            context_score = self.context_based_affordance(cur_objects, description)
            context_scores[li[j]] = context_score

            if li[j] in sample['logits'].keys():
                affordance_score = float(promt_score * context_score * np.exp(sample['logits'][li[j]]))
            else:
                affordance_score = 0
            #('promt_score', promt_score)
            #print('context_score', context_score)
            #print('affordance_score!!!!', affordance_score)
            affordance_scores[li[j]] = affordance_score

        return (context_scores, promt_scores, affordance_scores)


    def generate_answer(self, sample):
        llm = LLM(self.config.answering['model'],
                  self.config.answering['generation_kwargs'])
        
        #prompt = prompt.replace("You", "Options")
        prompt = answer_generation.replace('<TASK>', sample['task'])
        prompt = prompt.replace('<DESCRIPTION>', sample['description'])
        prompt = prompt.replace('<PREFIX>', sample['prefix'])
        prompt = prompt.replace('<ACT>', sample['action'])
        prompt = prompt.replace('<OPTIONS>', sample['options_str'])
    
        text, logits = llm.generate(prompt, return_logits=True)
        filtered_logits = llm.filter_logits(logits[-1][0], words=["A", "B", "C", "D", 'a', 'b', 'c', 'd', '1', '2', '3', '4'])
        llm = None
        sample['logits'] = filtered_logits
        context_scores, promt_scores, affordance_scores = self.generate_affordances(sample)
                                                                                    #, description, task, environment_objects)
        sample['affordance_scores'] = affordance_scores
        #print('affordance_scores', affordance_scores)
        
        #filtered_logits = llm.filter_logits(logits[-1][0], words=["A", "B", "C", "D", 'a', 'b', 'c', 'd', '1', '2', '3', '4'])
        #llm = None
        return sample, self.answer_with_cp(sample)

    def run(self, sample): 
        sample = affordances.predict_examples(sample) 
        gc.collect()
        sample, cp_ans = affordances.generate_answer(sample)
        #print(sample)
        #print(cp_ans)
        
        if len(cp_ans)==0:
            return sample, []
        return sample, cp_ans

    
if __name__ == "__main__":
    configs = parse_config("./configs/affordances.yaml" , use_args=True)
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
    affordances_config = AffordancesConfig(configs)
    affordances = AffordancesPipe(affordances_config)
    
    #Test set: choose any subpart of AmbiK for test. We take 500-72 examples
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
    
    #Data fo metrics
    amb_type = dataset['ambiguity_type'].values
    intents = dataset['user_intent'].values
    amb_shortlist = dataset['amb_shortlist'].values
    
    calibration_data = []
    metrics_batch = {'llm_answers':[], 'scores':[], 'y_amb_type':[], 'y_amb_intents':[], 'y_amb_shortlist':[],
                     'SR':[], 'help_rate': [], 'correct_help_rate': [], 'SSC': []}
    option_prompts = []
                                  
    test_set = []
    
    for i in range(len(dataset)): #len(dataset)
        plan = dataset.loc[i, 'plan'].split('\n')
        point = dataset.loc[i, 'end_of_ambiguity']
        action = plan[point]
        if point == 0:
            prefix = 'Your first action is:'
        else:
            prefix = 'Your previous actions were:\n'
            for act in plan[:point]:
                prefix += act
                prefix += '\n'
        test_set.append({
            'description': dataset.loc[i, 'environment_full'],
            'task': dataset.loc[i, 'task'],
            'point': point,
            'prefix':prefix,
            'action': action})
        
    for i in range(len(test_set)): #len(test_set)
        sample, answer = affordances.run(test_set[i])
        scores = sample['options']
        #print('scores', scores)
        llm_answers = []
        for key, option in scores.items():
            #print(key, option)
            if key in answer:
                llm_answers.append(option)
                #llm_answers[key] = option

        if isinstance(amb_shortlist[i], str):
            sample_keywords = amb_shortlist[i].split(",")
        else:
            sample_keywords = -1
        metrics = _calculate_metrics(llm_answers, scores, amb_type[i], intents[i].split(', '), sample_keywords)
        for key in metrics:
            metrics_batch[key].append(metrics[key])
        if i%10 == 0:
           agg_metrics = aggreate(metrics_batch)
           agg_metrics_df = pd.DataFrame(agg_metrics)
           agg_metrics_df.to_csv(f"affordances_{exp_res_dir}/affordances_agg_metrics_{i}.csv")        
           metrics = pd.DataFrame(metrics_batch)
           metrics.to_csv(f"affordances_{exp_res_dir}/affordances_metrics_{i}.csv")

    metrics = pd.DataFrame(metrics_batch)
    metrics.to_csv(f"affordances_{exp_res_dir}/affordances_metrics_{i}.csv")


    metrics, amb_dif = ambiguity_differentiation(metrics)
    print('AmbDif: ',amb_dif)
    with open(f"{exp_res_dir}/affordances_ambdif_{i}.txt", 'a') as file:
        file.write(str(amb_dif))

    metrics.to_csv(f"{exp_res_dir}/affordances_metrics_{i}.csv")