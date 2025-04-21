import gc
import sys
import glob
import math
from scipy import spatial
from gensim.models import FastText
from gensim.test.utils import common_texts
import numpy as np
import pandas as pd
import random
import os
import re

#print(glob.glob("*"))
sys.path.append(".")
sys.path.append("./utils")
from llm import LLM
from parse_config import parse_args, parse_config
from metrics import _calculate_metrics, aggreate, ambiguity_differentiation

from lofree.prompts import examples_generation

class LoFreeConfig():
    def __init__(self, config):
        config_mapping = {
            'examples_generation': ['model', 'generation_kwargs'],
        }
        self.config = config
        self.examples_generation = config['examples_generation']

def format_examples(examples):
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

class LoFreePipe():
    def __init__(self, config=None):
        self.config = config
        self.prompting_number = 2
        self.lambda1 = 0.1
        self.lambda2 = 0.1
        self.cp = 1.093328028091499
        self.mapping_1 = ['A', 'B', 'C', 'D']
        self.model_ss = FastText(sentences=common_texts, vector_size=200, window=5, min_count=1, workers=4)
        self.model_ss.save("ft.model")

        
    def answer_with_cp(self, scores):
        possible_options = []
        for key in scores.keys():
            if scores[key] >= self.cp:
                possible_options.append(key) 

        return scores, possible_options


    def run(self, sample):
        sample = lofree.generate_options(sample)
        sample = lofree.frequency(sample)
        normalized_entropy = lofree.normalized_entropy(sample['frequences'])
        sample = lofree.semantic_similarity(sample)
        sample = lofree.nonconformity_score(sample, normalized_entropy)
        nonconformity_scores, rest_options = lofree.answer_with_cp(sample['nonconformity_scores'])

        if len(rest_options)==0:
            return sample, []
        return sample, rest_options  

    
    def generate_options(self, sample):
        #20 = 5* 4
        llm = LLM(self.config.examples_generation['model'],
                  self.config.examples_generation['generation_kwargs'])

        prompt = examples_generation.replace('<DESCRIPTION>', sample['description'])
        prompt = prompt.replace('<TASK>', sample['task'])
        prompt = prompt.replace('<PREFIX>', sample['prefix'])
        prompt = prompt.replace('<ACT>', sample['action'])

        answers_raw = []
        answers = {}
         
        for i in range(self.prompting_number):
            text = llm.generate(prompt)
            text = text.replace(prompt, "")
            options, options_str = format_examples(text)
            for k, v in options.items():
                option = v.lower()
                option = option[3:]
                answers_raw.append(text)
                if option in answers.keys():
                    answers[option] += 1
                else:
                    answers[option] = 1

        llm = None
        sample['options'] = answers.keys()
        sample['answers'] = answers
        self.unique_options = len(answers)
        
        return sample, answers
    
    def frequency(self, sample):
        frequences = {}
        sample = sample[0]
        for k, v in sample['answers'].items():
            frequences[k] = v/(self.prompting_number * 4)
        sample['frequences'] = frequences
        return sample
    
    def normalized_entropy(self, frequences):
        log_sum = 0
        for option, freq in frequences.items():
            log_sum += freq * math.log(freq)
        normalized_entropy = abs(log_sum/ math.log(self.prompting_number))
        return normalized_entropy
    
    def get_sentence_vector(self, sentence):
        words = sentence.split()
        vector = [self.model_ss.wv[word] for word in words if word in self.model_ss.wv]
        if vector:
            return sum(vector) / len(vector)
        else:
            return np.zeros(self.model_ss.vector_size)

    def semantic_similarity(self, sample):
        frequences = sample['frequences']
        
        similarities = {}
        frequences_sorted = dict(sorted(frequences.items(), key=lambda item: item[1], reverse = True))
        best_option, highest_freq = next(iter(frequences_sorted.items()))
        vector_top =  self.get_sentence_vector(best_option)

        for option, freq in frequences.items():
            vector_cur = self.get_sentence_vector(option)
            if freq != highest_freq:
                similarities[option] = 1 - spatial.distance.cosine(vector_cur, vector_top)
            else:
                similarities[option] = 0

        sample['similarities'] = similarities 
        return sample

    def nonconformity_score(self, sample, normalized_entropy):
        
        similarities = sample['similarities']
        frequences = sample['frequences']

        nonconformities = {}

        #self.lambda1 и self.lambda2 are hyperparameters
        for option, freq in frequences.items():
            nonconformities[option] = 1 - freq + self.lambda1 * normalized_entropy - self.lambda2 * similarities[option]

        sample['nonconformity_scores'] = nonconformities
        
        return sample


if __name__ == "__main__":
    configs = parse_config("./configs/lofree.yaml" , use_args=True)
    gen_model = configs['examples_generation']['model']
    if "/" in gen_model:
        gen_model = gen_model.split("/")[1]
    exp_res_dir = f"./lofree_{gen_model}"
    os.makedirs(exp_res_dir, exist_ok=True)

    print()
    print(" Start experiment !", exp_res_dir)
    print()


    lofree_config = LoFreeConfig(configs)
    lofree = LoFreePipe(lofree_config)

    dataset = pd.read_csv("./ambik_dataset/ambik_test_900.csv") #ambik_test_400.csv #ambik_test_for_testing.csv
    amb = dataset[['environment_short', 'environment_full',  'ambiguity_type', 'amb_shortlist', 'ambiguous_task', 'question', 'answer', 'plan_for_amb_task', 'end_of_ambiguity', 'user_intent']]
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
        
    
    for i in range(len(test_set)):
        sample, llm_answers = lofree.run(test_set[i])
                
        if isinstance(amb_shortlist[i], str):
            sample_keywords = amb_shortlist[i].split(",")
        else:
            sample_keywords = -1
        
        metrics = _calculate_metrics(llm_answers, sample['nonconformity_scores'], amb_type[i], intents[i].split(', '), sample_keywords)

        for key in metrics:
            metrics_batch[key].append(metrics[key])
        
        
        if i%50 == 0:
            agg_metrics = aggreate(metrics_batch)
            agg_metrics_df = pd.DataFrame(agg_metrics)
            agg_metrics_df.to_csv(f"{exp_res_dir}/lofree_agg_metrics_{i}.csv")        
            metrics = pd.DataFrame(metrics_batch)
            metrics.to_csv(f"{exp_res_dir}/lofree_metrics_{i}.csv")
     
    metrics = pd.DataFrame(metrics_batch)

    metrics, amb_dif = ambiguity_differentiation(metrics)
    print(amb_dif)
    with open (f"{exp_res_dir}/lofree_ambdif_{i}.txt", 'a') as file:
        file.write(str(amb_dif))

    metrics.to_csv(f"{exp_res_dir}/lofree_metrics_{i}.csv")
