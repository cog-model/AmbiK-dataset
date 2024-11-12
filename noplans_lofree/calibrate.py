import gc
import sys
import glob
import math
from scipy import spatial
from gensim.models import FastText
from gensim.test.utils import common_texts
import numpy as np
import pandas as pd
import re

#print(glob.glob("*"))
sys.path.append(".")
sys.path.append("./utils")
from llm import LLM
from parse_config import parse_args, parse_config

from noplans_lofree.prompts import examples_generation

from noplans_lofree.pipeline import LoFreeConfig, LoFreePipe


def get_logits(lofree, sample): #def get_logits(knowno, prompt)
    sample = lofree.generate_options(sample) 
    gc.collect()
    sample = lofree.frequency(sample)
    normalized_entropy = lofree.normalized_entropy(sample['frequences'])
    sample = lofree.semantic_similarity(sample)
    sample = lofree.nonconformity_score(sample, normalized_entropy)
    
    gc.collect()
    return sample


def filter_similar_sentences(A, B):
    """
    Фильтрует словарь A, оставляя только те предложения, которые по смыслу похожи на предложения из списка B.

    :param A: Словарь с предложениями для фильтрации
    :param B: Список предложений для сравнения
    :param threshold: Порог схожести для сравнения предложений (по умолчанию 0.7)
    :return: Отфильтрованный словарь
    """
    
    # Функция для удаления частей "A)", "B)", и т.д.

    def is_similar(sent1, sent2):
        right = 0
        splitted = sent2.split(', ')
        total = len(splitted)
        target = sent1.lower()
        for el in splitted:
            if el.startswith('-'):
                flag = False
                variables = el.replace('-', '')
                variables = variables.split('|')
                for var in variables:
                    if ' '+var in target:
                        flag = True
                if flag == False:
                    right += 1
            else:
                flag = False
                variables = el.split('|')
                for var in variables:
                    if ' '+var in target:
                        flag = True
                if flag == True:
                    right += 1

        if right == total:
            similarity = True
        else:
            similarity = False
            
        return similarity
        
    filtered_A = {}
    for sentence_A, score in A.items():
        if any(is_similar(sentence_A, sentence_B) for sentence_B in B): #processed_B
            filtered_A[sentence_A] = score 
    return filtered_A

def calibtation():
    target_success = 0.8 
    epsilon = 1-target_success
    
    configs = parse_config("./configs/lofree.yaml" , use_args=True)
    lofree_config = LoFreeConfig(configs)
    lofree = LoFreePipe(lofree_config)

    #Calibration set
    dataset = pd.read_csv("./ambik_dataset/ambik_calib_100.csv")
    
    calib_data = pd.DataFrame(columns=['id', 'task', 'all variants', 'right variants'])

    calibration_data = []
    for i in range(len(dataset)): #len(dataset)
        if dataset.loc[i, 'take_amb'] == 1:
            plan = dataset.loc[i, 'plan_for_amb_task'].split('\n')
            task = dataset.loc[i, 'ambiguous_task']
        else:
            plan = dataset.loc[i, 'plan_for_clear_task'].split('\n')
            task = dataset.loc[i, 'unambiguous_direct']            
        point = int(dataset.loc[i, 'end_of_ambiguity'])
        action = plan[point]
        
        sample = {'id': dataset.loc[i, 'id'], 
            'description': dataset.loc[i, 'environment_full'],
            'action': action}
         
        sample = get_logits(lofree, sample)
        variants = dataset.loc[i, 'variants'].split("\n") #'variants_best'

        nonconformity_scores = sample['nonconformity_scores']
        #print(nonconformity_scores)
        filtered_options = filter_similar_sentences(nonconformity_scores, variants)
        #success_logits = [nonconformity_scores[value] for key, value in filtered_options.items()]
        
        row = {'id': dataset.loc[i, 'id'], 'task': task, 'all variants':
               sample['nonconformity_scores'].keys(),
               'right variants': ", ".join(variants),
               'filtered variants': ", ".join(filtered_options.keys()), 
               'success_logits': filtered_options}
        calib_data = pd.concat([calib_data, pd.DataFrame([row])], ignore_index=True)
        
        calibration_data+=filtered_options.values()

    gen_model = configs['examples_generation']['model']
    if "/" in gen_model:
        gen_model = gen_model.split("/")[1]
    calib_data.to_csv('calib_data/noplans_LofreeCP_' + gen_model  +'.csv')

    num_calibration_data = len(calibration_data)
    q_level = np.ceil((num_calibration_data + 1) * (1 - epsilon)) / num_calibration_data
    qhat = np.quantile(calibration_data, q_level)
    return qhat

if __name__ == "__main__":
    print("CP: ", calibtation())