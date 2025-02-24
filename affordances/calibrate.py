import gc
import sys
import glob
import numpy as np
import pandas as pd
import re

print(glob.glob("*"))
sys.path.append(".")
sys.path.append("./utils")
from llm import LLM
from parse_config import parse_args, parse_config

from affordances.prompts import examples_generation, affordance_generation, answer_generation

from affordances.pipeline import AffordancesConfig, AffordancesPipe


def get_logits(affordances, sample):
     sample = affordances.predict_examples(sample) 
     gc.collect()
     sample, cp_ans = affordances.generate_answer(sample)
     gc.collect()
     return sample, cp_ans


def filter_similar_sentences(A, B):
    """
    Фильтрует словарь A, оставляя только те предложения, которые по смыслу похожи на предложения из списка B.

    :param A: Словарь с предложениями для фильтрации
    :param B: Список предложений для сравнения
    :param threshold: Порог схожести для сравнения предложений (по умолчанию 0.7)
    :return: Отфильтрованный словарь
    """
    
    # Функция для удаления частей "A)", "B)", и т.д.
    
    def remove_prefix(text):
        return re.sub(r'^[A-Z]\)\s*', '', text)

    processed_A = {key: remove_prefix(sentence) for key, sentence in A.items()}
   # processed_B = [remove_prefix(sentence) for sentence in B]

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
    for key, sentence_A in processed_A.items():
        if any(is_similar(sentence_A, sentence_B) for sentence_B in B): #processed_B
            filtered_A[key] = A[key] 
    return filtered_A


def calibration():
    target_success = 0.8 
    epsilon = 1-target_success
    
    configs = parse_config("./configs/affordances.yaml" , use_args=True)
    affordances_config = AffordancesConfig(configs)
    affordances = AffordancesPipe(affordances_config)

    #Calibration set
    dataset = pd.read_csv("./ambik_dataset/ambik_calib_100.csv")

    calibration_set = []
    num_calibration_data = 100 #not less than 5
    
    calibration_data = []
    calib_data = pd.DataFrame(columns=['id', 'task', 'all variants', 'right variants'])
    for i in range(num_calibration_data): #range(num_calibration_data)
        if dataset.loc[i, 'take_amb'] == 1:
            plan = dataset.loc[i, 'plan_for_amb_task'].split('\n')
            task = dataset.loc[i, 'ambiguous_task']
        else:
            plan = dataset.loc[i, 'plan_for_clear_task'].split('\n')
            task = dataset.loc[i, 'unambiguous_direct']            
        point = int(dataset.loc[i, 'end_of_ambiguity'])
        action = plan[point]
        if point == 0:
            prefix = 'Your first action is:'
        else:
            prefix = 'Your previous actions were:\n'
            for act in plan[:point]:
                prefix += act
                prefix += '\n'
        variants = dataset.loc[i, 'variants'].split("\n")
        id_ = dataset.loc[i, 'id']
        calibration_set.append({
                 'id':id_,
                 'description' : dataset.loc[i, 'environment_full'],
                 'task': task,
                 'point': point,
                 'prefix':prefix,
                'action': action,
                'variants':variants})
           
    for sample in calibration_set:
        sample, cp_ans = get_logits(affordances, sample)
        variants = sample['variants']
        filtered_options = filter_similar_sentences(sample['options'], variants)
        success_logits = [sample['affordance_scores'][key] for key in filtered_options]
        calibration_data+=success_logits
        
        row = {'id':sample['id'], 'task':sample['task'], 'all variants': sample['options'],
               'right variants': ", ".join(variants),
               'filtered variants': ", ".join(filtered_options.values()), 
               'success_logits': ', '.join(str(x) for x in success_logits)}
        #calib_data = pd.concat([calib_data, pd.DataFrame([row])], ignore_index=True) Neede if one wants to save calibration data

    gen_model = configs['examples_generation']['model']
    if "/" in gen_model:
        gen_model = gen_model.split("/")[1]
    answ_model = configs['answering']['model']
    if "/" in answ_model:
        answ_model = answ_model.split("/")[1]
    #calib_data.to_csv('calib_data/LAP_' + gen_model + '_' + answ_model +'.csv')
        
    num_calibration_data = len(calibration_data)
    q_level = np.ceil((num_calibration_data + 1) * (1 - epsilon)) / num_calibration_data
    print('q_level' + str(q_level))
    print(calibration_data)
    qhat = np.quantile(calibration_data, q_level)
    return qhat

if __name__ == "__main__":
    print("CP: ", calibration())