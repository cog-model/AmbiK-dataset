import gc
import sys
import glob
import random
import numpy as np
import pandas as pd
import re

#print(glob.glob("*"))
sys.path.append(".")
sys.path.append("./utils")
from llm import LLM
from parse_config import parse_args, parse_config

from knowno.prompts import examples_generation, question_generation, answer_generation

from knowno.pipeline import KnowNoConfig, KnowNoPipe

def get_logits(knowno, description, task, prefix, action): #def get_logits(knowno, prompt)
     options = knowno.predict_examples(description, task, prefix, action) 
     gc.collect()
     choose = knowno.generate_answer(options, description, task, prefix, action)
     gc.collect()
     #print('options')
     #print(options)
     #print('choose[0]')
     print(choose[0])
     return options, choose[0]

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
    return filtered_A #A, 

def calibration():
    target_success = 0.8 
    epsilon = 1-target_success
    
    configs = parse_config("./configs/knowno.yaml" , use_args=True)
    knowno_config = KnowNoConfig(configs)
    knowno = KnowNoPipe(knowno_config)

    #Calibration set
    dataset = pd.read_csv("./ambik_dataset/ambik_calib_100.csv")

    calibration_data = []
    
    #calib data датасет и файл для сохранения результатов калибровки (обычно не нужен)
    #calib_data = pd.DataFrame(columns=['id', 'task', 'all variants', 'right variants'])
    for i in range(len(dataset)):
        description = dataset.loc[i, 'environment_full']
        if dataset.loc[i, 'take_amb'] == 1:
            plan = dataset.loc[i, 'plan_for_amb_task'].split('\n')
            task = dataset.loc[i, 'ambiguous_task']
        else:
            plan = dataset.loc[i, 'plan_for_clear_task'].split('\n')
            task = dataset.loc[i, 'unambiguous_direct']            
        point = dataset.loc[i, 'end_of_ambiguity']
        action = plan[point]
        if point == 0:
            prefix = 'Your first action is:'
        else:
            prefix = 'Your previous actions were:\n'
            for act in plan[:point]:
                prefix += act
                prefix += '\n'
        action = plan[point]
         
        options, answers_logits = get_logits(knowno, description, task, prefix, action)
        options_to_filter = {}
        for key in answers_logits.keys():
            if key in options.keys():
                options_to_filter[key] = options[key]
        #print('options ==== ', options)
        #print('logits ==== ', answers_logits)
        variants = dataset.loc[i, 'variants'].split("\n")

        filtered_options = filter_similar_sentences(options_to_filter, variants) #options,
        #print('filtered_options === ', filtered_options)

        success_logits = [answers_logits[key] for key in filtered_options]
        #print('success_logits ==== ', success_logits)
        calibration_data+=success_logits
        
        row = {'id':dataset.loc[i, 'id'], 'task':task, 'all variants':", ".join(options.values()),
               'right variants': ", ".join(variants),
               'filtered variants': ", ".join(filtered_options.values()), 
               'success_logits': ', '.join(str(x) for x in success_logits)}
        #calib_data = pd.concat([calib_data, pd.DataFrame([row])], ignore_index=True)
        
    model = configs['examples_generation']['model']
    model = model.split('/')[-1]
    #calib_data.to_csv('calib_data/knowno_' + model +'.csv')
            

    num_calibration_data = len(calibration_data)
    q_level = np.ceil((num_calibration_data + 1) * (1 - epsilon)) / num_calibration_data
    qhat = np.quantile(calibration_data, q_level)
    return qhat

if __name__ == "__main__":
    print("CP: ", calibration())