import numpy as np
import pandas as pd

def safe_mean(arr):
    if len(arr) == 0:
        return -1  # or any other value you deem appropriate
    return np.mean(arr)

def batch_metric_calculation(llm_answers_batch, scores, y_amb_type_batch, y_amb_intents_batch, y_amb_shortlist_batch):
    metrics_batch = {'llm_answers':[], 'scores':[], 'y_amb_type':[], 'y_amb_intents':[], 'y_amb_shortlist':[],
                     'SR':[], 'help_rate': [], 'correct_help_rate': [], 'SSC': []}
    for i in range(len(llm_answers_batch)):
        if isinstance(y_amb_intents_batch[i], str):
            y_amb_intents = y_amb_intents_batch[i].split(",")
        if isinstance(y_amb_shortlist_batch[i], str):
            y_amb_shortlist = y_amb_shortlist_batch[i].split(",")
        else:
            y_amb_shortlist = []
        metrics = _calculate_metrics(llm_answers_batch[i],
                                     scores[i],
                                     y_amb_type_batch[i], 
                                     y_amb_intents,
                                     y_amb_shortlist)
        for key in metrics:
            metrics_batch[key].append(metrics[key])      
    return metrics_batch


def batch_binary_metric_calculation(llm_answers_batch, scores, y_amb_type_batch, y_amb_intents_batch, y_amb_shortlist_batch):
    metrics_batch = {'llm_answers':[], 'scores':[], 'y_amb_type':[], 'y_amb_intents':[], 'y_amb_shortlist':[],
                     'SR':[], 'help_rate': [], 'correct_help_rate': []}
    for i in range(len(llm_answers_batch)):
        if isinstance(y_amb_intents_batch[i], str):
            y_amb_intents = y_amb_intents_batch[i].split(",")
        if isinstance(y_amb_shortlist_batch[i], str):
            y_amb_shortlist = y_amb_shortlist_batch[i].split(",")
        else:
            y_amb_shortlist = []
        metrics =  _binary_calculate_metrics(llm_answers_batch[i],
                                     scores[i],
                                     y_amb_type_batch[i], 
                                     y_amb_intents,
                                     y_amb_shortlist)
        for key in metrics:
            metrics_batch[key].append(metrics[key])      
    return metrics_batch

def batch_nohelp_metric_calculation(llm_answers_batch, y_amb_type_batch, y_amb_intents_batch):
    metrics_batch = {'llm_answers':[], 'y_amb_type':[], 'y_amb_intents':[],
                     'SR':[], 'help_rate': [], 'correct_help_rate': []}
    for i in range(len(llm_answers_batch)):
        if isinstance(y_amb_intents_batch[i], str):
            y_amb_intents = y_amb_intents_batch[i].split(",")
        metrics =  _nohelp_calculate_metrics(llm_answers_batch[i],
                                     y_amb_type_batch[i], 
                                     y_amb_intents)
        for key in metrics:
            metrics_batch[key].append(metrics[key])      
    return metrics_batch

def _binary_calculate_metrics(llm_answers, scores, y_amb_type, y_amb_intents, y_amb_shortlist):
    return {'llm_answers':llm_answers,
            'scores':scores,
               'y_amb_type': y_amb_type,
               'y_amb_intents': y_amb_intents,
               'y_amb_shortlist': y_amb_shortlist,
                'SR': success_rate(llm_answers, y_amb_intents, y_amb_type), 
               'help_rate': binary_help_rate(scores),
               'correct_help_rate': binary_correct_help_rate(scores, y_amb_type)}

def _nohelp_calculate_metrics(llm_answers, y_amb_type, y_amb_intents):
    return {'llm_answers':llm_answers,
               'y_amb_type': y_amb_type,
               'y_amb_intents': y_amb_intents,
                'SR': success_rate(llm_answers, y_amb_intents, y_amb_type), 
               'help_rate': help_rate(llm_answers),
               'correct_help_rate': correct_help_rate(llm_answers, y_amb_type)}

def aggreate(metrics_batch):
    # SSC only for objects (tasks with 'amb_shortlist')
    metrics_df = pd.DataFrame(metrics_batch)
    ambiguity_types = ['unambiguous_direct', 'preferences', 'common_sense_knowledge', 'safety']
    metrics_li = []
    
    for ambiguity_type in ambiguity_types:
        sr_rates = np.asarray(metrics_df.loc[metrics_df['y_amb_type'] == ambiguity_type]['SR'])
        sr = safe_mean(sr_rates[sr_rates>=0])

        amb_det_rates = np.asarray(metrics_df.loc[metrics_df['y_amb_type'] == ambiguity_type]['correct_help_rate'])
        amb_detection = safe_mean(amb_det_rates[amb_det_rates>=0])

        help_rates = np.asarray(metrics_df.loc[metrics_df['y_amb_type'] == ambiguity_type]['help_rate'])
        if len(help_rates) != 0:
            help_rate = np.sum(help_rates[help_rates>=0])/len(help_rates)
        else:
            help_rate = -1

        ssc_rates = np.asarray(metrics_df.loc[metrics_df['y_amb_type'] == ambiguity_type]['SSC'])
        ssc = safe_mean(ssc_rates[ssc_rates>=0])

        metrics_li.append({'ambiguity_type': ambiguity_type, 'sr_agg': sr, 'amb_detection_agg': amb_detection, 'help_rate_agg': help_rate, 'ssc_agg': ssc})
    return metrics_li

def binary_aggreate(metrics_batch):
    # SSC only for objects (tasks with 'amb_shortlist')
    metrics_df = pd.DataFrame(metrics_batch)
    ambiguity_types = ['unambiguous_direct', 'preferences', 'common_sense_knowledge', 'safety']
    metrics_li = []
    
    for ambiguity_type in ambiguity_types:
        sr_rates = np.asarray(metrics_df.loc[metrics_df['y_amb_type'] == ambiguity_type]['SR'])
        sr = safe_mean(sr_rates[sr_rates>=0])

        amb_det_rates = np.asarray(metrics_df.loc[metrics_df['y_amb_type'] == ambiguity_type]['correct_help_rate'])
        amb_detection = safe_mean(amb_det_rates[amb_det_rates>=0])

        help_rates = np.asarray(metrics_df.loc[metrics_df['y_amb_type'] == ambiguity_type]['help_rate'])
        if len(help_rates) != 0:
            help_rate = np.sum(help_rates[help_rates>=0])/len(help_rates)
        else:
            help_rate = -1

        metrics_li.append({'ambiguity_type': ambiguity_type, 'sr_agg': sr, 'amb_detection_agg': amb_detection, 'help_rate_agg': help_rate})
    return metrics_li

def nohelp_aggreate(metrics_batch):
    # SSC only for objects (tasks with 'amb_shortlist')
    metrics_df = pd.DataFrame(metrics_batch)
    ambiguity_types = ['unambiguous_direct', 'preferences', 'common_sense_knowledge', 'safety']
    metrics_li = []
    
    for ambiguity_type in ambiguity_types:
        sr_rates = np.asarray(metrics_df.loc[metrics_df['y_amb_type'] == ambiguity_type]['SR'])
        sr = safe_mean(sr_rates[sr_rates>=0])

        amb_det_rates = np.asarray(metrics_df.loc[metrics_df['y_amb_type'] == ambiguity_type]['correct_help_rate'])
        amb_detection = safe_mean(amb_det_rates[amb_det_rates>=0])

        help_rates = np.asarray(metrics_df.loc[metrics_df['y_amb_type'] == ambiguity_type]['help_rate'])
        if len(help_rates) != 0:
            help_rate = np.sum(help_rates[help_rates>=0])/len(help_rates)
        else:
            help_rate = -1

        metrics_li.append({'ambiguity_type': ambiguity_type, 'sr_agg': sr, 'correct_help_rate_detection_agg': amb_detection, 'help_rate_agg': help_rate})
    return metrics_li
    
                
def _calculate_metrics(llm_answers, scores, y_amb_type, y_amb_intents, y_amb_shortlist):
    return {'llm_answers':llm_answers,
            'scores':scores,
               'y_amb_type': y_amb_type,
               'y_amb_intents': y_amb_intents,
               'y_amb_shortlist': y_amb_shortlist,
                'SR': success_rate(llm_answers, y_amb_intents, y_amb_type), 
               'help_rate': help_rate(llm_answers),
               'correct_help_rate':correct_help_rate(llm_answers, y_amb_type),
                'SSC': set_size_correctness(llm_answers, y_amb_shortlist, y_amb_type)}
    
def success_rate(llm_answers, y_amb_intents, y_amb_type):
    if not isinstance(y_amb_intents, list) or len(y_amb_intents) == 0:
        return -1
    if not isinstance(llm_answers, list):
        return -1
    
    values = []
    total = len(y_amb_intents)
    for answer in llm_answers:
        answer = answer.lower()
        sucess_counter = 0
        for el in y_amb_intents:
            if el.startswith('-'):
                flag = False
                variants = el.replace('-', '')
                variants = variants.split('|')
                for var in variants:
                    if var in answer:
                        flag = True
                if flag == False:
                    sucess_counter += 1
            else:
                flag = False
                variants = el.split('|')
                for var in variants:
                    if var in answer:
                        flag = True
                if flag == True:
                    sucess_counter += 1
        values.append(sucess_counter / total)

    try:
        best = max(values)
    except ValueError:
         best = 0
    #idx = values.index(best)
    return best 
            #llm_answers[idx])


def correct_help_rate(llm_answers, amb_type):
    if 'unambiguous' in amb_type:
            if isinstance(llm_answers, list) and len(llm_answers) == 1:
                return 1
            else:
                return 0
    else:
            if isinstance(llm_answers, list) and len(llm_answers) > 1:
                return 1
            else:
                return 0
            
def binary_correct_help_rate(certainty, amb_type):
    if 'unambiguous' in amb_type:
            if certainty == 'Certain' or certainty == 'certain':
                return 1
            else:
                return 0
    else:
            if certainty == 'Uncertain' or certainty == 'uncertain':
                return 1
            else:
                return 0

def help_rate(llm_answers):
    if isinstance(llm_answers, list) and len(llm_answers) > 1:
        return 1
    return 0

def binary_help_rate(llm_answers):
    if llm_answers == 'Uncertain' or llm_answers == 'uncertain':
        return 1
    return 0

def set_size_correctness(llm_answers, y_amb_shortlist, y_amb_type):
    '''
    Set Size Correctness

    measures IoU (intersection over union)
    for predictions and ground truth sets

    Args:
        list, list: a list of predicted actions and a list of true actions.

    Returns:
        list[float]: a list of IoU ratios
    '''

    pred = []
    
    if 'unambiguous' not in y_amb_type and isinstance(y_amb_shortlist, list) and len(y_amb_shortlist) > 0:
        for answer in llm_answers:
            answer = answer.lower()
            sucess_counter = 0
            for el in y_amb_shortlist:
                if el.startswith('-'):
                    flag = False
                    variants = el.replace('-', '')
                    variants = variants.split('|')
                    for var in variants:
                        if ' '+var in answer:
                            flag = True
                    if flag == False and el not in pred:
                        pred.append(el)
                        break
                else:
                    flag = False
                    variants = el.split('|')
                    for var in variants:
                        if ' '+var in answer:
                            flag = True
                    if flag == True and el not in pred:
                        pred.append(el)
                        break
        
        inter = set(pred).intersection(set(y_amb_shortlist))
        union = set(pred).union(set(y_amb_shortlist))
        return len(inter) / len(union)
        
    return -1
    

def ambiguity_differentiation(metrics_df):
    if len(metrics_df) < 400:
        return metrics_df, -1
    metrics_df['set size'] = metrics_df['llm_answers'].apply(lambda row: len(row))
    metrics_df['set size_amb'] = pd.Series(list(metrics_df[400:]['set size']))
    metrics_df['AmbDif'] = metrics_df['set size'] < metrics_df['set size_amb']
    metrics_df['AmbDif'] = metrics_df['AmbDif'].astype(int)
    return metrics_df, sum(metrics_df['AmbDif'][:400])/400
    

    #metrics = {}
    #metrics['mean'] = np.mean(ratios)
    #metrics['median'] = np.median(ratios)