import numpy as np
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, GemmaForCausalLM
import torch
import openai
from openai import OpenAI
import signal

openai_api_key = #"your-api-key"
openai.api_key = openai_api_key

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self):
        raise TimeoutException(self.error_message)

    def __enter__(self):
        self.timer = threading.Timer(self.seconds, self.handle_timeout)
        self.timer.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.timer.cancel()

def temperature_scaling(logits, temperature=1):
    logits = np.array(logits)
    logits /= temperature

    # apply softmax
    try:
        logits -= logits.max()
    except:
        logits = logits
    logits = logits - np.log(np.sum(np.exp(logits)))
    smx = np.exp(logits)
    smx = [float(x) for x in smx]
    return smx

class LLM:
    def __init__(self, model_name, generation_settings):
        self.device = "cuda" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mtype = None
        
        if "t5" in model_name or "bart" in model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.mtype ='d'
        elif 'gemma' in model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = GemmaForCausalLM.from_pretrained(model_name)
            self.mtype ='ed'
        elif 'turbo' or 'gpt-4' in model_name:
            self.model = model_name
            self.mtype = 'gpt_api'
            self.client = OpenAI(api_key=openai_api_key)
   # base_url='https://api.proxyapi.ru/openai/v1',

        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.mtype ='ed'
            
        if self.mtype != 'gpt_api':
            self.model.to(self.device)  # Move model to CUDA device

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        self.generation_settings = generation_settings
        
    def get_full_logits(self, logits, words):
        # Initialize an empty list to collect all token IDs
        token_ids = []
        filtered_logits = []
        # Tokenize each word and convert to IDs
        for word in words:
            word_tokens = self.tokenizer.tokenize(word)
            word_token_ids = self.tokenizer.convert_tokens_to_ids(word_tokens)
            token_ids.append(word_token_ids)
        
        for option_ids in token_ids:
            ids = [t.item() for t in torch.tensor(option_ids, device=self.device)]
            count_tokens = dict(Counter(ids).most_common())
            token_ids_target = [key for key in count_tokens.keys() if count_tokens[key] == 1]
            filtered_logits.append([logits[t].item() for t in token_ids_target])
        
        return dict(zip(words, filtered_logits))

    def filter_logits(self, logits, words, use_softmax=True):
        if self.mtype != 'gpt_api':
            # Initialize an empty list to collect all token IDs
            token_ids = []
            # Tokenize each word and convert to IDs
            for word in words:
                word_tokens = self.tokenizer.tokenize(word)
                word_token_ids = self.tokenizer.convert_tokens_to_ids(word_tokens)
                token_ids.extend(word_token_ids)
        
            token_ids = [t.item() for t in torch.tensor(token_ids, device=self.device)]
            
            count_tokens = dict(Counter(token_ids).most_common())
            token_ids_target = [key for key in count_tokens.keys() if count_tokens[key] == 1]
            filtered_logits = [logits[t].item() for t in token_ids_target]
            #print(filtered_logits)
            if use_softmax:
                filtered_logits = temperature_scaling(filtered_logits)
                #print(filtered_logits)
            return dict(zip(words, filtered_logits))
        else:
            filtered_logits = []
            filtered_letters = []
            for key, value in logits.items():
                for word in words:
                    if key == word:
                        filtered_logits.append(value)
                        filtered_letters.append(key)
            if use_softmax:
                filtered_logits = temperature_scaling(filtered_logits)
            #print('filtered_logits', filtered_logits)
            #print('filtered_letters', filtered_letters)
            return dict(zip(filtered_letters, filtered_logits))


    def generate(self, prompt, return_logits=False):
        if self.mtype != 'gpt_api':
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs['input_ids'].to(self.device)  # Move input_ids to CUDA device

            generated_ids = self.model.generate(input_ids, **self.generation_settings)
            generated_text = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids][0]
            if return_logits:
                with torch.no_grad():
                    if self.mtype=='ed':
                        combined_ids = torch.cat([input_ids, generated_ids[:, 1:]], dim=1)
                        outputs = self.model(combined_ids)
                        logits = outputs.logits[:, input_ids.size(1):]
                    else:
                        outputs = self.model(input_ids=input_ids, decoder_input_ids=generated_ids[:, :-1])
                        logits = outputs.logits

                    # Create a mask to filter out special tokens
                    special_tokens = self.tokenizer.all_special_ids
                    mask = torch.ones_like(generated_ids[0, 1:], dtype=torch.bool)
                    for token_id in special_tokens:
                        mask &= (generated_ids[0, 1:] != token_id)

                    filtered_logits = logits[:, mask] 
                return (generated_text, filtered_logits)
            else:
                return generated_text
        else:
            max_attempts = 3
            temperature=0
            timeout_seconds=40

            for _ in range(max_attempts):
                #try:
                with timeout(seconds=timeout_seconds):
                    response = self.client.chat.completions.create(
                        messages=[
                                        {
                                            'role': 'user',
                                            'content': prompt,
                                        }
                                    ],
                        model='gpt-3.5-turbo',
                        temperature=temperature,
                                    #top_p=0.1,
                                    n=1,
                        **self.generation_settings
                                    #stop=list(stop_seq) if stop_seq is not None else None,
                        )
                        
                    generated_text = response.choices[0].message.content.strip()
                    #print('stop_seq', stop_seq)
                    break
               # except:
                #    print('Timeout, retrying...')

            if return_logits:
                filtered_logits = {}
                        # Create a mask to filter out special tokens
                logits = response.choices[0].logprobs.content[0].top_logprobs
                for logit in logits:
                    filtered_logits[logit.token] =  logit.logprob
                return (generated_text, [[filtered_logits]])
            else:
                return generated_text
            #return response.choices[0].message.content.strip(), response
        
        #response, response["choices"][0]["text"].strip()

        
    def generate_batch(self, prompts, return_logits=False):
        if self.mtype != 'gpt_api':
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs['input_ids'].to(self.device)
        
            generated_ids = self.model.generate(input_ids, **self.generation_settings)
            generated_texts = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
        
            if return_logits:
                logits = self._compute_logits(input_ids, generated_ids)
                filtered_logits = self._filter_special_token_logits(generated_ids, logits)
                return (generated_texts, filtered_logits)
            else:
                return generated_texts
        else:
            generated_texts = []
            filtered_logits_li = []
            for prompt in prompts:
                temperature=0
                response = self.client.chat.completions.create(
                            messages=[
                                    {
                                        'role': 'user',
                                        'content': prompt,
                                    }
                                ],
                                model='gpt-3.5-turbo',
                                temperature=temperature,
                                top_p=0.1,
                                n=1,
                                **self.generation_settings
                                #stop=list(stop_seq) if stop_seq is not None else None,
                                )
                #print('stop_seq', stop_seq)
                
                generated_text = response.choices[0].message.content.strip()
                generated_texts.append(generated_text)

                if return_logits:
                    filtered_logits = {}
                        # Create a mask to filter out special tokens
                    logits = response.choices[0].logprobs.content[0].top_logprobs
                    for logit in logits:
                        filtered_logits[logit.token] =  logit.logprob
                    filtered_logits_li.append([filtered_logits])
            
            if return_logits:
                return (generated_texts, filtered_logits_li)
            else:
                return generated_texts


    def _compute_logits(self, input_ids, generated_ids):
        with torch.no_grad():
            if self.mtype == 'ed':
                combined_ids = torch.cat([input_ids, generated_ids[:, 1:]], dim=1)
                outputs = self.model(combined_ids)
                logits = outputs.logits[:, input_ids.size(1):]
            else:
                outputs = self.model(input_ids=input_ids, decoder_input_ids=generated_ids[:, :-1])
                logits = outputs.logits
        return logits
    
    def _filter_special_token_logits(self, generated_ids, logits):
        special_tokens = self.tokenizer.all_special_ids
        batch_size, seq_length = generated_ids.shape
        mask = torch.ones((batch_size, seq_length - 1), dtype=torch.bool, device=logits.device)
        for token_id in special_tokens:
            mask &= (generated_ids[:, 1:] != token_id)
        filtered_logits = torch.stack([logit[mask[i]] for i, logit in enumerate(logits)])
        return filtered_logits

if __name__ == "__main__":
    model = LLM("google/flan-t5-large", {"max_length": 50, "num_return_sequences": 1})
    print(model.generate("Choose one letter A/B/C?"))
    answer, logits = model.generate("Choose one letter A/B/C?", return_logits=True)
    print(model.filter_logits(logits[0][1], words=["A", "B", "C", "D", 'a', 'b', 'c', 'd', '1', '2', '3', '4']))




""""
           for _ in range(max_attempts):
                try:
                    with timeout(seconds=timeout_seconds):
                        response = self.client.chat.completions.create(
                        messages=[
                                {
                                    'role': 'user',
                                    'content': prompt,
                                }
                            ],
                            model='gpt-3.5-turbo',
                            temperature=temperature,
                            logprobs=logprobs,
                            max_tokens=max_tokens,
                            top_p=0.1,
                            n=1,
                            stop=list(stop_seq) if stop_seq is not None else None,
                            )
                         
                        break
                except:
                    print('Timeout, retrying...')
                    pass
                    

                        response = openai.Completion.create(
                        model='gpt-3.5-turbo',
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        logprobs=logprobs,
                        stop=list(stop_seq) if stop_seq is not None else None,
                    )
                    break if return_logits? #сразу фильтр!!!

"""
