import json
import os
import numpy as np
from nltk import word_tokenize

import torch
import transformers
from transformers import *

# from word_complexity import ComplexityRanker

transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)

def sent_encode(tokenizer, sent, device):
    # covert sentence to ids
    toks = [tokenizer.cls_token]
    word_index = [-1] # word index for toks
    
    ori_toks = word_tokenize(sent)
    
    for i, raw_tok in enumerate(ori_toks):
        new_tok = tokenizer.tokenize(raw_tok)
        toks += new_tok
        word_index.append([i for j in range(len(new_tok))])
    
    toks += [tokenizer.sep_token]
    word_index += [-1]
    
    input_ids = torch.LongTensor(tokenizer.convert_tokens_to_ids(toks)).unsqueeze(0).to(device)
    
    return input_ids, word_index


def output_selection(vec1, vec2):
    # max
    return True


def r_meaning_score(input_sent, output_sent, tokenizer, model, device):
    
    input_ids, input_index = sent_encode(tokenizer, input_sent, device)
    output_ids, output_index = sent_encode(tokenizer, output_sent, device)

    input_rep = model(input_ids)
    input_rep = input_rep[0].cpu().detach().numpy()
    output_rep = model(output_ids)
    output_rep = output_rep[0].cpu().detach().numpy()
    
    # todo: word_level
    # pooled_input = []
    # pooled_output = []
    
    r_sim = []
    
    for _, vec in enumerate(input_rep[0]):
        o_sim = []
        for i, cand in enumerate(output_rep[0]):
            sim = np.dot(cand, vec)/(np.linalg.norm(cand)*np.linalg.norm(vec))
            o_sim.append(sim)
            
        r_sim.append(max(o_sim))
        
    return(sum(r_sim)/len(r_sim))


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    model = "bert-large-uncased"

    tokenizer = BertTokenizer.from_pretrained(model)
    lm_model =  BertModel.from_pretrained(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm_model.to(device)

    input_sent = "About 95 species are currently accepted ."
    output_sent = "About 95 species are now accepted ."

    print("testing started...")
    test = r_meaning_score(input_sent, output_sent, tokenizer, lm_model, device)
    
    print(test)