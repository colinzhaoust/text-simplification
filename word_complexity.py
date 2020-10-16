import ujson as json
import logging
import argparse
import os
import math

import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

import numpy
import collections
import random
from tqdm import tqdm

from transformers import *
import pandas as pd
import numpy as np
import xml.etree.ElementTree as etree

from sklearn.utils import shuffle
from nltk.stem import WordNetLemmatizer
from scipy.stats import pearsonr
from sklearn.metrics import precision_score, accuracy_score


class DataLoader:
    def __init__(self, train_data_path, dev_data_path, test_data_path, args):
        self.args = args
        self.wnl = WordNetLemmatizer()
        self.threshold = 0.4
        self.tokenizer = BertTokenizer.from_pretrained(args.model)

        print("Loading the SimplePPDB++ dataset...")
        # GET self.ppdb
        self.ppdb = self.simpleppdb_loading(train_data_path)

        if dev_data_path is not None:
            self.dev_ppdb = self.simpleppdb_loading(dev_data_path)
        print("Finished...")

        # GET self.gold_rankings
        print("Loading SemEval 2012 goldrankings...")
        # already prepared
        with open(test_data_path, "r") as f:
            self.semeval = json.load(f)
        print("Finished...")

        print("Caching the Examples")
        self.cache_cleaned_data(self.ppdb, "./cleaned_ppdb.json")
        print("Finished")

        print("Tensorizing the example...")
        self.set = dict()
        self.set["ppdb"] = self.tensorize_ppdb_dataset(self.ppdb)
        self.set["semeval"] = self.tensorize_semeval_dataset(self.semeval)
        print("Finished")

    def simpleppdb_loading(self, datapath):
        with open(datapath, 'r') as f:
            ppdb = [] # word1, word2, label
            raw_ppdb = f.readlines()
            for line in raw_ppdb:
                temp_datapiece = line.rstrip().split("\t")
                if self.args.do_clean == True:
                    if not self.simpleppdb_cleaning(temp_datapiece):
                        score = float(temp_datapiece[2]) # simple ppdb score
                        if -1*self.threshold < score < self.threshold:
                            label = 0.0
                        elif score < -self.threshold:
                            label = -1.0
                        else:
                            label = 1.0
                        ppdb.append((temp_datapiece[0],temp_datapiece[1],label))
                else:
                    score = float(temp_datapiece[2]) # simple ppdb score
                    if -1*self.threshold < score < self.threshold:
                        label = 0.0
                    elif score < -self.threshold:
                        # A is less complicated
                        label = -1.0
                    else:
                        label = 1.0
                    ppdb.append((temp_datapiece[0],temp_datapiece[1],label))
        return ppdb

    def simpleppdb_cleaning(self, datapiece):
        # clean the not useful pieces
        if datapiece[0] == datapiece[1]:
            return True
        elif "www" in datapiece[0]:
            return True
        # elif self.wnl(datapiece[0]) == self.wnl(datapiece[1]):
        #     return True
        else:
            return False

    def cache_cleaned_data(self, datalist, name):
        # cleaned_ppdb.json
        # cleaned semeval_rankings.json
        with open(name, "w") as f:
            json.dump(datalist, f)


    def tensorize_ppdb_dataset(self, ppdb):
        # tensorize the examples
        tensorized_dataset = list()
        
        for instance in ppdb:
            one_example = self.tensorize_one_exampe(instance)

            tensorized_dataset += one_example

        return tensorized_dataset


    def tensorize_semeval_dataset(self, semeval):
        tensorized_dataset = list()

        for inst_id in range(301,2011): # include 2010
            instance_pairs = {}

            instance = semeval[str(inst_id)]
            target = instance["target"]
            candidates = instance["candidates"] # list
            ranks = instance["rank_cands"]
            label = -2.0


            instance_pairs["pairs"] = []

            for cand in candidates:
                one_example = self.tensorize_one_exampe((target, cand, label))

                instance_pairs["pairs"].append(one_example)
            
            instance_pairs["inst_id"] = inst_id
            instance_pairs["ranks"] = ranks
            instance_pairs["cands"] = candidates

            tensorized_dataset += instance_pairs
            
            
    def tensorize_one_exampe(self, instance):
        # label: -1, 0, 1  for ppdb; -2 for semeval as the target is diff
        one_example = []
        token1 = instance[0]
        token2 = instance[1]
        label = instance[2]

        tokenized_token1 = self.tokenizer.encode(token1, add_special_tokens=True, return_tensors='pt')
        tokenized_token2 = self.tokenizer.encode(token2, add_special_tokens=True, return_tensors='pt')

        one_example.append(
            {   'token1':torch.tensor(tokenized_token1).to(device),
                'token2': torch.tensor(tokenized_token2).to(device),
                'label': torch.FloatTensor([int(label)]).to(device)
                })

        return one_example

class BertEncoder(BertModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.lm = BertModel(config)
        # self.embedding_size = 300

    def forward(self, sents):
        # forwarding the sents and use the average embedding as the results

        representation  = self.lm(sents) #.unsqueeze(0)) # num_sent * sent_len * emb
        # print(representation[0].size)
        sent_representation = torch.mean(representation[0], dim=1) # num_sent * emb
        # print(sent_representation.size)
        overall_representation = torch.mean(sent_representation, dim=0) # 1 *  emb
        # output size: 1024

        return overall_representation


class ComplexityRanker(torch.nn.Module):

    def __init__(self, config):
        super(ComplexityRanker, self).__init__()

        self.bert = BertEncoder.from_pretrained(config) # output: 1024

        for param in self.bert.parameters():
            param.requires_grad = False

        self.text_specific = torch.nn.Linear(1024, 256)

        d_in, h, d_out = 256*2, 300, 1 
        drop_out = 0.2

        self.classification = torch.nn.Sequential(
                torch.nn.Linear(d_in, h),
                torch.nn.Tanh(),
                torch.nn.Dropout(p=drop_out),
                torch.nn.Linear(h, h),
                torch.nn.Tanh(),
                torch.nn.Dropout(p=drop_out),
                torch.nn.Linear(h, h),
                torch.nn.Tanh(),
                torch.nn.Dropout(p=drop_out),
                torch.nn.Linear(h, h),
                torch.nn.Tanh(),
                torch.nn.Dropout(p=drop_out),
                torch.nn.Linear(h, d_out)
        )

    def forward(self, token1, token2):
        rep1 = self.text_specific(self.bert(token1))
        rep2 = self.text_specific(self.bert(token2))

        cls_rep = torch.cat([rep1, rep2])
        predictions = self.classification(cls_rep)

        return predictions


def train(model, data):
    all_loss = 0
    print('training:')
    random.shuffle(data)
    model.train()
    selected_data = data
    for tmp_example in tqdm(selected_data):
        # print(tmp_example)
        prediction = model(token1=tmp_example['token1'], token2=tmp_example['token2']) 
        loss = loss_func(prediction, tmp_example['label'])
        test_optimizer.zero_grad()
        loss.backward()
        test_optimizer.step()
        all_loss += loss.item()
    print('current loss:', all_loss / len(data))


def test_on_sem(model, data):
    # token1
    correct_count = 0
    total = 0
    # print('Testing:')
    model.eval()
    for tmp_example in tqdm(data):
        ranks = tmp_example["ranks"]
        cands = tmp_example["cands"]
        predictions = []

        for cand_pair in tmp_example["pairs"]:
            prediction = model(token1=cand_pair['token1'], token2=cand_pair['token2']) 
            predictions.append(prediction.data.numpy())

        # p@1
        pred1 = cands[np.argsort(predictions)[-1]]
        if pred1 in ranks[0]:
            correct_count += 1

        total += 1

    return 1.0 * correct_count / total


# todo
# def eval_on_ppdb


# todo
# def eval_on_simplicity

parser = argparse.ArgumentParser()

## parameters
parser.add_argument("--gpu", default='0', type=str, required=False,
                    help="choose which gpu to use")
parser.add_argument("--model", default='bert-large-uncased', type=str, required=False,
                    help="choose the model to test")
parser.add_argument("--lr", default=0.0001, type=float, required=False,
                    help="initial learning rate")
parser.add_argument("--lrdecay", default=0.8, type=float, required=False,
                    help="learning rate decay every 5 epochs")
parser.add_argument("--do_clean", default=True, type=bool, required=False,
                    help="if we do cleaning on the simpleppdb")
parser.add_argument("--train_bert", default=False, type=bool, required=False,
                    help="if we do cleaning on the simpleppdb")
parser.add_argument("--max_len", default=5, type=int, required=False,
                    help="number of words")
parser.add_argument("--epochs", default=15, type=int, required=False,
                    help="number of epochs")


args = parser.parse_args()

logging.basicConfig(level=logging.INFO)

if args.gpu == -1:
    device = torch.device("cpu")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# device = torch.device("cpu")

print('current device:', device)
# n_gpu = torch.cuda.device_count()
# print('number of gpu:', n_gpu)
# torch.cuda.get_device_name(0)


current_model = ComplexityRanker("bert-large-uncased")

current_model.to(device)
test_optimizer = torch.optim.Adam(current_model.parameters(), lr=args.lr)
loss_func = torch.nn.MSELoss() # torch.nn.CrossEntropyLoss()

torch.save(current_model, "test_save.ckpt")

all_data = DataLoader(train_data_path="./datasets/ppdb/simpleppdbpp-s-lexical.txt", dev_data_path=None, test_data_path= "./datasets/semeval/semeval_rankings.json", args=args)

best_dev_performance = 0
final_performance = 0

for i in range(args.epochs):
    print('Iteration:', i + 1, '|', 'Current best performance:', final_performance)
    train(current_model, all_data.set["ppdb"])
    test_performance = test_on_sem(current_model, all_data.set["semeval"])
    print('Test accuracy:', test_performance)
    if test_performance >= best_dev_performance:
        print('New best performance!!!')
        best_dev_performance = test_performance
        torch.save(current_model, "./best_model_"+str(best_dev_performance)+".ckpt")
        final_performance = test_performance

print("Best performance:", final_performance)

print('end')
