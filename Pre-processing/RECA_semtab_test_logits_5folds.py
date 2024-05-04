import os

MAX_LEN = 512
SEP_TOKEN_ID = 102
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pandas as pd
from tqdm import trange
from os.path import join
import tqdm
import time
import json
import numpy as np
import random
import torch
import pickle
import functools
from transformers import BertTokenizer
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
import jsonlines
from transformers import BertModel, BertForSequenceClassification, BertConfig
from tqdm import tqdm
from math import sqrt

NERs = {'PERSON1': 0, 'PERSON2': 1, 'NORP': 2, 'FAC': 3, 'ORG': 4, 'GPE': 5, 'LOC': 6, 'PRODUCT': 7, 'EVENT': 8,
        'WORK_OF_ART': 9, 'LAW': 10, 'LANGUAGE': 11, 'DATE1': 12, 'DATE2': 13, 'DATE3': 14, 'DATE4': 15, 'DATE5': 16,
        'TIME': 17, 'PERCENT': 18, 'MONEY': 19, 'QUANTITY': 20, 'ORDINAL': 21, 'CARDINAL': 22, 'EMPTY': 23}


def load_jsonl(jsonl_path, label_dict):
    target_cols = []
    labels = []
    rel_cols = []
    sub_rel_cols = []
    one_hot = []
    headers_alias = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
                     "U", "V", "W", "X", "Y", "Z"]
    mapping = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "J": 9, "K": 10, "L": 11,
               "M": 12, "N": 13, "O": 14, "P": 15, "Q": 16, "R": 17, "S": 18, "T": 19, "U": 20, "V": 21, "W": 22,
               "X": 23, "Y": 24, "Z": 25}
    with open(jsonl_path, "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            target_cols.append(np.array(item['content'])[:, int(item['target'])])
            target_alias = headers_alias[int(item['target'])]
            labels.append(int(label_dict[item['label']]))
            cur_rel_cols = []
            cur_sub_rel_cols = []
            for rel_col in item['related_cols']:
                cur_rel_cols.append(np.array(rel_col))
            for sub_rel_col in item['sub_related_cols']:
                cur_sub_rel_cols.append(np.array(sub_rel_col))
            rel_cols.append(cur_rel_cols)
            sub_rel_cols.append(cur_sub_rel_cols)
    return target_cols, rel_cols, sub_rel_cols, labels


class TableDatasetold(Dataset):
    def __init__(self, target_cols, rel_cols, sub_rel_cols, tokenizer, labels):
        self.labels = labels
        self.target_cols = target_cols
        self.tokenizer = tokenizer
        self.rel_cols = rel_cols
        self.sub_rel_cols = sub_rel_cols

    def tokenize(self, text):
        text = list(text)
        print(type(text))
        tokenized_text = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=MAX_LEN,
                                                    padding='max_length', truncation=True)
        ids = torch.Tensor(tokenized_text["input_ids"]).long()
        return ids

    def tokenize_set(self, cols):
        text = ''
        for i, col in enumerate(cols):
            for cell in col:
                text += cell
                text += ' '
            if not i == len(cols) - 1:
                text += '[SEP]'
        tokenized_text = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=MAX_LEN,
                                                    padding='max_length', truncation=True)
        ids = torch.Tensor(tokenized_text["input_ids"]).long()
        return ids

    def tokenize_set_equal(self, cols):
        init_text = ''
        for i, col in enumerate(cols):
            for cell in col:
                init_text += cell
                init_text += ' '
            if not i == len(cols) - 1:
                init_text += '[SEP]'
        total_length = len(self.tokenizer.tokenize(init_text))
        if total_length <= MAX_LEN:
            tokenized_text = self.tokenizer.encode_plus(init_text, add_special_tokens=True, max_length=MAX_LEN,
                                                        padding='max_length', truncation=True)
        else:
            ratio = MAX_LEN / total_length
            text = ''
            for i, col in enumerate(cols):
                for j, cell in enumerate(col):
                    if j > len(col) * ratio:
                        break
                    text += cell
                    text += ' '
                if not i == len(cols) - 1:
                    text += '[SEP]'
            tokenized_text = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=MAX_LEN,
                                                        padding='max_length', truncation=True)
        ids = torch.Tensor(tokenized_text["input_ids"]).long()
        return ids

    def __getitem__(self, idx):
        target_token_ids = self.tokenize(self.target_cols[idx])
        if len(self.rel_cols[idx]) == 0:
            rel_token_ids = target_token_ids
        else:
            rel_token_ids = self.tokenize_set_equal(self.rel_cols[idx])
        if len(self.sub_rel_cols[idx]) == 0:
            sub_token_ids = target_token_ids
        else:
            sub_token_ids = self.tokenize_set_equal(self.sub_rel_cols[idx])

        return target_token_ids, rel_token_ids, sub_token_ids, torch.tensor(self.labels[idx])

    def __len__(self):
        return len(self.labels)

    def collate_fn(self, batch):
        token_ids = torch.stack([x[0] for x in batch])
        rel_ids = torch.stack([x[1] for x in batch])
        sub_ids = torch.stack([x[2] for x in batch])
        labels = torch.stack([x[3] for x in batch])
        return token_ids, rel_ids, sub_ids, labels


class TableDataset(Dataset):  # Generate tabular dataset
    def __init__(self, target_cols, rel_cols, sub_rel_cols, tokenizer, labels, file_name, col_ids):
        self.labels = []
        self.data = []
        self.tokenizer = tokenizer
        self.rel = []
        self.sub = []
        self.file_name = file_name
        self.col_id = col_ids
        for i in trange(len(labels)):
            self.labels.append(torch.tensor(labels[i]))
            target_token_ids = self.tokenize(target_cols[i])
            self.data.append(target_token_ids)
            if len(rel_cols[i]) == 0:  # If there is no related tables, use the target column content
                rel_token_ids = target_token_ids
            else:
                rel_token_ids = self.tokenize_set_equal(rel_cols[i])
            self.rel.append(rel_token_ids)
            if len(sub_rel_cols[i]) == 0:  # If there is no sub-related tables, use the target column content
                sub_token_ids = target_token_ids
            else:
                sub_token_ids = self.tokenize_set_equal(sub_rel_cols[i])
            self.sub.append(sub_token_ids)

    def tokenize(self, text):
        tokenized_text = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=MAX_LEN,
                                                    padding='max_length', truncation=True)
        ids = torch.Tensor(tokenized_text["input_ids"]).long()
        return ids

    def tokenize_set_equal(self, cols):  # Assigning the tokens equally to each identified column
        init_text = ''
        for i, col in enumerate(cols):
            for cell in col:
                init_text += cell
                init_text += ' '
            if not i == len(cols) - 1:
                init_text += '[SEP]'
        total_length = len(self.tokenizer.tokenize(init_text))
        if total_length <= MAX_LEN:
            tokenized_text = self.tokenizer.encode_plus(init_text, add_special_tokens=True, max_length=MAX_LEN,
                                                        padding='max_length', truncation=True)
        else:
            ratio = MAX_LEN / total_length
            text = ''
            for i, col in enumerate(cols):
                for j, cell in enumerate(col):
                    if j > len(col) * ratio:
                        break
                    text += cell
                    text += ' '
                if not i == len(cols) - 1:
                    text += '[SEP]'
            tokenized_text = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=MAX_LEN,
                                                        padding='max_length', truncation=True)
        ids = torch.Tensor(tokenized_text["input_ids"]).long()
        return ids

    def __getitem__(self, idx):
        return self.data[idx], self.rel[idx], self.sub[idx], self.labels[idx], self.file_name[idx], self.col_id[idx]

    def __len__(self):
        return len(self.labels)

    def collate_fn(self, batch):
        token_ids = torch.stack([x[0] for x in batch])
        rel_ids = torch.stack([x[1] for x in batch])
        sub_ids = torch.stack([x[2] for x in batch])
        labels = torch.stack([x[3] for x in batch])
        file_names = [x[4] for x in batch]
        col_ids = [x[5] for x in batch]
        return token_ids, rel_ids, sub_ids, labels, file_names, col_ids

def get_loader_load(target_cols, rel_cols, sub_rel_cols, labels, batch_size=8, is_train=True):
    ds_df = TableDataset(target_cols, rel_cols, sub_rel_cols, Tokenizer, labels)
    loader = torch.utils.data.DataLoader(ds_df, batch_size=batch_size, shuffle=is_train, num_workers=0,
                                         collate_fn=ds_df.collate_fn)
    loader.num = len(ds_df)
    return loader

def get_loader(path, batch_size, is_train): # Generate the dataloaders for the training process
    dataset = torch.load(path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=0, collate_fn=dataset.collate_fn)
    loader.num = len(dataset)
    return loader

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class KREL(torch.nn.Module):
    def __init__(self, n_classes=275, dim_k=768, dim_v=768):
        super(KREL, self).__init__()
        self.model_name = 'KREL'
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = torch.nn.Dropout(0.3)
        self.fcc_tar = torch.nn.Linear(768, n_classes)
        self.fcc_rel = torch.nn.Linear(768, n_classes)
        self.fcc_sub = torch.nn.Linear(768, n_classes)
        self.weights = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(1)) for i in range(3)])

    def encode(self, target_ids, rel_ids, sub_ids):
        att_tar = (target_ids > 0)
        _, tar = self.bert_model(input_ids=target_ids, attention_mask=att_tar, return_dict=False)
        att_rel = (rel_ids > 0)
        _, rel = self.bert_model(input_ids=rel_ids, attention_mask=att_rel, return_dict=False)
        att_sub = (sub_ids > 0)
        _, sub = self.bert_model(input_ids=sub_ids, attention_mask=att_sub, return_dict=False)

        return tar, rel, sub

    def forward(self, tar_ids, rel_ids, sub_ids):
        tar, rel, sub = self.encode(tar_ids, rel_ids, sub_ids)
        tar_out = self.dropout(tar)
        rel_out = self.dropout(rel)
        sub_out = self.dropout(sub)
        out_tar = self.fcc_tar(tar_out)
        out_rel = self.fcc_rel(rel_out)
        out_sub = self.fcc_sub(sub_out)
        res = self.weights[0] * out_tar + self.weights[1] * out_rel + self.weights[2] * out_sub
        return res


def metric_fn(preds, labels):
    weighted = f1_score(labels, preds, average='weighted')
    macro = f1_score(labels, preds, average='macro')
    return {
        'weighted_f1': weighted,
        'macro_f1': macro
    }


def test_model(model, test_loader, lr, new_dict, model_save_path='.pkl', early_stop_epochs=5, epochs=20):
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    bar = tqdm(test_loader)
    pred_labels = []
    true_labels = []
    dfs = []
    with torch.no_grad():
        for i, (ids, rels, subs, labels, filenames, col_ids) in enumerate(bar):
        # for i, (ids, rels, subs, labels) in enumerate(bar):
            labels = labels.to(device)  # .cuda()
            rels = rels.to(device)  # .cuda()
            subs = subs.to(device)  # .cuda()
            output = model(ids.to(device), rels, subs)
            y_pred_prob = output
            batch_data = {
                'filename': filenames,  # Assuming filenames is already a list or similar iterable
                'label': labels.cpu().numpy(),
                'output': [list(row) for row in y_pred_prob.cpu().numpy()]
            }
            dfs.append(pd.DataFrame(batch_data))
            y_pred_label = y_pred_prob.argmax(dim=1)
            pred_labels.append(y_pred_label.detach().cpu().numpy())
            true_labels.append(labels.detach().cpu().numpy())
            del ids, rels, subs
            torch.cuda.empty_cache()
    df_results = pd.concat(dfs, ignore_index=True)
    df_grouped = df_results.groupby('filename').agg({
        'label': lambda x: list(x),
        'output': lambda x: list(x)
    }).reset_index()
    print(df_grouped.shape)
    df_grouped['label'] = df_grouped['label'].apply(np.array)
    df_grouped['output'] = df_grouped['output'].apply(np.array)
    total_data = []
    for i in range(df_grouped.shape[0]):
        col_count = len(df_grouped.loc[i, 'label'])
        mask = np.zeros(7, dtype=int)
        mask[:col_count].fill(1)
        tmp_mask = mask
        features = df_grouped.loc[i, 'output']
        second_dimension = features.shape[1]
        new_features = np.zeros((7, second_dimension))
        new_features[:features.shape[0], :] = features
        new_labels = np.ones(7, dtype=int) * -1
        new_labels[:col_count] = df_grouped.loc[i, 'label']
        tmp_dict = {'features': new_features, 'labels': new_labels, 'masks': tmp_mask, 'table_id': df_grouped.loc[i, 'filename']}
        total_data.append(tmp_dict)
        # if i < 3:
        #     print(type(df_grouped.loc[i, 'output']), df_grouped.loc[i, 'output'])
        #     print(type(df_grouped.loc[i, 'label']), df_grouped.loc[i, 'label'])
        #     print(type(tmp_mask), tmp_mask)
    total_data = np.array(total_data)
    # for j in range(5):
    #     print(df_grouped.loc[j, 'filename'])
    #     print(df_grouped.loc[j, 'label'])
    #     print(len(df_grouped.loc[j, 'output']))

    pred_labels = np.concatenate(pred_labels)
    true_labels = np.concatenate(true_labels)
    f1_scores = metric_fn(pred_labels, true_labels)
    print("weighted f1:", f1_scores['weighted_f1'], "\t", "macro f1:", f1_scores['macro_f1'])
    return f1_scores['weighted_f1'], f1_scores['macro_f1'], total_data


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    setup_seed(20)
    base_dir = "/localhome/ehoseinz/PycharmProjects/RECA-paper-main"
    logit_loc = join(base_dir, 'data', 'Semtab', 'probability')
    if not os.path.exists(logit_loc):
        os.makedirs(logit_loc)
    with open('./semtab_labels.json', 'r') as dict_in:
        label_dict = json.load(dict_in)
    Tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    new_dict = {v: k for k, v in label_dict.items()}
    BS = 8
    lrs = [1e-5]
    # test_loader_path = '../data/tokenized_data/test_' + str(MAX_LEN) + 'GNN'
    # test_loader = get_loader(path=test_loader_path, batch_size=1, is_train=False)

    print("###############################")
    for lr in lrs:
        rounds = [0,1,2,3,4]
        print("start for testing learning rate:", lr)
        weighted_f1s_train = []
        macro_f1s_train = []
        weighted_f1s_valid = []
        macro_f1s_valid = []
        weighted_f1s_test = []
        macro_f1s_test = []
        for cur_fold in rounds:
            if True:
                train_loader_path = '../data/tokenized_data/train_'+str(MAX_LEN)+'_fold_'+str(cur_fold)+'5fold'+'GNN'
                valid_loader_path = '../data/tokenized_data/valid_'+str(MAX_LEN)+'_fold_'+str(cur_fold)+'5fold'+'GNN'
                val_loader = get_loader(path=valid_loader_path, batch_size=1, is_train=False)
                train_loader = get_loader(path=train_loader_path, batch_size=1, is_train=True)
                test_loader_path = '../data/tokenized_data/test_'+str(MAX_LEN)+'_fold_'+str(cur_fold)+'5fold'+'GNN'
                test_loader = get_loader(path=test_loader_path, batch_size=1, is_train=False)
                model = KREL().to(device)  # .cuda()
                # model_save_path = '../checkpoints/semtab-RECA'+"_lr="+str(lr)+'_bs='+str(BS)+'_max='+str(MAX_LEN)+'_{}.pkl'.format(cur_fold+1)
                model_save_path = '../checkpoints/semtab-RECA'+"_lr="+str(lr)+'_bs='+str(BS)+'_max='+str(MAX_LEN)+'_{}_5folds.pkl'.format(cur_fold+1)
                print("Starting fold", cur_fold + 1)
                cur_w_train, cur_m_train, total_train = test_model(model, train_loader, lr, new_dict, model_save_path=model_save_path)
                weighted_f1s_train.append(cur_w_train)
                macro_f1s_train.append(cur_m_train)
                with open(join(logit_loc, 'dataset_semtab_{}_5folds_train'.format(cur_fold)), "wb") as fp:  # Pickling
                    pickle.dump(total_train, fp)
                cur_w_val, cur_m_val, total_valid = test_model(model, val_loader, lr, new_dict, model_save_path=model_save_path)
                weighted_f1s_valid.append(cur_w_val)
                macro_f1s_valid.append(cur_m_val)
                with open(join(logit_loc, 'dataset_semtab_{}_5folds_validation'.format(cur_fold)), "wb") as fp:  # Pickling
                    pickle.dump(total_valid, fp)
                cur_w_test, cur_m_test, total_test = test_model(model, test_loader, lr, new_dict, model_save_path=model_save_path)
                weighted_f1s_test.append(cur_w_test)
                macro_f1s_test.append(cur_m_test)
                with open(join(logit_loc, 'dataset_semtab_{}_5folds_test'.format(cur_fold)), "wb") as fp:  # Pickling
                    pickle.dump(total_test, fp)

        print("The mean F1 score of train is:", np.mean(weighted_f1s_train))
        print("The sd of train is:", np.std(weighted_f1s_train))
        print("The mean macro F1 score of train is:", np.mean(macro_f1s_train))
        print("The sd of train is:", np.std(macro_f1s_train))
        print("===============================")

        print("The mean F1 score of valid is:", np.mean(weighted_f1s_valid))
        print("The sd of valid is:", np.std(weighted_f1s_valid))
        print("The mean macro F1 score of valid is:", np.mean(macro_f1s_valid))
        print("The sd of valid is:", np.std(macro_f1s_valid))
        print("===============================")

        print(weighted_f1s_test)
        print(macro_f1s_test)

        print("The mean F1 score of test is:", np.mean(weighted_f1s_test))
        print("The sd of test is:", np.std(weighted_f1s_test))
        print("The mean macro F1 score of test is:", np.mean(macro_f1s_test))
        print("The sd of test is:", np.std(macro_f1s_test))
        print("===============================")


