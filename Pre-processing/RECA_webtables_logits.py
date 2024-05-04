import os

MAX_LEN = 128
SEP_TOKEN_ID = 102
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import pickle
from os.path import join
import tqdm
import time
import json
import numpy as np
import random
import torch
import functools
from transformers import BertTokenizer
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
import jsonlines
from transformers import BertModel, BertForSequenceClassification, BertConfig
from tqdm import tqdm
from tqdm import trange
from math import sqrt
import pandas as pd

NERs = {'PERSON1': 0, 'PERSON2': 1, 'NORP': 2, 'FAC': 3, 'ORG': 4, 'GPE': 5, 'LOC': 6, 'PRODUCT': 7, 'EVENT': 8,
        'WORK_OF_ART': 9, 'LAW': 10, 'LANGUAGE': 11, 'DATE1': 12, 'DATE2': 13, 'DATE3': 14, 'DATE4': 15, 'DATE5': 16,
        'TIME': 17, 'PERCENT': 18, 'MONEY': 19, 'QUANTITY': 20, 'ORDINAL': 21, 'CARDINAL': 22, 'EMPTY': 23}


def setup_seed(seed):  # Set up random seeds for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class TableDataset(Dataset):  # Generate tabular dataset
    def __init__(self, target_cols, rel_cols, sub_rel_cols, tokenizer, labels, file_name, col_id):
        self.labels = []
        self.target_cols = []
        self.tokenizer = tokenizer
        self.rel_cols = []
        self.sub_rel_cols = []
        self.file_names = file_name
        self.col_ids = col_id
        for i in trange(len(labels)):
            self.labels.append(torch.tensor(labels[i]))
            target_token_ids = self.tokenize(target_cols[i])
            self.target_cols.append(target_token_ids)
            if len(rel_cols[i]) == 0:  # If there is no related tables, use the target column content
                rel_token_ids = target_token_ids
            else:
                rel_token_ids = self.tokenize_set_equal(rel_cols[i])
            self.rel_cols.append(rel_token_ids)
            if len(sub_rel_cols[i]) == 0:  # If there is no sub-related tables, use the target column content
                sub_token_ids = target_token_ids
            else:
                sub_token_ids = self.tokenize_set_equal(sub_rel_cols[i])
            self.sub_rel_cols.append(sub_token_ids)

    def tokenize(self, col):  # Normal practice of tokenization
        text = ''
        for cell in col:
            text += cell
            text += ' '
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
        return self.target_cols[idx], self.rel_cols[idx], self.sub_rel_cols[idx], self.labels[idx], self.file_names[idx], self.col_ids[idx]

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
def get_loader(path, batch_size, is_train):  # Generate the dataloaders for the training process
    dataset = torch.load(path)
    # print(dataset.filename)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=0,
                                         collate_fn=dataset.collate_fn)
    loader.num = len(dataset)
    return loader


class KREL(torch.nn.Module):
    def __init__(self, n_classes=78):
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


def metric_fn(preds, labels):  # The Support-weighted F1 score and Macro Average F1 score
    weighted = f1_score(labels, preds, average='weighted')
    macro = f1_score(labels, preds, average='macro')
    return {
        'weighted_f1': weighted,
        'macro_f1': macro
    }


def test_model(model, val_loaders, model_save_path):
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    with torch.no_grad():
        pred_labels = []
        true_labels = []
        dfs = []
        for val_loader in val_loaders:
            bar2 = tqdm(val_loader)
            for i, (ids, rels, subs, labels, filenames, col_ids) in enumerate(bar2):
            # for j, (ids, rels, subs, labels) in enumerate(bar2):
                labels = labels.to(device)  # .cuda()
                rels = rels.to(device)  # .cuda()
                subs = subs.to(device)  # .cuda()
                output = model(ids.to(device), rels, subs)
                y_pred_prob = output
                # print(filenames)
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
            mask = np.zeros(6, dtype=int)
            mask[:col_count].fill(1)
            tmp_mask = mask
            features = df_grouped.loc[i, 'output']
            second_dimension = features.shape[1]
            new_features = np.zeros((6, second_dimension))
            new_features[:features.shape[0], :] = features
            new_labels = np.ones(6, dtype=int) * -1
            new_labels[:col_count] = df_grouped.loc[i, 'label']
            tmp_dict = {'features': new_features, 'labels': new_labels, 'masks': tmp_mask,
                        'table_id': df_grouped.loc[i, 'filename']}
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
    base_dir = "/localhome/ehoseinz/PycharmProjects/RECA-paper-main"
    logit_loc = join(base_dir, 'data', 'WebTables', 'probability')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    setup_seed(20)
    with open('./label_dict.json', 'r') as dict_in:
        label_dict = json.load(dict_in)  # load the labels
    rounds = [0, 1, 2, 3, 4]

    BS = 8
    lr = 1e-5
    print('start loading data')

    weighted_f1s_train = []
    macro_f1s_train = []
    weighted_f1s_test = []
    macro_f1s_test = []

    for round in rounds:
        train_loaders = []
        test_loaders = []
        for fold_idx in range(5):
            loader_path = '../data/tokenized_data/' + str(MAX_LEN) + '/fold_' + str(fold_idx) + 'GNN'
            if fold_idx == round:
                test_loader = get_loader(path=loader_path, batch_size=BS, is_train=False)
                test_loaders.append(test_loader)
            else:
                train_loaders.append(get_loader(path=loader_path, batch_size=BS, is_train=True))

        print('start testing fold', round, 'learning rate', lr, 'batch size', BS, 'max length', MAX_LEN)
        model = KREL().cuda()
        model_save_path = '../checkpoints/webtables-RECA' + "_lr=" + str(lr) + '_bs=' + str(BS) + '_max=' + str(MAX_LEN) + '_{}'.format(round)
        # cur_w_train, cur_m_train, total_train = test_model(model, train_loaders, model_save_path)
        # weighted_f1s_train.append(cur_w_train)
        # macro_f1s_train.append(cur_m_train)
        # with open(join(logit_loc, 'dataset_webtables_{}_train'.format(round)), "wb") as fp:  # Pickling
            # pickle.dump(total_train, fp)
        cur_w_test, cur_m_test, total_test = test_model(model, test_loaders, model_save_path)
        weighted_f1s_test.append(cur_w_test)
        macro_f1s_test.append(cur_m_test)
        with open(join(logit_loc, 'dataset_webtables_{}_test'.format(round)), "wb") as fp:  # Pickling
            pickle.dump(total_test, fp)

    # print("The mean F1 score of train is:", np.mean(weighted_f1s_train))
    # print("The sd of train is:", np.std(weighted_f1s_train))
    # print("The mean macro F1 score of train is:", np.mean(macro_f1s_train))
    # print("The sd of train is:", np.std(macro_f1s_train))
    # print("===============================")

    print("The mean F1 score of test is:", np.mean(weighted_f1s_test))
    print("The sd of test is:", np.std(weighted_f1s_test))
    print("The mean macro F1 score of test is:", np.mean(macro_f1s_test))
    print("The sd of test is:", np.std(macro_f1s_test))
    print("===============================")

    print(macro_f1s_test)
    print(weighted_f1s_test)