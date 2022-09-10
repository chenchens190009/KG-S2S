import os
from tqdm import tqdm
import scipy.sparse as sp
from collections import defaultdict as ddict
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.rnn import pad_sequence
import pygtrie


def get_num(dataset_path, dataset, mode='entity'):  # mode: {entity, relation}
    return int(open(os.path.join(dataset_path, dataset, mode + '2id.txt')).readline().strip())


def read(configs, dataset_path, dataset, filename):
    file_name = os.path.join(dataset_path, dataset, filename)
    with open(file_name) as file:
        lines = file.read().strip().split('\n')
    n_triples = int(lines[0])
    triples = []
    for line in lines[1:]:
        split = line.split(' ')
        for i in range(3):
            split[i] = int(split[i])
        if configs.temporal:
            split[3] = split[3].replace('-', ' ')
        triples.append(split)
    assert n_triples == len(triples), 'number of triplets is not correct.'
    return triples


def read_file(configs, dataset_path, dataset, filename, mode='descrip'):
    id2name = []
    file_name = os.path.join(dataset_path, dataset, filename)
    with open(file_name, encoding='utf-8') as file:
        lines = file.read().strip('\n').split('\n')
    for i in range(1, len(lines)):
        ids, name = lines[i].split('\t')
        if mode == 'descrip':
            name = name.split(' ')
            name = ' '.join(name)
        id2name.append(name)
    return id2name


def read_name(configs, dataset_path, dataset):
    ent_name_file = 'entityid2name.txt'
    rel_name_file = 'relationid2name.txt'
    ent_name_list = read_file(configs, dataset_path, dataset, ent_name_file, 'name')
    rel_name_list = read_file(configs, dataset_path, dataset, rel_name_file, 'name')
    return ent_name_list, rel_name_list


def get_ground_truth(configs, triples):
    tail_ground_truth, head_ground_truth = ddict(list), ddict(list)
    for triple in triples:
        if configs.temporal:
            head, tail, rel, time = triple
            tail_ground_truth[(head, rel, time)].append(tail)
            head_ground_truth[(tail, rel, time)].append(head)
        else:
            head, tail, rel = triple
            tail_ground_truth[(head, rel)].append(tail)
            head_ground_truth[(tail, rel)].append(head)
    return tail_ground_truth, head_ground_truth


def get_next_token_dict(configs, ent_token_ids_in_trie, prefix_trie):
    neg_candidate_mask = []
    next_token_dict = {(): [32099] * configs.n_ent}
    for ent_id in tqdm(range(configs.n_ent)):
        rows, cols = [0], [32099]
        input_ids = ent_token_ids_in_trie[ent_id]
        for pos_id in range(1, len(input_ids)):
            cur_input_ids = input_ids[:pos_id]
            if tuple(cur_input_ids) in next_token_dict:
                cur_tokens = next_token_dict[tuple(cur_input_ids)]
            else:
                seqs = prefix_trie.keys(prefix=cur_input_ids)
                cur_tokens = [seq[pos_id] for seq in seqs]
                next_token_dict[tuple(cur_input_ids)] = Counter(cur_tokens)
            cur_tokens = list(set(cur_tokens))
            rows.extend([pos_id] * len(cur_tokens))
            cols.extend(cur_tokens)
        sparse_mask = sp.coo_matrix(([1] * len(rows), (rows, cols)), shape=(len(input_ids), configs.vocab_size), dtype=np.long)
        neg_candidate_mask.append(sparse_mask)
    return neg_candidate_mask, next_token_dict


def get_soft_prompt_pos(configs, source_ids, target_ids, mode):
    if configs.temporal:
        sep1, sep2, sep3 = [ids for ids in range(len(source_ids)) if source_ids[ids] == 1820]
        if mode == 'tail':
            input_index = [0] + list(range(0, sep1)) + [0] + [sep1] + [0] + list(range(sep1 + 1, sep2)) + [0] + list(range(sep2, len(source_ids)))
            soft_prompt_index = torch.LongTensor([0, sep1 + 1, sep1 + 3, sep2 + 3])
        elif mode == 'head':
            input_index = list(range(0, sep1 + 1)) + [0] + list(range(sep1 + 1, sep2)) + [0, sep2, 0] + list(range(sep2 + 1, sep3)) + [0] + list(range(sep3, len(source_ids)))
            soft_prompt_index = torch.LongTensor([sep2 + 3, sep3 + 3, sep1 + 1, sep2 + 1])
    else:
        sep1, sep2 = [ids for ids in range(len(source_ids)) if source_ids[ids] == 1820]
        if mode == 'tail':
            input_index = [0] + list(range(0, sep1)) + [0] + [sep1] + [0] + list(range(sep1 + 1, sep2)) + [0] + list(range(sep2, len(source_ids)))
            soft_prompt_index = torch.LongTensor([0, sep1 + 1, sep1 + 3, sep2 + 3])
        elif mode == 'head':
            input_index = list(range(0, sep1 + 1)) + [0] + list(range(sep1 + 1, sep2)) + [0, sep2, 0] + list(range(sep2 + 1, len(source_ids) - 1)) + [0] + [len(source_ids) - 1]
            soft_prompt_index = torch.LongTensor([sep2 + 3, len(source_ids) + 2, sep1 + 1, sep2 + 1])
    if target_ids is None:
        target_soft_prompt_index = None
    else:
        extra_token_01, extra_token_02 = target_ids.index(32099), target_ids.index(32098)
        target_soft_prompt_index = torch.LongTensor([extra_token_01, extra_token_02])
    return input_index, soft_prompt_index, target_soft_prompt_index


def construct_prefix_trie(ent_token_ids_in_trie):
    trie = pygtrie.Trie()
    for input_ids in ent_token_ids_in_trie:
        trie[input_ids] = True
    return trie


def batchify(output_dict, key, padding_value=None, return_list=False):
    tensor_out = [out[key] for out in output_dict]
    if return_list:
        return tensor_out
    if not isinstance(tensor_out[0], torch.LongTensor) and not isinstance(tensor_out[0], torch.FloatTensor):
        tensor_out = [torch.LongTensor(value) for value in tensor_out]
    if padding_value is None:
        tensor_out = torch.stack(tensor_out, dim=0)
    else:
        tensor_out = pad_sequence(tensor_out, batch_first=True, padding_value=padding_value)
    return tensor_out


def _get_performance(ranks, dataset):
    ranks = np.array(ranks, dtype=np.float)
    out = dict()
    out['mr'] = ranks.mean(axis=0)
    out['mrr'] = (1. / ranks).mean(axis=0)
    out['hit1'] = np.sum(ranks == 1, axis=0) / len(ranks)
    out['hit3'] = np.sum(ranks <= 3, axis=0) / len(ranks)
    out['hit10'] = np.sum(ranks <= 10, axis=0) / len(ranks)
    if dataset == 'NELL':
        out['hit5'] = np.sum(ranks <= 5, axis=0) / len(ranks)
    return out


def get_performance(model, tail_ranks, head_ranks):
    tail_out = _get_performance(tail_ranks, model.configs.dataset)
    head_out = _get_performance(head_ranks, model.configs.dataset)
    mr = np.array([tail_out['mr'], head_out['mr']])
    mrr = np.array([tail_out['mrr'], head_out['mrr']])
    hit1 = np.array([tail_out['hit1'], head_out['hit1']])
    hit3 = np.array([tail_out['hit3'], head_out['hit3']])
    hit10 = np.array([tail_out['hit10'], head_out['hit10']])

    if model.configs.dataset == 'NELL':
        val_mrr = tail_out['mrr'].item()
        model.log('val_mrr', val_mrr)
        hit5 = np.array([tail_out['hit5'], head_out['hit5']])
        perf = {'mrr': mrr, 'mr': mr, 'hit@1': hit1, 'hit@3': hit3, 'hit@5': hit5, 'hit@10': hit10}
    else:
        val_mrr = mrr.mean().item()
        model.log('val_mrr', val_mrr)
        perf = {'mrr': mrr, 'mr': mr, 'hit@1': hit1, 'hit@3': hit3, 'hit@10': hit10}
    perf = pd.DataFrame(perf, index=['tail ranking', 'head ranking'])
    perf.loc['mean ranking'] = perf.mean(axis=0)
    for hit in ['hit@1', 'hit@3', 'hit@5', 'hit@10']:
        if hit in list(perf.columns):
            perf[hit] = perf[hit].apply(lambda x: '%.2f%%' % (x * 100))
    return perf