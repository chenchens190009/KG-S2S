import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer
from helper import batchify, get_soft_prompt_pos


class TrainDataset(Dataset):
    def __init__(self, configs, tokenizer, train_triples, name_list_dict, prefix_trie_dict, ground_truth_dict):
        self.configs = configs
        self.train_triples = train_triples
        self.tokenizer = tokenizer
        self.original_ent_name_list = name_list_dict['original_ent_name_list']
        self.ent_name_list = name_list_dict['ent_name_list']
        self.rel_name_list = name_list_dict['rel_name_list']
        self.src_description_list = name_list_dict['src_description_list']
        self.tgt_description_list = name_list_dict['tgt_description_list']
        self.ent_token_ids_in_trie = prefix_trie_dict['ent_token_ids_in_trie']
        self.neg_candidate_mask = prefix_trie_dict['neg_candidate_mask']

    def __len__(self):
        return len(self.train_triples) * 2

    def __getitem__(self, index):
        train_triple = self.train_triples[index // 2]
        mode = 'tail' if index % 2 == 0 else 'head'
        if self.configs.temporal:
            head, tail, rel, time = train_triple
        else:
            head, tail, rel = train_triple
        head_name, tail_name, rel_name = self.original_ent_name_list[head], self.original_ent_name_list[tail], self.rel_name_list[rel]
        if self.configs.src_descrip_max_length > 0:
            head_descrip, tail_descrip = '[' + self.src_description_list[head] + ']', '[' + self.src_description_list[tail] + ']'
        else:
            head_descrip, tail_descrip = '', ''
        if self.configs.tgt_descrip_max_length > 0:
            head_target_descrip, tail_target_descrip = '[' + self.tgt_description_list[head] + ']', '[' + self.tgt_description_list[tail] + ']'
        else:
            head_target_descrip, tail_target_descrip = '', ''

        if mode == 'tail':
            if self.configs.temporal:
                src = head_name + ' ' + head_descrip + ' | ' + rel_name + ' | ' + '<extra_id_0>' + ' | ' + time
            else:
                src = head_name + ' ' + head_descrip + ' | ' + rel_name + ' | ' + '<extra_id_0>'
            tgt = '<extra_id_0>' + tail_name + tail_target_descrip + '<extra_id_1>'
        else:
            if self.configs.temporal:
                src = '<extra_id_0>' + ' | ' + rel_name + ' | ' + tail_name + ' ' + tail_descrip + ' | ' + time
            else:
                src = '<extra_id_0>' + ' | ' + rel_name + ' | ' + tail_name + ' ' + tail_descrip
            tgt = '<extra_id_0>' + head_name + head_target_descrip + '<extra_id_1>'

        tokenized_src = self.tokenizer(src, max_length=self.configs.src_max_length, truncation=True)
        source_ids = tokenized_src.input_ids
        source_mask = tokenized_src.attention_mask
        tokenized_tgt = self.tokenizer(tgt, max_length=self.configs.train_tgt_max_length, truncation=True)
        target_ids = tokenized_tgt.input_ids
        target_mask = tokenized_tgt.attention_mask

        ent_rel = torch.LongTensor([head, rel]) if mode == 'tail' else torch.LongTensor([tail, rel])
        out = {
                'source_ids': source_ids,
                'source_mask': source_mask,
                'target_ids': target_ids,
                'target_mask': target_mask,
                'train_triple': train_triple,
                'ent_rel': ent_rel,
        }

        if self.configs.use_soft_prompt:
            input_index, soft_prompt_index, target_soft_prompt_index = get_soft_prompt_pos(self.configs, source_ids, target_ids, mode)
            out['input_index'] = input_index
            out['soft_prompt_index'] = soft_prompt_index
            out['target_soft_prompt_index'] = target_soft_prompt_index
        return out

    def collate_fn(self, data):
        agg_data = dict()
        agg_data['source_ids'] = batchify(data, 'source_ids', padding_value=0)
        agg_data['source_mask'] = batchify(data, 'source_mask', padding_value=0)
        agg_data['target_ids'] = batchify(data, 'target_ids', padding_value=0)
        agg_data['target_mask'] = batchify(data, 'target_mask', padding_value=0)
        agg_data['train_triple'] = batchify(data, 'train_triple', return_list=True)
        agg_data['ent_rel'] = batchify(data, 'ent_rel')
        if self.configs.use_soft_prompt:
            agg_data['input_index'] = batchify(data, 'input_index', padding_value=0)
            agg_data['soft_prompt_index'] = batchify(data, 'soft_prompt_index')
            agg_data['target_soft_prompt_index'] = batchify(data, 'target_soft_prompt_index')
        return agg_data


class TestDataset(Dataset):
    def __init__(self, configs, tokenizer, test_triples, name_list_dict, prefix_trie_dict, ground_truth_dict, mode):  # mode: {tail, head}
        self.configs = configs
        self.test_triples = test_triples
        self.original_ent_name_list = name_list_dict['original_ent_name_list']
        self.ent_name_list = name_list_dict['ent_name_list']
        self.rel_name_list = name_list_dict['rel_name_list']
        self.src_description_list = name_list_dict['src_description_list']
        self.tgt_description_list = name_list_dict['tgt_description_list']
        self.ent_token_ids_in_trie = prefix_trie_dict['ent_token_ids_in_trie']
        self.tokenizer = tokenizer
        self.mode = mode

    def __len__(self):
        return len(self.test_triples)

    def __getitem__(self, index):
        test_triple = self.test_triples[index]
        if self.configs.temporal:
            head, tail, rel, time = test_triple
        else:
            head, tail, rel = test_triple
        head_name, tail_name, rel_name = self.original_ent_name_list[head], self.original_ent_name_list[tail], self.rel_name_list[rel]

        if self.configs.src_descrip_max_length > 0:
            head_descrip, tail_descrip = '[' + self.src_description_list[head] + ']', '[' + self.src_description_list[tail] + ']'
        else:
            head_descrip, tail_descrip = '', ''

        if self.mode == 'tail':
            if self.configs.temporal:
                src = head_name + ' ' + head_descrip + ' | ' + rel_name + ' | ' + '<extra_id_0>' + ' | ' + time
            else:
                src = head_name + ' ' + head_descrip + ' | ' + rel_name + ' | ' + '<extra_id_0>'
            tgt_ids = tail
        else:
            if self.configs.temporal:
                src = '<extra_id_0>' + ' | ' + rel_name + ' | ' + tail_name + ' ' + tail_descrip + ' | ' + time
            else:
                src = '<extra_id_0>' + ' | ' + rel_name + ' | ' + tail_name + ' ' + tail_descrip
            tgt_ids = head

        tokenized_src = self.tokenizer(src, max_length=self.configs.src_max_length, truncation=True)
        source_ids = tokenized_src.input_ids
        source_mask = tokenized_src.attention_mask
        source_names = src
        target_names = self.ent_name_list[tgt_ids]

        # ent_rel = test_triple[[0, 2]] if self.mode == 'tail' else test_triple[[1, 2]]
        ent_rel = torch.LongTensor([head, rel]) if self.mode == 'tail' else torch.LongTensor([tail, rel])
        out = {
            'source_ids': source_ids,
            'source_mask': source_mask,
            'source_names': source_names,
            'target_names': target_names,
            'test_triple': test_triple,
            'ent_rel': ent_rel
        }
        if self.configs.use_soft_prompt:
            input_index, soft_prompt_index, _ = get_soft_prompt_pos(self.configs, source_ids, None, self.mode)
            out['input_index'] = input_index
            out['soft_prompt_index'] = soft_prompt_index
        return out

    def collate_fn(self, data):
        agg_data = dict()
        agg_data['source_ids'] = batchify(data, 'source_ids', padding_value=0)
        agg_data['source_mask'] = batchify(data, 'source_mask', padding_value=0)
        agg_data['source_names'] = [dt['source_names'] for dt in data]
        agg_data['target_names'] = [dt['target_names'] for dt in data]
        agg_data['test_triple'] = batchify(data, 'test_triple', return_list=True)
        agg_data['ent_rel'] = batchify(data, 'ent_rel')
        if self.configs.use_soft_prompt:
            agg_data['input_index'] = batchify(data, 'input_index', padding_value=0)
            agg_data['soft_prompt_index'] = batchify(data, 'soft_prompt_index')
        return agg_data


class DataModule(pl.LightningDataModule):
    def __init__(self, configs, train_triples, valid_triples, test_triples, name_list_dict, prefix_trie_dict, ground_truth_dict):
        super().__init__()
        self.configs = configs
        self.train_triples = train_triples
        self.valid_triples = valid_triples
        self.test_triples = test_triples
        # ent_name_list, rel_name_list .type: list
        self.name_list_dict = name_list_dict
        self.prefix_trie_dict = prefix_trie_dict
        self.ground_truth_dict = ground_truth_dict

        self.tokenizer = T5Tokenizer.from_pretrained(configs.pretrained_model)
        self.train_both = None
        self.valid_tail, self.valid_head = None, None
        self.test_tail, self.test_head = None, None

    def prepare_data(self):
        self.train_both = TrainDataset(self.configs, self.tokenizer, self.train_triples, self.name_list_dict, self.prefix_trie_dict, self.ground_truth_dict)
        self.valid_tail = TestDataset(self.configs, self.tokenizer, self.valid_triples, self.name_list_dict, self.prefix_trie_dict, self.ground_truth_dict, 'tail')
        self.valid_head = TestDataset(self.configs, self.tokenizer, self.valid_triples, self.name_list_dict, self.prefix_trie_dict, self.ground_truth_dict, 'head')
        self.test_tail = TestDataset(self.configs, self.tokenizer, self.test_triples, self.name_list_dict, self.prefix_trie_dict, self.ground_truth_dict, 'tail')
        self.test_head = TestDataset(self.configs, self.tokenizer, self.test_triples, self.name_list_dict, self.prefix_trie_dict, self.ground_truth_dict, 'head')

    def train_dataloader(self):
        train_loader = DataLoader(self.train_both,
                                  batch_size=self.configs.batch_size,
                                  shuffle=True,
                                  collate_fn=self.train_both.collate_fn,
                                  pin_memory=True,
                                  num_workers=self.configs.num_workers)
        return train_loader

    def val_dataloader(self):
        valid_tail_loader = DataLoader(self.valid_tail,
                                       batch_size=self.configs.val_batch_size,
                                       shuffle=False,
                                       collate_fn=self.valid_tail.collate_fn,
                                       pin_memory=True,
                                       num_workers=self.configs.num_workers)
        valid_head_loader = DataLoader(self.valid_head,
                                       batch_size=self.configs.val_batch_size,
                                       shuffle=False,
                                       collate_fn=self.valid_head.collate_fn,
                                       pin_memory=True,
                                       num_workers=self.configs.num_workers)
        return [valid_tail_loader, valid_head_loader]

    def test_dataloader(self):
        test_tail_loader = DataLoader(self.test_tail,
                                      batch_size=self.configs.val_batch_size,
                                      shuffle=False,
                                      collate_fn=self.test_tail.collate_fn,
                                      pin_memory=True,
                                      num_workers=self.configs.num_workers)
        test_head_loader = DataLoader(self.test_head,
                                      batch_size=self.configs.val_batch_size,
                                      shuffle=False,
                                      collate_fn=self.test_head.collate_fn,
                                      pin_memory=True,
                                      num_workers=self.configs.num_workers)
        return [test_tail_loader, test_head_loader]
