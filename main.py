import os
import datetime
import argparse
from datetime import datetime
import warnings
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import T5Tokenizer
from transformers import T5Config
from models.model import T5Finetuner
from data import DataModule
from helper import get_num, read, read_name, read_file, get_ground_truth, get_next_token_dict, construct_prefix_trie
from callbacks import PrintingCallback


def main():
    ## read triples
    train_triples = read(configs, configs.dataset_path, configs.dataset, 'train2id.txt')
    valid_triples = read(configs, configs.dataset_path, configs.dataset, 'valid2id.txt')
    test_triples = read(configs, configs.dataset_path, configs.dataset, 'test2id.txt')
    all_triples = train_triples + valid_triples + test_triples

    ## construct name list
    original_ent_name_list, rel_name_list = read_name(configs, configs.dataset_path, configs.dataset)
    tokenizer = T5Tokenizer.from_pretrained(configs.pretrained_model)
    description_list = read_file(configs, configs.dataset_path, configs.dataset, 'entityid2description.txt', 'descrip')
    print('tokenizing entities...')
    src_description_list = tokenizer.batch_decode([descrip[:-1] for descrip in tokenizer(description_list, max_length=configs.src_descrip_max_length, truncation=True).input_ids])
    tgt_description_list = tokenizer.batch_decode([descrip[:-1] for descrip in tokenizer(description_list, max_length=configs.tgt_descrip_max_length, truncation=True).input_ids])

    ## construct prefix trie
    # ent_token_ids_in_trie .type: list(list(ids))
    ent_token_ids_in_trie = tokenizer(['<extra_id_0>' + ent_name + '<extra_id_1>' for ent_name in original_ent_name_list], max_length=configs.train_tgt_max_length, truncation=True).input_ids

    if configs.tgt_descrip_max_length > 0:
        ent_token_ids_in_trie_with_descrip = tokenizer(['<extra_id_0>' + ent_name + '[' + tgt_description_list[i] + ']' + '<extra_id_1>' for i, ent_name in enumerate(original_ent_name_list)], max_length=configs.train_tgt_max_length, truncation=True).input_ids
        prefix_trie = construct_prefix_trie(ent_token_ids_in_trie_with_descrip)
        neg_candidate_mask, next_token_dict = get_next_token_dict(configs, ent_token_ids_in_trie_with_descrip, prefix_trie)
    else:
        prefix_trie = construct_prefix_trie(ent_token_ids_in_trie)
        neg_candidate_mask, next_token_dict = get_next_token_dict(configs, ent_token_ids_in_trie, prefix_trie)
    ent_name_list = tokenizer.batch_decode([tokens[1:-2] for tokens in ent_token_ids_in_trie])
    name_list_dict = {
        'original_ent_name_list': original_ent_name_list,
        'ent_name_list': ent_name_list,
        'rel_name_list': rel_name_list,
        'src_description_list': src_description_list,
        'tgt_description_list': tgt_description_list
    }

    prefix_trie_dict = {
        'prefix_trie': prefix_trie,
        'ent_token_ids_in_trie': ent_token_ids_in_trie,
        'neg_candidate_mask': neg_candidate_mask,
        'next_token_dict': next_token_dict
    }
    if configs.tgt_descrip_max_length > 0:
        prefix_trie_dict['ent_token_ids_in_trie_with_descrip'] = ent_token_ids_in_trie_with_descrip

    ## construct ground truth dictionary
    # ground truth .shape: dict, example: {hr_str_key1: [t_id11, t_id12, ...], (hr_str_key2: [t_id21, t_id22, ...], ...}
    train_tail_ground_truth, train_head_ground_truth = get_ground_truth(configs, train_triples)
    all_tail_ground_truth, all_head_ground_truth = get_ground_truth(configs, all_triples)

    ground_truth_dict = {
        'train_tail_ground_truth': train_tail_ground_truth,
        'train_head_ground_truth': train_head_ground_truth,
        'all_tail_ground_truth': all_tail_ground_truth,
        'all_head_ground_truth': all_head_ground_truth,
    }

    datamodule = DataModule(configs, train_triples, valid_triples, test_triples, name_list_dict, prefix_trie_dict, ground_truth_dict)
    print('datamodule construction done.', flush=True)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_mrr',
        dirpath=configs.save_dir,
        filename=configs.dataset + '-{epoch:03d}-{' + 'val_mrr' + ':.4f}',
        mode='max'
    )
    printing_callback = PrintingCallback()

    gpu = [int(configs.gpu)] if torch.cuda.is_available() else 0
    trainer_params = {
        'gpus': gpu,
        'max_epochs': configs.epochs,  # 1000
        'checkpoint_callback': True,  # True
        'logger': False,  # TensorBoardLogger
        'num_sanity_val_steps': 0,  # 2
        'check_val_every_n_epoch': 3,
        'enable_progress_bar': True,
        'callbacks': [
            checkpoint_callback,
            printing_callback
        ],
    }
    trainer = pl.Trainer(**trainer_params)
    kw_args = {
        'ground_truth_dict': ground_truth_dict,
        'name_list_dict': name_list_dict,
        'prefix_trie_dict': prefix_trie_dict
    }

    if configs.model_path == '':
        model = T5Finetuner(configs, **kw_args)
        print('model construction done.', flush=True)
        trainer.fit(model, datamodule)
        model_path = checkpoint_callback.best_model_path
    else:
        model_path = configs.model_path
    print('model_path:', model_path, flush=True)
    model = T5Finetuner.load_from_checkpoint(model_path, strict=False, configs=configs, **kw_args)
    trainer.test(model, dataloaders=datamodule)


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset_path', type=str, default='./data/processed')
    parser.add_argument('-dataset', dest='dataset', default='WN18RR', help='Dataset to use, default: WN18RR')
    parser.add_argument('-model', default='T5Finetuner', help='Model Name')
    parser.add_argument('-gpu', type=str, default='0', help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('-seed', dest='seed', default=41504, type=int, help='Seed for randomization')
    parser.add_argument('-num_workers', type=int, default=4, help='Number of processes to construct batches')
    parser.add_argument('-save_dir', type=str, default='', help='')

    parser.add_argument('-pretrained_model', type=str, default='t5-base', help='')
    parser.add_argument('-batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('-val_batch_size', default=8, type=int, help='Batch size')
    parser.add_argument('-num_beams', default=40, type=int, help='Number of samples from beam search')
    parser.add_argument('-num_beam_groups', default=1, type=int, help='')
    parser.add_argument('-src_max_length', default=512, type=int, help='')
    parser.add_argument('-train_tgt_max_length', default=512, type=int, help='')
    parser.add_argument('-eval_tgt_max_length', default=30, type=int, help='')
    parser.add_argument('-epoch', dest='epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('-lr', type=float, default=0.001, help='Starting Learning Rate')
    parser.add_argument('-diversity_penalty', default=0., type=float, help='')

    parser.add_argument('-model_path', dest='model_path', default='', help='The path for reloading models')
    parser.add_argument('-optim', default='Adam', type=str, help='')
    parser.add_argument('-decoder', type=str, default='beam_search', help='[beam_search, do_sample, beam_sample_search, diverse_beam_search]')
    parser.add_argument('-log_text', action='store_true', help='')
    parser.add_argument('-use_prefix_search', action='store_true', help='')
    parser.add_argument('-src_descrip_max_length', default=0, type=int, help='')
    parser.add_argument('-tgt_descrip_max_length', default=0, type=int, help='')
    parser.add_argument('-use_soft_prompt', action='store_true', help='')
    parser.add_argument('-use_rel_prompt_emb', action='store_true', help='')
    parser.add_argument('-skip_n_val_epoch', default=0, type=int, help='')
    parser.add_argument('-seq_dropout', default=0., type=float, help='')
    parser.add_argument('-temporal', action='store_true', help='')
    configs = parser.parse_args()
    n_ent = get_num(configs.dataset_path, configs.dataset, 'entity')
    n_rel = get_num(configs.dataset_path, configs.dataset, 'relation')
    configs.n_ent = n_ent
    configs.n_rel = n_rel
    configs.vocab_size = T5Config.from_pretrained(configs.pretrained_model).vocab_size
    configs.model_dim = T5Config.from_pretrained(configs.pretrained_model).d_model
    if configs.save_dir == '':
        configs.save_dir = os.path.join('./checkpoint', configs.dataset + '-' + str(datetime.now()))
    os.makedirs(configs.save_dir, exist_ok=True)
    print(configs, flush=True)

    pl.seed_everything(configs.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_printoptions(profile='full')
    main()
