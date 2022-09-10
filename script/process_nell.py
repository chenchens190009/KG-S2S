import json
import os
from base import Processor

in_folder = '../data/raw'
out_folder = '../data/processed'
dataset = 'NELL'

non_capitalized_list = ['a', 'an', 'the', 'for', 'and', 'nor', 'but', 'or', 'yet', 'so', 'at', 'around', 'by', 'after', 'along', 'from', 'of', 'on', 'to', 'with', 'without']


class NELL_Processor(Processor):
    def __init__(self, in_folder, out_folder, dataset):
        super().__init__(in_folder, out_folder, dataset)

    def create_ent2id_rel2id(self):
        train_lines = self.read_file('train.tsv')
        valid_lines = self.read_file('dev.tsv')
        test_lines = self.read_file('test.tsv')
        lines = train_lines + valid_lines + test_lines
        for line in lines:
            h, r, t = line.split('\t')
            h, r, t = h.strip(), r.strip(), t.strip()
            if h not in self.ent2id:
                self.ent2id[h] = len(self.ent2id)
                self.id2ent[len(self.ent2id)] = h
            if t not in self.ent2id:
                self.ent2id[t] = len(self.ent2id)
                self.id2ent[len(self.ent2id)] = t
            if r not in self.rel2id:
                self.rel2id[r] = len(self.rel2id)
                self.id2rel[len(self.rel2id)] = r

    def create_entid2name(self, filename):
        lines = self.read_file(filename)
        for line in lines:
            ent, entname = line.split('\t')
            ent, entname = ent.strip(), entname.strip()
            entname = self.process_name(entname)
            self.ent2name[ent] = entname
            self.entid2name[self.ent2id[ent]] = entname

    def process_name(self, name):
        s = name.split(':')
        if len(s) == 1:
            return name
        elif len(s) == 2:
            enttype, entname = s
            segments = entname.strip().split(' ')
            entname = ' '.join(list(map(self.mapper, segments)))
            return enttype + ': ' + entname
        else:
            raise ValueError('Invalid pattern %s' % name)

    @staticmethod
    def mapper(word):
        if word in non_capitalized_list:
            return word
        capitalized_word = word.capitalize()
        return capitalized_word
        # if capitalized_word in words.words():
        #     return capitalized_word
        # else:
        #     if word in words.words():
        #         return word
        #     else:
        #         return capitalized_word

    def create_entid2descrip(self):
        for entid, _ in self.entid2name.items():
            self.entid2descrip[entid] = ' '

    def create_rel2name(self, filename):
        lines = self.read_file(filename)
        for line in lines:
            rel, relname = line.split('\t')
            rel, relname = rel.strip(), relname.strip()
            self.relid2name[self.rel2id[rel]] = relname

    def create_typecons(self, filename):
        lines = self.read_json_file(filename)
        self.typecons = {}
        for rel, value in lines.items():
            if rel in self.rel2id:
                relid = self.rel2id[rel]
                heads, tails = value['head'], value['tail']
                heads = [self.ent2id[head] for head in heads]
                tails = [self.ent2id[tail] for tail in tails]
                self.typecons[relid] = {'head': heads, 'tail': tails}

    def write_typecons(self, filename):
        with open(os.path.join(self.out_folder, self.dataset, filename), 'w') as file:
            json.dump(self.typecons, file)


print('preprocessing NELL-One...')
processor = NELL_Processor(in_folder, out_folder, dataset)
processor.create_out_folder()
processor.create_ent2id_rel2id()
processor.create_entid2name('entity2text.txt')
processor.create_entid2descrip()
processor.create_rel2name('relation2text.txt')
processor.write_file('entity2id.txt', sort_key=lambda x: x[1])
processor.write_file('relation2id.txt', sort_key=lambda x: x[1])
processor.write_file('entityid2name.txt')
processor.write_file('relationid2name.txt')
processor.write_file('entityid2description.txt')
processor.create_typecons('typecons.json')
processor.write_typecons('typecons.json')

in_files = ['train.tsv', 'dev.tsv', 'test.tsv']
out_files = ['train2id.txt', 'valid2id.txt', 'test2id.txt']
for i in range(3):
    in_file, out_file = in_files[i], out_files[i]
    triples = processor.read_triples(in_file)
    processor.write_triples(out_file, triples)

