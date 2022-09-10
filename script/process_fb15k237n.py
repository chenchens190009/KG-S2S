from base import Processor

in_folder = '../data/raw'
out_folder = '../data/processed'
in_dataset = 'FB15k-237'
out_dataset = 'FB15k-237N'


class FB15K237N_Processor(Processor):
    def __init__(self, in_folder, out_folder, in_dataset, out_dataset):
        super().__init__(in_folder, out_folder, out_dataset)
        self.in_dataset = in_dataset

    def create_ent2name(self, filename):
        lines = self.read_file(filename, dataset=in_dataset)
        for i, line in enumerate(lines):
            ent, name = line.split('\t')
            name = name.strip('\"').replace(r'\n', '').replace(r'\t', '').replace('\\', '')
            if ent not in self.ent2name:
                self.ent2name[ent] = name
            else:
                raise ValueError('%s dupliated entities!' % ent)

    def create_ent2descrip(self, filename):
        lines = self.read_file(filename, dataset=in_dataset)
        for i, line in enumerate(lines):
            ent, descrip = line.split('\t')
            if descrip.endswith('@en'):
                descrip = descrip[:-3]
            descrip = descrip.strip('\"').replace(r'\n', '').replace(r'\t', '').replace('\\', '')
            if ent not in self.ent2descrip:
                self.ent2descrip[ent] = descrip
            else:
                raise ValueError('%s dupliated entities!' % ent)

    def create_ent2id(self, filename):
        lines = self.read_file(filename, dataset=in_dataset)
        for i, line in enumerate(lines):
            ent = line.strip()
            self.ent2id[ent] = i
            self.entid2name[i] = self.ent2name[ent]
            if ent in self.ent2descrip:
                self.entid2descrip[i] = self.ent2descrip[ent]
            else:
                self.entid2descrip[i] = ''

    def create_rel2id(self, filename):
        lines = self.read_file(filename, dataset=in_dataset)
        for line in lines:
            rel = line.split('\t')[0]
            if '.' not in rel:
                self.rel2id[rel] = len(self.rel2id)
                self.relid2name[len(self.relid2name)] = rel

print('preprocessing FB15k-237N...')
processor = FB15K237N_Processor(in_folder, out_folder, in_dataset, out_dataset)
processor.create_out_folder()
processor.create_ent2name('entity2text.txt')
processor.create_ent2descrip('entity2textlong.txt')
processor.create_ent2id('entities.txt')
processor.create_rel2id('relation2text.txt')
processor.write_file('entity2id.txt', sort_key=lambda x: x[1])
processor.write_file('relation2id.txt', sort_key=lambda x: x[1])
processor.write_file('entityid2name.txt')
processor.write_file('relationid2name.txt', func=lambda x: x.strip('/').replace('/', ' , ').replace('_', ' '))
processor.write_file('entityid2description.txt')

in_files = ['train.tsv', 'dev.tsv', 'test.tsv']
out_files = ['train2id.txt', 'valid2id.txt', 'test2id.txt']
for i in range(3):
    in_file, out_file = in_files[i], out_files[i]
    triples = processor.read_triples(in_file, dataset=in_dataset)
    processor.write_triples(out_file, triples)

