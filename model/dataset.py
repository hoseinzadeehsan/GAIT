
from os.path import join
import pickle
import torch
from torch.utils.data import Dataset
import configs
import dgl
import networkx as nx

class TableDatasetMulti(Dataset):

    def __init__(self, args, data_name, data_type='test', ratio=1):
        self.args = args
        self.data_name = data_name
        self.data_dir = join(configs.root_dir, 'data')
        self.sample_size = 0

        with open(join(self.data_dir, self.data_name), "rb") as f:
            self.samples = pickle.load(f)
        if data_type == 'train' and data_name.rsplit('_', 2)[0] != 'dataset_semtab':# and data_name.rsplit('_', 3)[0] != 'dataset_semtab' :
            self.samples = self.samples[:int(ratio*self.samples.shape[0])]
        elif data_type == 'validation' and data_name.rsplit('_', 2)[0] != 'dataset_semtab': #and data_name.rsplit('_', 3)[0] != 'dataset_semtab':
            print('using ratio for validation')
            self.samples = self.samples[int(ratio * self.samples.shape[0]):]

        self.sample_size = len(self.samples)
        self.feature_size = self.samples[0]['features'].shape[1]

        print('Sample size', self.sample_size)

        # print(self.samples)
        # print(type(self.samples[0]))

    def __len__(self):
        return self.sample_size

    def __getitem__(self, idx):
        # g is a complete graph with self.number_of_nodes
        self.number_of_nodes = self.samples[idx]['masks'].sum()
        g = nx.complete_graph(self.number_of_nodes)
        g = dgl.from_networkx(g)

        # removing and adding self loop
        g = dgl.remove_self_loop(g)
        if self.args.self_loop:
            g= dgl.add_self_loop(g)

        g.ndata['feat'] = torch.tensor(self.samples[idx]['features'][self.samples[idx]['masks'] == 1], dtype=torch.float)
        g.ndata['label'] = torch.tensor(self.samples[idx]['labels'][self.samples[idx]['masks'] == 1]).view(-1, 1)

        return g












