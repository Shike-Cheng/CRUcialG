import os
import re
import pickle
import torch
import numpy as np
import configparser

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import ARGVA
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics.pairwise import cosine_similarity
import argparse


D = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eventMapDict = {'cru_vs_gt': {'RD': 1, 'WR': 2, 'EX': 3, 'UK': 4, 'CD': 5, 'FR': 6, 'IJ': 7, 'ST': 8, 'RF': 9}, 'cru_vs_extractor': {'RD': 1, 'WR': 2, 'EX': 3, 'UK': 4, 'CD': 5, 'FR': 6, 'IJ': 7, 'ST': 8, 'RF': 9}, 'cru_vs_attackg': {'edge': 1}}
entityMapDict = {'cru_vs_gt': {"MP": 0, "TP": 1, "MF": 2, "SF": 3, "TF": 4, "SO": 5}, 'cru_vs_extractor': {0: "P", 1: "F", 2: "S"}, 'cru_vs_attackg': {0: "P", 1: "F", 2: "S"}}
types_map = {'cru_vs_gt': ['test_cru', 'test_cru_gt'], 'cru_vs_extractor': ['test_cru', 'test_extractor', 'test_cru_gt'], 'cru_vs_attackg': ['test_cru', 'test_attackg', 'test_cru_gt']}

def build_mapping():
    nodeItems = entityMapDict.values()
    unique_nodeItems = list(set(nodeItems))
    item_to_idxNode = {item: idx for idx, item in enumerate(unique_nodeItems)}

    edgeItems = eventMapDict.values()
    unique_edgeItems = list(set(edgeItems))
    item_to_idxEdge = {item: idx for idx, item in enumerate(unique_edgeItems)}

    return item_to_idxNode, item_to_idxEdge


def graph_to_data(graph, node_mapping, edge_mapping):
    node_idx_mapping = {}
    node_attrs = []
    edge_index = []
    edge_attr = []

    for idx, (node, attrs) in enumerate(graph.nodes(data=True)):
        node_attrs.append(node_mapping[str(attrs['type'])])
        node_idx_mapping[node] = idx
    num_nodes = len(node_attrs)

    for source, target, type, _ in graph.edges(data=True, keys=True):
        src_idx = node_idx_mapping[source]
        tgt_idx = node_idx_mapping[target]
        edge_index.append([src_idx, tgt_idx])
        edge_index.append([tgt_idx, src_idx])
        edge_attr.append(edge_mapping[type])

    x = torch.tensor(node_attrs, dtype=torch.long)
    one_hot_matrix = torch.eye(len(node_mapping))
    x = one_hot_matrix[x]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32).unsqueeze(1)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)

    return data


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class Discriminator(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = self.lin1(x).relu()
        x = self.lin2(x).relu()
        return self.lin3(x)


def test(x, edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, edge_index)
    return z  # model.test(z, pos_edge_index)


def make_data(data_path):

    graphs = []
    with open(data_path, 'rb') as f:
        snapshotSeq = pickle.load(f)
    for _, snapshot in snapshotSeq.items():
        graphs.append(snapshot)

    data_list = []
    for graph in graphs:
        data = graph_to_data(graph, node_mapping, edge_mapping)
        data_list.append(data)

    dataloader = DataLoader(data_list, batch_size=batch_size, shuffle=True)

    return dataloader, data_list


def train(dataloader, d_l):
    hidden_dim = 2 * D
    output_dim = D
    epochs = 1000
    patience = 2000
    encoder = Encoder(num_node_features, hidden_channels=hidden_dim, out_channels=output_dim)
    discriminator = Discriminator(in_channels=output_dim, hidden_channels=hidden_dim,
                                  out_channels=output_dim)
    model = ARGVA(encoder, discriminator).to(device)

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.00001)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(),
                                               lr=0.00001)
    scheduler = ReduceLROnPlateau(encoder_optimizer, mode='min', factor=0.1, patience=patience)
    model = model.to(device)

    flag = False

    total_batch = 0
    total_batch_loss = 0
    last_improve = 0
    best_loss = float('inf')
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for datas in dataloader:
            x = datas.x.to(device)
            edge_index = datas.edge_index.to(device)
            num_nodes = x.shape[0]
            model.train()
            encoder_optimizer.zero_grad()
            z = model.encode(x, edge_index)
            for i in range(5): 
                discriminator_optimizer.zero_grad()
                discriminator_loss = model.discriminator_loss(z)
                discriminator_loss.backward()
                discriminator_optimizer.step()
            loss = model.recon_loss(z, edge_index) 
            loss = loss + model.reg_loss(z)
            loss = loss + (1 / num_nodes) * model.kl_loss() 
            loss.backward()
            encoder_optimizer.step()
            if total_batch % 100 == 0 and total_batch > 0:
                avg_batch_loss = total_batch_loss / (batch_size * 100)
                print(f"Iter {total_batch}, Loss: {avg_batch_loss}")
                if avg_batch_loss < best_loss:
                    best_loss = avg_batch_loss
                    # create_folder_for_file(modelPath)
                    torch.save(model, modelPath)
                    last_improve = total_batch
                total_batch_loss = 0
            if total_batch - last_improve > patience:

                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
            total_batch_loss += float(loss)
            total_loss += float(loss)
            total_batch += 1
        if flag:
            break
        avg_loss = total_loss / len(d_l)
        print(f"Epoch {epoch}, Loss: {avg_loss}")

    torch.save(model, modelPath)

def make_embedding(type_data):
    embeddings = None
    for data in datalist:
        x = data.x.to(device)
        g_edge_index = data.edge_index.to(device)
        z = test(x, g_edge_index).unsqueeze(0).cpu()
        z = torch.mean(z, 1)
        if embeddings is None:
            embeddings = z
        else:
            embeddings = torch.cat((embeddings, z), 0)

    embeddingFilePath = "evaluate/graph_similartiy/" + args.scene + '/' + t + '_embedding.pkl'
    with open(embeddingFilePath, 'wb') as fs:
        pickle.dump(embeddings, fs)

def calculate(embedding1_path, embedding2_path):
    similarity_dict = []

    sum_similarity = 0
    with open(embedding1_path, 'rb') as f:
        snapshotSeq1 = pickle.load(f)
    with open(embedding2_path, 'rb') as f:
        snapshotSeq2 = pickle.load(f)
    for i in range(len(snapshotSeq1)):

        embeddings1 = snapshotSeq1[i].reshape(1, -1)
        embeddings2 = snapshotSeq2[i].reshape(1, -1)
        similarity = cosine_similarity(embeddings1, embeddings2)
        cosine_similarity_score = similarity[0][0]

        cosine_similarity_score = float(cosine_similarity_score)
        sum_similarity += cosine_similarity_score
        similarity_dict.append(cosine_similarity_score)

    print(len(snapshotSeq1))

    return sum_similarity/len(snapshotSeq1), similarity_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true', help="whether to run training")
    # parser.add_argument('--data_dir', type=str, default=None, required=True, help="path to the training dataset")
    parser.add_argument('--scene', type=str, default='cru_vs_gt')
    args = parser.parse_args()
    # assert args.scene in ['cru_vs_gt', 'cru_vs_extractor', 'cru_vs_attackg']
    batch_size = 4
    edge_mapping = eventMapDict[args.scene]
    node_mapping = entityMapDict[args.scene]

    num_node_features = len(node_mapping)
    num_edge_features = len(edge_mapping)
    modelPath = 'evaluate/graph_similartiy/' + args.scene + 'autoencoder.pth'

    if args.do_train:
        path = 'evaluate/graph_similartiy/train_data/' + args.scene + '.pkl'
        data_loader, datalist = make_data(path)
        train(data_loader, datalist)
    else:
        model = torch.load(modelPath)
        for t in types_map[args.scene]:
            path = "evaluate/graph_similartiy/" + args.scene + '/' + t + '.pkl'
            data_loader, datalist = make_data(path)
            make_embedding(t)
        if args.scene == 'cru_vs_gt':
            ASG_embedding = "evaluate/graph_similartiy/" + args.scene + '/' + types_map[args.scene][0] + '_embedding.pkl'
            GT_embedding = "evaluate/graph_similartiy/" + args.scene + '/' + types_map[args.scene][1] + '_embedding.pkl'
            avg_sim, sim_list = calculate(ASG_embedding, GT_embedding)

        else:
            ASG_embedding = "evaluate/graph_similartiy/" + args.scene + '/' + types_map[args.scene][0] + '_embedding.pkl'
            GT_embedding = "evaluate/graph_similartiy/" + args.scene + '/' + types_map[args.scene][1] + '_embedding.pkl'
            SOTA__embedding = "evaluate/graph_similartiy/" + args.scene + '/' + types_map[args.scene][2] + '_embedding.pkl'
            avg_sim, sim_list = calculate(ASG_embedding, GT_embedding)
            avg_sim1, sim_list1 = calculate(SOTA__embedding, GT_embedding)



