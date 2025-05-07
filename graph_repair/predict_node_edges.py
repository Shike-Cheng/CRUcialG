import time
import argparse
import numpy as np
import math
import os
import sys
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils import *

from model_CTI_predict import GraphFlowModel
from gran_stage_data import *
from dataloader import PretrainZinkDataset
import environment as env
import warnings
from logmaking import make_print_to_file


def read_molecules(path):
    print('reading data from %s' % path)
    node_features = np.load(path + '_np_node.npy')   # 节点类型
    adj_features = np.load(path + '_np_adj.npy')     # 大的邻接矩阵，99，9，100，100
    mol_sizes = np.load(path + '_np_mol.npy')        # 节点类型 99，100

    f = open(path + '_config.txt', 'r')
    data_config = eval(f.read())
    f.close()

    fp = open(path + '_graph.txt')
    all_smiles = []
    for smiles in fp:
        all_smiles.append(smiles.strip())
    fp.close()

    return node_features, adj_features, mol_sizes, data_config, all_smiles

def read(json_file):
    data_ = open(json_file, "r", encoding="utf-8")
    docs = json.load(data_)
    return docs

def swap_keys_values(input_dict):
    swapped_dict = {v: k for k, v in input_dict.items()}
    return swapped_dict

def get_miss_stage(stages_index, edges):
    miss_id = []
    stage_id = 0
    for s in stages_index:
        if len(s) == 0:
            miss_id.append(stage_id)
        if len(s) > 1:
            if abs(s[0] - s[-1]) >= len(edges) / 2:
                miss_id.append(stage_id)
        stage_id += 1

    return miss_id


def get_up_to_index1(miss_stage, stages_index, nodes, edges):
    # result_node_index = []
    # result_edge_index = []
    if miss_stage in [[0], [0, 1], [0, 1, 2], [2]]:
        result_edge_index = stages_index[miss_stage[-1] + 1][0] - 1
        stage_edge = edges[:stages_index[miss_stage[-1] + 1][0]]
        current_nodes = set()
        for e in stage_edge:
            current_nodes.add(e[0])
            current_nodes.add(e[1])
        if len(current_nodes) == 0:
            result_node_index = 0
        elif max(current_nodes) > 100:
            result_node_index = 0
            result_edge_index = 0
        else:
            result_node_index = max(current_nodes)

    elif miss_stage in [[0, 1, 2, 3], [1, 2, 3], [2, 3], [3]]:
        if len(nodes) >= 80:
            result_edge_index = stages_index[miss_stage[0] - 1][-1]
            stage_edge = edges[:stages_index[miss_stage[0] - 1][-1] + 1]
            current_nodes = set()
            for e in stage_edge:
                current_nodes.add(e[0])
                current_nodes.add(e[1])
            result_node_index = max(current_nodes)
        else:
            result_edge_index = len(edges) - 1
            result_node_index = len(nodes) - 1

    elif miss_stage == [0, 1, 3]:
        result_edge_index = stages_index[miss_stage[-1] - 1][0] - 1
        stage_edge = edges[:stages_index[miss_stage[-1] - 1][0]]
        current_nodes = set()
        for e in stage_edge:
            current_nodes.add(e[0])
            current_nodes.add(e[1])
        result_node_index = max(current_nodes)

    else:
        result_edge_index = stages_index[0][-1]
        stage_edge = edges[:stages_index[0][-1] + 1]
        current_nodes = set()
        for e in stage_edge:
            current_nodes.add(e[0])
            current_nodes.add(e[1])
        result_node_index = max(current_nodes)

    return result_node_index, result_edge_index

def time_sort(relations):

    sort_re = []
    n = len(relations)
    for i in range(n - 1):
        for j in range(n - 1 - i):
            sub1 = relations[j][0]
            obj1 = relations[j][1]
            sub2 = relations[j + 1][0]
            obj2 = relations[j + 1][1]
            if sub1 == sub2 and obj1 == obj2:
                continue
            elif sub1 <= sub2 and sub2 < obj2 and obj2 < obj1:
                relations[j], relations[j + 1] = relations[j + 1], relations[j]
            elif sub1 <= obj2 and obj2 < sub2 and sub2 < obj1:
                relations[j], relations[j + 1] = relations[j + 1], relations[j]
            elif obj1 < sub2 and sub2 < sub1 and sub1 <= obj2:
                relations[j], relations[j + 1] = relations[j + 1], relations[j]
            elif obj1 < sub2 and sub2 < obj2 and obj2 <= sub1:
                relations[j], relations[j + 1] = relations[j + 1], relations[j]
            elif obj1 <= obj2 and obj2 < sub2 and sub2 < sub1:
                relations[j], relations[j + 1] = relations[j + 1], relations[j]
            elif sub2 <= sub1 and sub1 <= obj2 and obj2 <= obj1:
                relations[j], relations[j + 1] = relations[j + 1], relations[j]
            elif sub2 < obj1 and obj1 <= obj2 and obj2 <= sub1:
                relations[j], relations[j + 1] = relations[j + 1], relations[j]
            elif sub2 < obj2 and obj2 <= sub1 and sub1 < obj1:
                relations[j], relations[j + 1] = relations[j + 1], relations[j]
            elif sub2 < obj2 and obj2 <= obj1 and obj1 < sub1:
                relations[j], relations[j + 1] = relations[j + 1], relations[j]
            elif obj2 <= obj1 and obj1 < sub2 and sub2 <= sub1:
                relations[j], relations[j + 1] = relations[j + 1], relations[j]
            elif obj2 < sub2 and sub2 <= sub1 and sub1 < obj1:
                relations[j], relations[j + 1] = relations[j + 1], relations[j]
            elif obj2 < sub2 and sub2 < obj1 and obj1 < sub1:
                relations[j], relations[j + 1] = relations[j + 1], relations[j]
            else:
                continue

    return relations

def get_up_to_index2(miss_stage, stages_index, nodes, edges):
    if miss_stage in [[0, 2], [0, 3], [0, 2, 3]]:
        print(stages_index)
        result_edge_index = stages_index[miss_stage[0] + 1][0] - 1
        stage_edge = edges[:stages_index[miss_stage[0] + 1][0]]
        print(result_edge_index, stage_edge)
        if result_edge_index < 0:
            result_edge_index = len(edges)//3
            stage_edge = edges[:result_edge_index]
        current_nodes = set()
        for e in stage_edge:
            current_nodes.add(int(e[0]))
            current_nodes.add(int(e[1]))
        result_node_index = max(current_nodes)
        stage_target = miss_stage[1:]
    elif miss_stage == [0, 2, 3]:
        result_edge_index = stages_index[miss_stage[0] + 1][0] - 1
        stage_edge = edges[:stages_index[miss_stage[0] + 1][0]]
        current_nodes = set()
        for e in stage_edge:
            current_nodes.add(e[0])
            current_nodes.add(e[1])
        result_node_index = max(current_nodes)

        stage_target = miss_stage[1:]
    else:
        result_edge_index = stages_index[miss_stage[0] - 1][-1]
        stage_edge = edges[:stages_index[miss_stage[0] - 1][-1] + 1]
        current_nodes = set()
        for e in stage_edge:
            current_nodes.add(e[0])
            current_nodes.add(e[1])
        result_node_index = max(current_nodes)

        stage_target = miss_stage[1:]

    return result_node_index, result_edge_index, stage_target

def read_data(json_file):
    docs = [json.loads(line) for line in open(json_file)]
    return docs

if __name__ == '__main__':
    start_time = time.time()
    # make_print_to_file(path='./logs')
    parser = argparse.ArgumentParser(description='GraphFlow model')

    # ******data args******
    parser.add_argument('--dataset', type=str, default='ASG', help='dataset')
    parser.add_argument('--path', type=str, help='path of dataset', required=True)
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--edge_unroll', type=int, default=10, help='max edge to model for each node in bfs order.')
    parser.add_argument('--num_flow_layer', type=int, default=6, help='num of affine transformation layer in each timestep')
    parser.add_argument('--checkpoint_path', type=str, help='path of model', required=True)
    # parser.add_argument("--stages_data_path", type=str, help='path of stage', required=True)
    parser.add_argument("--predict_data_path", type=str, help='path of stage', required=True)
    parser.add_argument("--predict_result_path", type=str, help='path of stage', required=True)
    parser.add_argument("--need_to_be_predicted_stages", type=list, default=[3, 4])
    parser.add_argument('--nhid', type=int, default=128, help='num of hidden units of gcn')
    parser.add_argument('--nout', type=int, default=128, help='num of out units of gcn')
    parser.add_argument('--learn_prior', action='store_true', default=False, help='learn log-var of gaussian prior.')
    parser.add_argument('--is_bn', action='store_true', default=False, help='batch norm on node embeddings.')
    parser.add_argument('--gcn_layer', type=int, default=3, help='num of rgcn layers')
    parser.add_argument('--st_type', type=str, default='sigmoid', help='architecture of st net, choice: [sigmoid, exp, softplus, spine]')
    parser.add_argument('--scale_weight_norm', action='store_true', default=False, help='apply weight norm on scale factor.')
    parser.add_argument('--sigmoid_shift', type=float, default=2.0, help='sigmoid shift on s.')
    parser.add_argument('--is_bn_before', action='store_true', default=False, help='batch norm on node embeddings on st-net input.')
    parser.add_argument('--divide_loss', action='store_true', default=False, help='divide loss by length of latent.')
    parser.add_argument('--temperature', type=float, default=0.75, help='temperature for normal distribution')
    parser.add_argument('--max_atoms', type=int, default=100, help='maximum #atoms of generated mol')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # stage_dict = read(args.stages_data_path)
    data_config = {'atom_list': ["MP", "TP", "MF", "SF", "TF", "SO"], 'freedom': 0, 'node_dim': 7, 'max_size': 100, 'bond_dim': 10}
    edge_dict = {"RD": 1, "WR": 2, "EX": 3, "UK": 4, "CD": 5, "FR": 6, "IJ": 7, "ST": 8, "RF": 9}
    node_dict = {"MP": 1, "TP": 2, "MF": 3, "SF": 4, "TF": 5, "SO": 6}
    edge_label_dict = swap_keys_values(edge_dict)
    max_size = data_config['max_size']
    node_dim = data_config['node_dim'] - 1  # exclude padding dim.
    bond_dim = data_config['bond_dim']


    num, graph_nodes, graph_edges = init_graph(args.path, edge_dict)
    data_js = read_data(args.predict_data_path)
    f = open(args.predict_result_path, 'w', encoding='utf-8')
    # node_numpy, edge_numpy, graph_line, graph_nodes, graph_edges = np_builder(args.path, stage_dict, args.need_to_be_predicted_stages[0])

    graphmodel = GraphFlowModel(max_size, node_dim, bond_dim, args.edge_unroll, args)

    checkpoint = torch.load(args.checkpoint_path)
    graphmodel.load_state_dict(checkpoint['model_state_dict'])

    # graphmodel.load_state_dict(torch.load(args.checkpoint_path))
    graphmodel.eval()

    all_smiles = []
    pure_valids = []
    appear_in_train = 0.
    # start_t = time()
    cnt_mol = 0
    cnt_gen = 0

    graph_id = 0

    rea_count = 0

    while graph_id < num:  # 依次生成每一个分子式
        new_json = {}
        print("开始图{}的补全".format(data_js[graph_id]['doc_key']))
        judge_graph = Reasonable_judgment(graph_nodes[graph_id], graph_edges[graph_id], edge_label_dict)
        flag, stage_num, flow_index, edges_index = judge_graph.knowledge_judge()
        stage_need_predict = get_miss_stage(edges_index, graph_edges[graph_id])
        print("图{}需要补充的阶段为：{}".format(graph_id, stage_need_predict))
        if len(stage_need_predict) != 0:
            if stage_need_predict in [[0, 2], [0, 3], [1, 2], [1, 3], [0, 2, 3]]:
                cut_node, cut_edge, target = get_up_to_index2(stage_need_predict, edges_index, graph_nodes[graph_id], graph_edges[graph_id])
                node_numpy, edge_numpy, graph_line = np_builder(graph_nodes[graph_id], graph_edges[graph_id], cut_node, cut_edge, args.max_atoms, node_dict, edge_dict)
                print("接下来补充第{}阶段".format(stage_need_predict[0]))
                new_nodes, new_edges, new_nodes_name = graphmodel.generate(node_numpy[0], edge_numpy[0], cut_node, cut_edge, graph_nodes[graph_id], graph_edges[graph_id], data_js[graph_id]["ners"], graph_line, edge_label_dict, target, args.temperature, mute=True, max_atoms=args.max_atoms, cnt=cnt_gen)

                judge_graph = Reasonable_judgment(new_nodes, new_edges, edge_label_dict)
                flag, stage_num, flow_index, edges_index = judge_graph.knowledge_judge()
                stage_need_predict = get_miss_stage(edges_index, new_edges)
                print("接下来补充第{}阶段".format(stage_need_predict))
                cut_node, cut_edge = get_up_to_index1(stage_need_predict, edges_index, new_nodes, new_edges)
                node_numpy, edge_numpy, graph_line = np_builder(new_nodes, new_edges, cut_node, cut_edge, args.max_atoms, node_dict, edge_dict)
                target = []
                new_nodes, new_edges, new_nodes_name = graphmodel.generate(node_numpy[0], edge_numpy[0], cut_node, cut_edge, new_nodes, new_edges, new_nodes_name, graph_line, edge_label_dict, target, args.temperature, mute=True, max_atoms=args.max_atoms, cnt=cnt_gen)

            else:
                print("接下来补充第{}阶段".format(stage_need_predict))
                cut_node, cut_edge = get_up_to_index1(stage_need_predict, edges_index, graph_nodes[graph_id], graph_edges[graph_id])
                print(cut_node, cut_edge)
                node_numpy, edge_numpy, graph_line = np_builder(graph_nodes[graph_id], graph_edges[graph_id], cut_node, cut_edge, args.max_atoms, node_dict, edge_dict)
                target = []
                new_nodes, new_edges, new_nodes_name = graphmodel.generate(node_numpy[0], edge_numpy[0], cut_node, cut_edge, graph_nodes[graph_id], graph_edges[graph_id], data_js[graph_id]["ners"], graph_line, edge_label_dict, target, args.temperature, mute=True, max_atoms=args.max_atoms, cnt=cnt_gen)

            print(new_nodes)
            print(new_nodes_name)
            print(new_edges)
            new_json['doc_key'] = data_js[graph_id]['doc_key']
            new_json['ners'] = new_nodes_name
            new_json['relations'] = new_edges
            json.dump(new_json, f)
            f.write('\n')
            judge_graph = Reasonable_judgment(new_nodes, new_edges, edge_label_dict)
            flag_, stage_num_, flow_index_, edges_index_ = judge_graph.knowledge_judge()


            if flag_:
                rea_count += 1
                print("图{}通过四段合理性验证, 每个阶段通过事件流的事件流序号为{}, 每个阶段通过的边序号为{}".format(graph_id, flow_index_, edges_index_))
            else:
                print("图{}通过{}段合理性验证, 每个阶段通过事件流的事件流序号为{}, 每个阶段通过的边序号为{}".format(graph_id, stage_num_, flow_index_, edges_index_))
        else:
            rea_count += 1
            print("图{}通过四段合理性验证, 每个阶段通过事件流的事件流序号为{}, 每个阶段通过的边序号为{}".format(graph_id, flow_index, edges_index))
            new_json['doc_key'] = data_js[graph_id]['doc_key']
            # new_json['ners'] = graph_nodes[graph_id]
            new_json['ners'] = data_js[graph_id]['ners']
            # new_json['relations'] = graph_edges[graph_id]
            new_json['relations'] = data_js[graph_id]['relations']
            json.dump(new_json, f)
            f.write('\n')
        graph_id += 1

    print("总计{}张图，补充完后{}通过合理性验证".format(num, rea_count))

    end_time = time.time()

    # 计算代码运行时间（以秒为单位）
    execution_time = end_time - start_time

    print("代码执行时间：", execution_time, "秒")
