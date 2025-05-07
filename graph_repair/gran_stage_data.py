# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : trans_ASG.py
# Time       ：2023/7/5 21:14
# Author     ：Qixuan Yuan
# Description：数据集转为3个npy文件
"""
import numpy as np
import re
import json
import os
import shutil

#制作每张图节点个数的npy文件
def np_mol_builder(fname, data, cut_stage):

    node_number_list = []
    for i in range(0, len(data["stage_nodes"])):
        node_number_list.append(data["stage_nodes"][i][-1] + 1)
    print(max(node_number_list), len(node_number_list))
    np_mol = np.array(node_number_list)
    # np.save("./data_preprocessed/99_CTI_stage12_np_mol.npy",np_mol)
    return (len(node_number_list), 100)


# def np_builder(fname, stage_data, cut_stage):
#     node_dic = {"MP": 1, "TP": 2, "MF": 3, "SF": 4, "TF": 5, "SO": 6}
#     edge_dic = {"RD": 1, "WR": 2, "EX": 3, "UK": 4, "CD": 5, "FR": 6, "IJ": 7, "ST": 8, "RF": 9}
#     graph_number, max_node_num = np_mol_builder(fname, stage_data, cut_stage)
#     np_node = np.zeros((graph_number, max_node_num))
#     np_adj = np.zeros((graph_number, len(edge_dic) + 1, max_node_num, max_node_num))
#     file_path = "./dataset/" + fname
#     data = open(file_path, "r", encoding="utf-8").read().splitlines()
#     graph_f = open('./data_preprocessed/99_CTI_stage12_graph.txt', 'w')
#     graph_node = []
#     graph_edge = []
#     graph_list = []
#     index = 0
#     graph_id = 0
#     as_of_index = cut_stage - 2
#     while index < len(data):
#         item = data[index]
#         index = index + 1
#         if index < len(data) and len(item) > 0 and item[0] == '#':
#             originl_graph = ""
#             num_nodes = int(data[index]) # 该图节点数
#             index = index + 1
#             sum_nodes = []
#             for i in range(num_nodes):
#                 if i <= stage_data['stage_nodes'][graph_id][as_of_index]:
#                     np_node[graph_id, i] = node_dic[data[index].rstrip("\n")]
#                     originl_graph += data[index].rstrip("\n") + " "
#                     sum_nodes.append(data[index])
#                 index = index + 1
#
#             num_edges = int(data[index])
#             index = index + 1
#             sum_edges = []
#             node_pair = []
#             for j in range(num_edges):
#                 if j <= stage_data['stage_edges'][graph_id][as_of_index]:
#                     tmp = data[index].split()
#                     [start_node, end_node, edge_label] = tmp
#                     np_adj[graph_id, edge_dic[edge_label] - 1, int(start_node), int(end_node)] = 1
#                     np_adj[graph_id, edge_dic[edge_label] - 1, int(end_node), int(start_node)] = 1
#                     originl_graph += start_node + " " + end_node + " " + edge_label + " "
#                     sum_edges.append([int(start_node), int(end_node), edge_dic[edge_label]])
#                     node_pair.append([int(start_node), int(end_node)])
#                 index += 1
#
#             for x in range(num_nodes):
#                 for y in range(num_nodes):
#                     if [x, y] not in node_pair and [y, x] not in node_pair:
#                         np_adj[graph_id, len(edge_dic), x, y] = 1
#                         np_adj[graph_id, len(edge_dic), y, x] = 1
#
#             graph_list.append(originl_graph)
#             graph_edge.append(sum_edges)
#             graph_node.append(sum_nodes)
#             # graph_f.write(originl_graph + "\n")
#             graph_id += 1
#
#     np_node = np_node.astype(int)
#     # np.save("./data_preprocessed/99_CTI_stage12_np_node.npy", np_node)
#     # print(np.array(np_node))
#     np_adj = np_adj.astype(int)
#     # print(np.array(np_adj))
#     # np.save("./data_preprocessed/99_CTI_stage12_np_adj.npy", np_adj)
#
#     return np_node, np_adj, graph_list, graph_node, graph_edge

def np_builder1(nodes, edges, max_node_num, node_dic, edge_dic):
    np_node = np.zeros((1, max_node_num))
    np_adj = np.zeros((1, len(edge_dic) + 1, max_node_num, max_node_num))
    edge_label_dict = {v: k for k, v in edge_dic.items()}
    originl_graph = ""
    for i in range(len(nodes)):
        np_node[0, i] = node_dic[nodes[i]]
        originl_graph += nodes[i] + " "

    node_pair = []
    for j in range(len(edges)):
        np_adj[0, edges[j][2] - 1, edges[j][0], edges[j][1]] = 1
        np_adj[0, edges[j][2] - 1, edges[j][1], edges[j][0]] = 1
        originl_graph += str(edges[j][0]) + " " + str(edges[j][1]) + " " + edge_label_dict[edges[j][2]] + " "
        node_pair.append([edges[j][0], edges[j][1]])

    for x in range(len(nodes)):
        for y in range(len(nodes)):
            if [x, y] not in node_pair and [y, x] not in node_pair:
                np_adj[0, len(edge_dic), x, y] = 1
                np_adj[0, len(edge_dic), y, x] = 1

    np_node = np_node.astype(int)
    np_adj = np_adj.astype(int)

    return np_node, np_adj, originl_graph


def np_builder(nodes, edges, cut_node, cut_edge, max_node_num, node_dic, edge_dic):
    np_node = np.zeros((1, max_node_num))
    np_adj = np.zeros((1, len(edge_dic) + 1, max_node_num, max_node_num))
    edge_label_dict = {v: k for k, v in edge_dic.items()}
    originl_graph = ""
    for i in range(len(nodes)):
        if i <= cut_node:
            np_node[0, i] = node_dic[nodes[i]]
            originl_graph += nodes[i] + " "
    print(len(nodes))
    node_pair = []
    for j in range(len(edges)):
        if j <= cut_edge:
            np_adj[0, edges[j][2] - 1, edges[j][0], edges[j][1]] = 1
            np_adj[0, edges[j][2] - 1, edges[j][1], edges[j][0]] = 1
            originl_graph += str(edges[j][0]) + " " + str(edges[j][1]) + " " + edge_label_dict[edges[j][2]] + " "
            node_pair.append([edges[j][0], edges[j][1]])

    # for x in range(len(nodes)):
    #     for y in range(len(nodes)):
    for x in range(max_node_num):
        for y in range(max_node_num):
            if [x, y] not in node_pair and [y, x] not in node_pair:
                np_adj[0, len(edge_dic), x, y] = 1
                np_adj[0, len(edge_dic), y, x] = 1

    np_node = np_node.astype(int)
    np_adj = np_adj.astype(int)

    return np_node, np_adj, originl_graph

def init_graph(fname, edge_dic):
    # node_dic = {"MP": 1, "TP": 2, "MF": 3, "SF": 4, "TF": 5, "SO": 6}
    # edge_dic = {"RD": 1, "WR": 2, "EX": 3, "UK": 4, "CD": 5, "FR": 6, "IJ": 7, "ST": 8, "RF": 9}
    data = open(fname, "r", encoding='utf-8').read().splitlines()
    index = 0
    nodes_graph = []
    edges_graph = []
    graph_id = 0
    while index < len(data):
        item = data[index]
        index = index + 1
        if index < len(data) and len(item) > 0 and item[0] == '#':
            num_nodes = int(data[index])  # 该图节点数
            index = index + 1
            sum_nodes = []
            for i in range(num_nodes):
                sum_nodes.append(data[index])
                index = index + 1

            sum_edges = []
            num_edges = int(data[index])
            index = index + 1

            for j in range(num_edges):
                tmp = data[index].split()
                [start_node, end_node, edge_label] = tmp
                sum_edges.append([int(start_node), int(end_node), edge_dic[edge_label]])
                index += 1

            nodes_graph.append(sum_nodes)
            edges_graph.append(sum_edges)
            graph_id += 1

    assert len(nodes_graph) == len(edges_graph)

    return len(nodes_graph), nodes_graph, edges_graph

def read(json_file):
    data_ = open(json_file, "r", encoding="utf-8")
    docs = json.load(data_)
    return docs

if __name__ == '__main__':
    json_path = r"E:\GraphAF\dataset\stage_index.json"
    cut_stage = 3
    json_data = read(json_path)
    np_builder("99_CTI.txt", json_data, cut_stage)

