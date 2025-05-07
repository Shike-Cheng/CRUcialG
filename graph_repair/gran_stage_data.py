# !/usr/bin/env python
# -*-coding:utf-8 -*-

import numpy as np
import re
import json
import os
import shutil


def np_mol_builder(fname, data, cut_stage):

    node_number_list = []
    for i in range(0, len(data["stage_nodes"])):
        node_number_list.append(data["stage_nodes"][i][-1] + 1)
    np_mol = np.array(node_number_list)
    return (len(node_number_list), 100)


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
            num_nodes = int(data[index])
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


