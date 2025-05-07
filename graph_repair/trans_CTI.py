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
import os
import shutil

#制作每张图节点个数的npy文件
def np_mol_builder(fname):
    file_path = "./dataset/" + fname
    f = open(file_path, "r", encoding="utf-8")
    graph_lines = f.readlines()
    f.close()
    node_number_list = []
    for i in range(0,len(graph_lines)):
        if graph_lines[i][0] == "#":
            if int(graph_lines[i+1]) > 100:
                continue
            else:
                node_number_list.append(int(graph_lines[i+1].replace("\n","")))
    print(max(node_number_list))
    np_mol =np.array(node_number_list)
    np.save("./data_preprocessed/99_train_np_mol.npy",np_mol)
    return (len(node_number_list),max(node_number_list))

#制作每张图节点类型的npy文件
def np_node_builder(fname):
    node_dic = {"MP":1,"TP":2,"MF":3,"SF":4,"TF":5,"SO":6}
    graph_number, max_node_num = np_mol_builder(fname)
    np_node = np.zeros((graph_number, max_node_num))
    file_path = "./dataset/" + fname
    f = open(file_path, "r", encoding="utf-8")
    graph_lines = f.readlines()
    f.close()
    graph_lindex = 0
    for i in range(0,len(graph_lines)):
        if graph_lines[i][0] == "#":
            if int(graph_lines[i+1]) > 100:
                continue
            else:
                graph_lindex += 1
                for j in range(1,max_node_num+1):
                    node_line = min(len(graph_lines)-1,i+j)
                    if graph_lines[node_line].rstrip("\n") in node_dic.keys():
                        np_node[graph_lindex-1,j-2] = node_dic[graph_lines[node_line].rstrip("\n")]
    np.node = np_node.astype(int)
    np.save("./data_preprocessed/99_train_np_node.npy",np_node)
    return np_node

#制作每张图邻接矩阵的npy文件
def np_adj_builder(fname):
    edge_dic = {"RD":1,"WR":2,"EX":3,"UK":4,"CD":5,"FR":6,"IJ":7,"ST":8,"RF":9}
    graph_number, max_node_num = np_node_builder(fname).shape
    np_adj = np.zeros((graph_number,len(edge_dic),max_node_num,max_node_num))
    file_path = "./dataset/" + fname
    f = open(file_path, "r", encoding="utf-8")
    graph_lines = f.readlines()
    f.close()
    graph_index = 0
    for i in range(0,len(graph_lines)):
        if graph_lines[i][0] == "#":
            if int(graph_lines[i+1]) > 100:
                continue
            else:
                graph_index += 1
                for j in range(1,1000):
                    edge_line = min(len(graph_lines) - 1, i + j)
                    if graph_lines[edge_line][0] == "#":
                        break
                    elif re.match(r"\d+ \d+ \w+", graph_lines[edge_line].rstrip("\n")):
                        match = re.match(r"(\d+) (\d+) (\w+)", graph_lines[edge_line].rstrip("\n"))
                        np_adj[graph_index-1,edge_dic[match.group(3)]-1,int(match.group(1)),int(match.group(2))] = 1
                        np_adj[graph_index - 1, edge_dic[match.group(3)]-1, int(match.group(2)), int(match.group(1))] = 1
                    else:
                        continue
        else:
            continue
    np.node = np_adj.astype(int)
    np.save("./data_preprocessed/99_train_np_adj.npy", np_adj)
    return np_adj

#转换为一行表示一张图的形式
def graph_txt_builder(fname):
    node_dic = {"MP":1,"TP":2,"MF":3,"SF":4,"TF":5,"SO":6}
    file_path = "./dataset/" + fname
    f = open(file_path, "r", encoding="utf-8")
    graph_lines = f.readlines()
    f.close()
    graph_lindex = -1
    graph_list = []
    with open('./data_preprocessed/99_train_graph.txt', 'w') as file:
        for i in range(0,len(graph_lines)):
            if graph_lines[i][0] == "#":
                if int(graph_lines[i+1]) > 100:
                    continue
                else:
                    originl_graph = ""
                    graph_lindex += 1
                    for j in range(1,1000):
                        node_edge_line = min(len(graph_lines) - 1, i + j)
                        if graph_lines[node_edge_line][0] == "#" or (i + j)>len(graph_lines)-1:
                            break
                        elif graph_lines[node_edge_line].rstrip("\n") in node_dic.keys():
                            originl_graph = originl_graph+graph_lines[node_edge_line].rstrip("\n")+" "
                        elif re.match(r"\d+ \d+ \w+", graph_lines[node_edge_line].rstrip("\n")):
                            edge_temp = graph_lines[node_edge_line].rstrip("\n")
                            originl_graph = originl_graph + edge_temp + " "
                        else:
                            continue
                    graph_list.append(originl_graph)
                    file.write(originl_graph+"\n")
            else:
                continue
    return graph_list
#统一转换三个numpy
def np_trans(fname):
    np_mol_builder(fname)
    print("节点个数文件已转换")
    np_node_builder(fname)
    print("节点类型文件已转换")
    np_adj_builder(fname)
    print("邻接矩阵文件已转换")
    graph_txt_builder(fname)
    print("逐行图形式已写入")
    return graph_txt_builder(fname)

if __name__ == '__main__':
    np_trans("99_train.txt")

