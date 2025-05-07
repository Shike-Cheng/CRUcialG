'''
Author: Wenrui Cheng
Date: 2023-09-26 12:58:30
FilePath: deal_with_generate.py
Description: 这里对阶段自动补全后的图进行冗余节点的删除，构图
'''
# 应该增加合并相同name的实体，未完成

import sys
sys.path.append('..')
import json
import graphviz
import argparse
from graph2repair import Unreasonable_judgment_revise
from config import Reasonable_judgment, no_re_map

node_shapes = {
    'F': 'rect',             # 正方形
    'O': 'diamond',          # 菱形
    'P': 'ellipse'           # 椭圆形（与圆形相同）
}

def read(json_file):
    docs = [json.loads(line) for line in open(json_file)]
    return docs


def get_new_ner_re(ns, rs):

    final_ner = []
    final_re = []
    free_nodes = []
    re_nodes = set()
    for e in rs:
        re_nodes.add(e[0])
        re_nodes.add(e[1])
    for i in range(len(ns)):
        if i in re_nodes:
            continue
        else:
            free_nodes.append(i)
    ner_indice_map = {}
    new_ner_indice_map = {}
    n_id = 0
    for n in ns:
        ner_indice_map[n_id] = (n[0], n[1])
        if n_id not in free_nodes:
            final_ner.append(n)
        n_id += 1
    # print(ner_indice_map)
    new_n_id = 0
    for n in final_ner:
        new_ner_indice_map[(n[0], n[1])] = new_n_id
        new_n_id += 1
    for re in rs:
        final_re.append([new_ner_indice_map[ner_indice_map[re[0]]], new_ner_indice_map[ner_indice_map[re[1]]], re[2]])

    return final_ner, final_re


def draw_graph(new_nodes, new_edges, li):

    graph_name_dot = li + ".dot"
    graph_name_dot_path = args.asg_reconstruction_graph + '\\' + graph_name_dot
    new_graph = graphviz.Digraph(graph_name_dot_path, filename=graph_name_dot)
    new_graph.body.extend(
        ['rankdir="LR"', 'size="9"', 'fixedsize="false"', 'splines="true"', 'nodesep=0.3', 'ranksep=0',
         'fontsize=10',
         'overlap="scalexy"',
         'engine= "neato"'])
    n_id = 0
    for n in new_nodes:
        if len(n) > 3:
            n[4] = n[4].replace('\\', '\\\\')
            new_graph.node(str(n_id), label=n[4] + '_' + n[2], shape=node_shapes[n[2][1]])
        else:
            new_graph.node(str(n_id), label=n[2], shape=node_shapes[n[2][1]])

        n_id += 1
    e_id = 0
    for edge in new_edges:
        new_graph.edge(str(edge[0]), str(edge[1]), label=str(e_id) + '. ' + no_re_map[edge[2]])
        e_id += 1

    # new_graph.render(graph_name_dot_path, view=False)
    # new_graph.save(graph_name_dot_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--graph_generator_json', type=str, default=None, required=True,
                        help="result from ner model")
    parser.add_argument('--asg_reconstruction_json', type=str, default=None, required=True,
                        help="ner screening results")
    parser.add_argument('--asg_reconstruction_graph', type=str, default=None, required=True,
                        help="ner screening results")

    args = parser.parse_args()

    # re_predict = r"E:\深信服测试\xiaoqi_graph_to_predict_result.json"
    # final_graph_data = r"E:\深信服测试\xiaoqi_completion_data"
    # final_graph = r"E:\深信服测试\xiaoqi_completion"


    # re_predict_data = read(re_predict)
    re_predict_data = read(args.graph_generator_json)
    f = open(args.asg_reconstruction_json, 'w', encoding='utf-8')
    for d_js in re_predict_data:
        new_json = {}
        print(d_js['doc_key'])
        ners = d_js['ners']
        res = d_js['relations']

        indice = -1
        for i in range(len(ners)):
            if len(ners[i]) < 2:
                ners[i] = [indice, indice, ners[i][0]]
                indice -= 1
        # print(ners)
        new_delete_judgment = Unreasonable_judgment_revise(ners, res)
        new_re = new_delete_judgment.Unreasonable_rule()

        # judge_graph = Reasonable_judgment(ners, new_re)
        # flag, stage_num, flow_index, edges_index = judge_graph.knowledge_judge()

        new_ners, new_res = get_new_ner_re(ners, new_re)
        draw_graph(new_ners, new_res, d_js['doc_key'])
        for r in new_res:
            r[2] = no_re_map[r[2]]
        new_json['doc_key'] = d_js['doc_key']
        new_json['ners'] = new_ners
        new_json['relations'] = new_res
        json.dump(new_json, f)
        f.write('\n')

