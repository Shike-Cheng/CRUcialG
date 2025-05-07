'''
Author: Wenrui Cheng
Date: 2023-09-26 12:58:30
FilePath: deal_with_generate.py
Description: 这里对阶段自动补全后的图进行共指节点的删除，构图
'''

# 修改，不应该删除游离节点，应该删除的是共指节点
# 输出结果应该统一，便于消融实验的评估，即输出一个NLP的结果，不包含共指关系

# 再修改，应该对边最后排序，合并同一名称的实体

import json
import argparse
import os


def read(json_file):
    docs = [json.loads(line) for line in open(json_file)]
    return docs

# def get_new_ner_re(ns, rs):
#
#     final_ner = []
#     final_re = []
#     free_nodes = []
#     re_nodes = set()
#     for e in rs:
#         re_nodes.add(e[0])
#         re_nodes.add(e[1])
#     for i in range(len(ns)):
#         if i in re_nodes:
#             continue
#         else:
#             free_nodes.append(i)
#     ner_indice_map = {}
#     new_ner_indice_map = {}
#     n_id = 0
#     for n in ns:
#         ner_indice_map[n_id] = (n[0], n[1])
#         if n_id not in free_nodes:
#             final_ner.append(n)
#         n_id += 1
#     # print(ner_indice_map)
#     new_n_id = 0
#     for n in final_ner:
#         new_ner_indice_map[(n[0], n[1])] = new_n_id
#         new_n_id += 1
#     for re in rs:
#         final_re.append([new_ner_indice_map[ner_indice_map[re[0]]], new_ner_indice_map[ner_indice_map[re[1]]], re[2]])
#
#     return final_ner, final_re
no_re_map = {1: 'RD', 2: 'WR', 3: 'EX', 4: 'UK', 5: 'CD', 6: 'FR', 7: 'IJ', 8: 'ST', 9: 'RF', 10: 'CO'}
class Unreasonable_judgment_revise(object):
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        self.node_type = []
        for n in self.nodes:
            self.node_type.append(n[2])

    def remove_adjacent_duplicates(self, input_list):
        result = []
        for item in input_list:
            if len(result) == 0 or item != result[-1]:
                result.append(item)
        return result


    def remove_elements_by_indices(self, lst, indices):
        return [elem for idx, elem in enumerate(lst) if idx not in indices]

    def Adjustment_sub_obj(self, ns, rs):
        for r in rs:
            if no_re_map[r[2]] == 'RF':
                if ns[r[0]][2] == 'SO':
                    r[0], r[1] = r[1], r[0]
        return rs

    def calculate_have_degree(self, node, start, edges):
        re_count = 0
        re_list = []
        for i in range(start + 1, len(edges)):
            if node == edges[i][0] or node == edges[i][1]:
                re_count += 1
                re_list.append(i)
        if re_count > 0:
            return True, re_list
        else:
            return False, None

    def calculate_FR(self, node, start, edges):
        fr_count = 0
        fr_list = []               # node作为客体，被FR的关系下标
        for i in range(start, len(edges)):
            if node == edges[i][1] and edges[i][2] == 6:
                fr_count += 1
                fr_list.append(i)

        if fr_count > 0:
            degree_list = []       # 每次被FR之后，node在该FR之后的出度
            for j in fr_list:
                degree_count = 0
                for x in range(j, len(edges)):           # 计算在j之后，node的出度
                    if node == edges[x][0]:
                        degree_count += 1
                degree_list.append(degree_count)

            if max(degree_list) == 0:
                del fr_list[0]
                return True, fr_list                     # 返回存在多次FR的情况，当node没有出度时，保留最开始的FR
            else:
                max_index = degree_list.index(max(degree_list))
                del fr_list[max_index]
                return True, fr_list                     # 返回存在多次FR的情况，除去需要保留的那次FR
        else:
            return False, None

    def find_interaction(self, sub, obj, start, edges):
        in_count = 0
        in_list = []
        for i in range(start, len(edges)):
            if sub == edges[i][1] and obj == edges[i][0]:
                in_count += 1
                in_list.append(i)

        if in_count > 0:
            return True, in_list
        else:
            return False, None

    def find_forward_action(self, sub, obj, start):
        find_re = []
        for i in range(0, start):
            if self.edges[i][0] == sub and self.edges[i][1] == obj:
                find_re.append(self.edges[2])
        return find_re

    def find_sub_action(self, sub, start):
        find_re = []
        for i in range(0, start):
            if self.edges[i][0] == sub:
                find_re.append(i)
        return find_re

    def add_elements_by_index(self, input_list, index_element_dict):
        for index, element in sorted(index_element_dict.items(), reverse=True):
            input_list.insert(index, element)

    def swap_elements_by_index(self, lst, index1, index2):
        if 0 <= index1 < len(lst) and 0 <= index2 < len(lst):
            lst[index1], lst[index2] = lst[index2], lst[index1]

    def Unreasonable_rule(self):
        self.edges = self.Adjustment_sub_obj(self.nodes, self.edges)

        delete_list = []

        # 删除边
        de_ti_me = 0
        for e in self.edges:
            # p/f在被进程UK后不能参与事件交互，删除其后续的事件交互
            if e[2] == 4:
                flag, k_list1 = self.calculate_have_degree(e[1], de_ti_me, self.edges)
                if flag:
                    # print("case 1: ", k_list1)
                    delete_list += k_list1
                    de_ti_me += 1
                    continue
            # p不能多次被FR，保留其后续作为主体，出度最多的FR或第一次的FR
            if e[2] == 6:
                flag, k_list2 = self.calculate_FR(e[1], de_ti_me, self.edges)
                if flag:
                    # print("case 2: ", k_list2)
                    delete_list += k_list2
                    de_ti_me += 1
                    continue
            # 客体进程无法对主体进程操作
            if self.node_type[e[0]][-1] == 'P' and self.node_type[e[1]][-1] == 'P':
                flag, k_list3 = self.find_interaction(e[0], e[1], de_ti_me, self.edges)
                if flag:
                    # print("case 3: ", k_list3)
                    delete_list += k_list3
                    de_ti_me += 1
                    continue

            de_ti_me += 1

        self.edges = self.remove_elements_by_indices(self.edges, delete_list)

        # 调整边的时序
        adjust_time = 0
        adjust_dict = {}
        for e in self.edges:
            if self.node_type[e[1]][-1] == 'P' and e[2] == 6:
                if len(self.find_sub_action(e[1], adjust_time)) != 0:
                    adjust_dict[adjust_time] = self.find_sub_action(e[1], adjust_time)[0]
            adjust_time += 1

        for k, v in adjust_dict.items():
            self.swap_elements_by_index(self.edges, k, v)

        self.edges = self.remove_adjacent_duplicates(self.edges)

        return self.edges


def get_new_ner_re(ns, rs, co_ns):

    final_ner = []
    final_re = []
    del_nodes = []
    re_nodes = set()
    for e in rs:
        re_nodes.add(e[0])
        re_nodes.add(e[1])

    for n in co_ns:
        if n not in re_nodes:
            del_nodes.append(n)

    ner_indice_map = {}
    new_ner_indice_map = {}
    n_id = 0
    for n in ns:
        ner_indice_map[n_id] = (n[0], n[1])
        if n_id not in del_nodes:
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


def make_graphaf_txt(new_nodes, new_edges, li):
    graph_name_dot = li + ".dot"
    ff = open(args.graph_to_repair_txt, 'a', encoding='utf-8')
    graph_name_dot_path = li + '\\' + graph_name_dot
    ff.write('#' + graph_name_dot_path + '\n')

    ff.write(str(len(new_nodes)) + '\n')
    for n in new_nodes:
        ff.write(n[2] + '\n')

    ff.write(str(len(new_edges)) + '\n')

    for edge in new_edges:
        ff.write(str(edge[0]) + ' ' + str(edge[1]) + ' ' + no_re_map[edge[2]] + '\n')


if __name__ == "__main__":
    no_re_map = {1: 'RD', 2: 'WR', 3: 'EX', 4: 'UK', 5: 'CD', 6: 'FR', 7: 'IJ', 8: 'ST', 9: 'RF', 10: 'CO'}
    parser = argparse.ArgumentParser()

    parser.add_argument('--graph_json', type=str, default=None, required=True,
                        help="Output of language model")
    parser.add_argument('--graph_to_repair_json', type=str, default=None, required=True,
                        help="graph repair json dataset")
    parser.add_argument('--graph_to_repair_txt', type=str, default=None, required=True,
                        help="graph repair txt dataset")

    args = parser.parse_args()
    # graphlist = os.listdir(args.graph_dir)
    data_json = read(args.graph_json)
    f = open(args.graph_to_repair_json, 'w', encoding='utf-8')
    for data in data_json:
        new_json = {}
        # data = json.load(open(args.graph_dir + '\\' + li, "r", encoding="utf-8"))
        ners, res = get_new_ner_re(data['ners'], data['co_replace_sort_relations'], data['co_ners'])
        if len(ners) == 0:
            continue
        new_delete_judgment = Unreasonable_judgment_revise(ners, res)
        new_re = new_delete_judgment.Unreasonable_rule()

        make_graphaf_txt(ners, new_re, data['doc_key'])
        new_json['doc_key'] = data['doc_key']
        new_json['ners'] = ners
        new_json['relations'] = new_re

        json.dump(new_json, f)
        f.write('\n')



