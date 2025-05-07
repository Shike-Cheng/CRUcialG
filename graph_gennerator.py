'''
Author: Wenrui Cheng
Date: 2023-09-26 12:58:30
Description: 合并实体和关系预测结果，保留NLP结果的原始信息（共指节点、关系）
'''

import json
import argparse
from operator import itemgetter

def read(json_file):
    docs = [json.loads(line) for line in open(json_file)]
    return docs

def ca_sentid(start, js):
    sum_len = 0
    for i in range(len(js["sentences"])):
        sum_len += len(js["sentences"][i])
        if start < sum_len:
            return i

def tran_token(h, t, win, cti_data, win_id):
    #win_sent_hid = ca_sentid(h, win)
    #cti_sent_id = win_sent_hid + win_id
    for i in range(win_id):
        h += len(cti_data["sentences"][i])
        t += len(cti_data["sentences"][i])
    return h, t

def get_graph(predict_data, data_js, ner_js, win_size):
    graph_list = []

    hrt_dict = dict()

    for hrt in predict_data:
        hrt_dict[hrt['title']] = []
    for hrt in predict_data:
        hrt_dict[hrt['title']].append([hrt['h_idx']] + [hrt['t_idx']] + [hrt['r']])

    win_data = []
    for data in data_js:
        new_data = {}
        new_data["doc_key"] = data["title"]
        new_data["sentences"] = data["sents"]
        new_data["hrt"] = []
        if data["title"] in hrt_dict.keys():
            hrt_list = hrt_dict[data["title"]]
            for sro in hrt_list:
                h_id1 = data["vertexSet"][sro[0]][0]["global_pos"][0]
                h_id2 = data["vertexSet"][sro[0]][0]["global_pos"][0] + (
                            data["vertexSet"][sro[0]][0]["pos"][1] - data["vertexSet"][sro[0]][0]["pos"][0]) - 1
                h_name = data["vertexSet"][sro[0]][0]["name"]
                h_type = data["vertexSet"][sro[0]][0]["type"]
                t_id1 = data["vertexSet"][sro[1]][0]["global_pos"][0]
                t_id2 = data["vertexSet"][sro[1]][0]["global_pos"][0] + (
                            data["vertexSet"][sro[1]][0]["pos"][1] - data["vertexSet"][sro[1]][0]["pos"][0]) - 1
                t_name = data["vertexSet"][sro[1]][0]["name"]
                t_type = data["vertexSet"][sro[1]][0]["type"]
                san_yuan_zu = [h_id1] + [h_id2] + [h_type] + [t_id1] + [t_id2] + [t_type] + [h_name] + [t_name] + [
                    sro[2]]

                # sent_id = ca_sentid(h_id1, new_data)
                new_data["hrt"].append(san_yuan_zu)
        else:
            new_data["hrt"] = []

        win_data.append(new_data)
        # print(win_data)

    title = []
    for win in win_data:
        temp = win['doc_key'].rsplit("-", 1)[0]
        if temp not in title:
            title.append(temp)
    cti_data = {}

    for t in title:
        cti_data[t] = []

    for win in win_data:
        cti_data[win['doc_key'].rsplit("-", 1)[0]].append(win)

    for key, value in cti_data.items():
        final_cti = {}
        final_cti['doc_key'] = key
        final_cti["sentences"] = []
        win_id = 0
        # 得到准确的CTI报告文本
        for win in value:
            if win_id == 0:
                for sent in win["sentences"]:
                    final_cti["sentences"].append(sent)
            else:
                final_cti["sentences"].append(win["sentences"][win_size - 1])
            win_id += 1

        final_cti["relation"] = [[] for i in range(len(final_cti["sentences"]))]
        win_id = 0
        for win in value:
            if win_id == 0:
                for hrt in win["hrt"]:
                    final_cti["relation"][ca_sentid(hrt[0], final_cti)].append(hrt)
            else:
                for hrt in win["hrt"]:
                    hrt[0], hrt[1] = tran_token(hrt[0], hrt[1], win, final_cti, win_id)
                    hrt[3], hrt[4] = tran_token(hrt[3], hrt[4], win, final_cti, win_id)
                    if hrt not in final_cti["relation"][ca_sentid(hrt[0], final_cti)]:
                        final_cti["relation"][ca_sentid(hrt[0], final_cti)].append(hrt)
            win_id += 1

        for ner in ner_js:
            if ner["doc_key"] == key:
                final_cti["ner"] = ner["ner"]
        # final_path = hrt_file + '\\' + key + '.json'
        # f = open(final_path, 'w', encoding='utf-8')
        # json.dump(final_cti, f)
        graph_list.append(final_cti)

    return graph_list

def read1(json_file):
    data_ = open(json_file, "r", encoding="utf-8")
    docs = json.load(data_)
    # docs = [json.loads(line) for line in open(json_file)]
    return docs

def sort_nested_list_by_field(lst, field):
    sorted_lst = sorted(lst, key=itemgetter(field))
    return sorted_lst

class Initialize_attack_graph(object):
    def __init__(self, sent_ners, sent_res):
        self.nodes = []
        self.res = []
        self.sum_res = []
        self.ner_site_map = {}
        for sent in sent_ners:
            for ner in sent:
                self.nodes.append(ner)
        self.nodes = sort_nested_list_by_field(self.nodes, 0)
        for sent in sent_res:
            for re in sent:
                self.res.append(re)

    def delete_re(self):
        # 删除一堆实体之间既存在共指又存在其他类型的关系
        final_re = []
        re_map = {}
        new_re_map = {}
        ner_name_map = {}
        for ner in self.nodes:
                # ner_name_map[(ner[0], ner[1])] = ner[4]       # 这里做一个实体位置和名字的map
            ner_name_map[(ner[0], ner[1])] = ner[4]

        for re1 in self.res:
            if (re1[0], re1[1], re1[3], re1[4]) not in re_map and (re1[3], re1[4], re1[0], re1[1]) not in re_map:
                re_map[(re1[0], re1[1], re1[3], re1[4])] = []
                re_map[(re1[0], re1[1], re1[3], re1[4])].append(re1[8])
            elif (re1[0], re1[1], re1[3], re1[4]) in re_map:
                re_map[(re1[0], re1[1], re1[3], re1[4])].append(re1[8])
            elif (re1[3], re1[4], re1[0], re1[1]) in re_map:
                re_map[(re1[3], re1[4], re1[0], re1[1])].append(re1[8])
        # print(ner_name_map)
        print(re_map)
        for key, values in re_map.items():
            if 'CO' in values and len(values) > 1:
                if ner_name_map[(key[0], key[1])] in ['it', 'It', 'them', 'Them', 'this', 'This'] or ner_name_map[(key[2], key[3])] in ['it', 'It', 'them', 'Them', 'this', 'This']:
                    new_re_map[key] = ['CO']
                else:
                    values.remove('CO')
                    new_re_map[key] = values
            else:
                new_re_map[key] = values
        # print(new_re_map)
        for key, values in new_re_map.items():
            for v in values:
                for re2 in self.res:
                    if (re2[0], re2[1], re2[3], re2[4]) == key and re2[8] == v:
                    # if (re2[0], re2[1], re2[2], re2[3]) == key and re2[4] == v:
                        final_re.append(re2)

        self.res = final_re
        self.get_unco_list(self.res)

    def get_unco_list(self, relations):
        for re in relations:
            if re[8] != 'CO':
                self.sum_res.append(re)
        self.sum_res = self.time_sort(self.sum_res)

    def find_cluster(self, co_list):
        sum_cluster_ = []
        cluster = []
        for re_co in co_list:
            cluster.append([(re_co[0], re_co[1], re_co[6]), (re_co[3], re_co[4], re_co[7])])
        scan_over = []
        for co1 in cluster:
            if co1 not in scan_over:
                scan_over.append(co1)
                co_cluster = co1
                for co2 in cluster:
                    if co2 not in scan_over:
                        if co2[0] in co1:
                            co_cluster.append(co2[1])
                            scan_over.append(co2)
                        elif co2[1] in co1:
                            co_cluster.append(co2[0])
                            scan_over.append(co2)
                        else:
                            continue
                sum_cluster_.append(co_cluster)
        return sum_cluster_

    def get_co_map(self):
        co_list_ = []
        co_map = {}
        # ner_site_map = {}
        for i in range(len(self.nodes)):
            self.ner_site_map[(self.nodes[i][0], self.nodes[i][1])] = i
        # print(ner_site_map)
        for re in self.res:
            if re[8] == 'CO':
                co_list_.append(re)
        sum_cluster_ = self.find_cluster(co_list_)
        print(sum_cluster_)
        for co_l in sum_cluster_:
            for co_re in co_l:
                co_map[self.ner_site_map[(co_re[0], co_re[1])]] = self.ner_site_map[(co_l[0][0], co_l[0][1])]

        return co_map

    def time_sort(self, relations):
        n = len(relations)
        for i in range(n - 1):
            for j in range(n - 1 - i):
                sub1 = relations[j][0]
                obj1 = relations[j][3]
                sub2 = relations[j + 1][0]
                obj2 = relations[j + 1][3]
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

    def time_sort_co(self):
        # 得到关系的序号列表,并添加时序元素
        co_ners = []
        sort_re = []
        co_replace_sort_re = []
        for i in range(len(self.sum_res)):
            self.sum_res[i].append(i)
            sort_re.append([self.ner_site_map[(self.sum_res[i][0], self.sum_res[i][1])], self.ner_site_map[(self.sum_res[i][3], self.sum_res[i][4])], relation_map[self.sum_res[i][8]]])
        print(sort_re)
        # new_data['sort_relations'] = sort_re
        co_map = self.get_co_map()
        # new_data['co_map'] = co_map
        # print(new_data['co_map'])
        # 得到共指节点列表
        co_ners = list(co_map.keys())
        print(co_ners)
        # 按照共指增加实体的name
        for n in self.nodes:
            if self.ner_site_map[(n[0], n[1])] in co_map:
                if n[2] == self.nodes[co_map[self.ner_site_map[(n[0], n[1])]]][2]:
                    continue
                elif n[2] in self.nodes[co_map[self.ner_site_map[(n[0], n[1])]]][2]:
                    continue
                elif self.nodes[co_map[self.ner_site_map[(n[0], n[1])]]][2] in n[2]:
                    continue
                else:
                    n[2] += '/' + self.nodes[co_map[self.ner_site_map[(n[0], n[1])]]][2]

        # 按照共指替换sort_re中实体的序号,便于构图
        for re in sort_re:
            if re[0] in co_map:
                co_replace_sort_re.append([co_map[re[0]], re[1], re[2]])
            elif re[1] in co_map:
                co_replace_sort_re.append([re[0], co_map[re[1]], re[2]])
            else:
                co_replace_sort_re.append(re)

        return co_map, sort_re, co_replace_sort_re, co_ners

def merge_co(graph_js):
    f = open(args.data_output_json, 'w', encoding='utf-8')
    for data in graph_js:
        new_data = {}
        new_data['doc_key'] = data['doc_key']
        new_data['sentences'] = data['sentences']

        init_graph = Initialize_attack_graph(data['ner'], data['relation'])
        new_data['original_ner'] = init_graph.nodes
        # init_graph.res
        # 删除不合理的边,同时存在共指和其他关系,得到除去共指关系的排序集合
        init_graph.delete_re()
        # init_graph.sum_res
        # 得到共指的实体替换字典
        init_graph.get_co_map()
        # 得到使用替换字典转换得到的新的实体和关系列表,其中新增了关系的序号列表,用于构图
        new_data['original_relations'] = data['relation']
        new_data['ners'] = init_graph.nodes
        new_data['co_map'], new_data['sort_relations'], new_data['co_replace_sort_relations'], new_data['co_ners'] = init_graph.time_sort_co()
        new_data['relations'] = init_graph.sum_res

        # print(new_data['sort_relations'])
        # print(new_data['co_replace_sort_relations'])
        # print(new_data['relations'])
        # final_path = args.data_output + '\\' + new_data['doc_key'] + '.json'
        # f = open(final_path, 'w', encoding='utf-8')
        json.dump(new_data, f)
        f.write('\n')


if __name__ == '__main__':
    relation_map = {'RD': 1, 'WR': 2, 'EX': 3, 'UK': 4, 'CD': 5, 'FR': 6, 'IJ': 7, 'ST': 8, 'RF': 9, 'CO': 10}
    parser = argparse.ArgumentParser()

    parser.add_argument('--sentence_window', type=int, default=8,
                        help="window for article segmentation")
    parser.add_argument('--data_re', type=str, default=None, required=True,
                        help="relations prediction dataset")
    parser.add_argument('--data_re_result', type=str, default=None, required=True,
                        help="result from relation model")
    parser.add_argument('--data_ner_result', type=str, default=None, required=True,
                        help="result from ner model")
    parser.add_argument('--data_output_json', type=str, default=None, required=True,
                        help="result with complete ner and re")
    # parser.add_argument('--data_output_delete_co', type=str, default=None, required=True,
    #                     help="result without co")

    args = parser.parse_args()
    win_size = args.sentence_window
    # json_file = r"E:\CTI_to_ASG\TTP_test\TTP_test_to_predict_re_result.json"       # 关系预测结果
    # data_file = r"E:\CTI_to_ASG\TTP_test\TTP_test_to_predict_re.json"    # 关系预测数据集
    # ner_file = r"E:\CTI_to_ASG\TTP_test\TTP_test_ner_result.json"           # 实体预测结果
    # hrt_file = r"E:\CTI_to_ASG\TTP_test\3TTP_graph"  # 图信息, output
    predict_json = json.load(open(args.data_re_result, "r", encoding="utf-8"))
    data_json = json.load(open(args.data_re, "r", encoding="utf-8"))
    ner_json = read(args.data_ner_result)

    graph_json = get_graph(predict_json, data_json, ner_json, win_size)
    merge_co_graph_data = merge_co(graph_json)






