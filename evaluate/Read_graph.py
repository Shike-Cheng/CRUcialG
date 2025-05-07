import os
import networkx as nx
import pickle
import json
import argparse


# def read(json_file):
#     data_ = open(json_file, "r", encoding="utf-8")
#     docs = json.load(data_)
#     return docs

def read(json_file):
    docs = [json.loads(line) for line in open(json_file)]
    return docs

no_re_map = {1: 'RD', 2: 'WR', 3: 'EX', 4: 'UK', 5: 'CD', 6: 'FR', 7: 'IJ', 8: 'ST', 9: 'RF', 10: 'CO'}
types_map = {'cru_vs_gt': ['test_cru', 'test_cru_gt'], 'cru_vs_extractor': ['test_cru', 'test_extractor', 'test_cru_gt'], 'cru_vs_attackg': ['test_cru', 'test_attackg', 'test_cru_gt']}

def read_graph(type_data):
    path = 'E:\\CRUcialG\\evaluate\\graph_data\\' + args.scene + '\\' + type_data + '.json'
    n_data = {}
    jsonlist = read(path)
    graph_id = 0
    snapshotSeq = {}  # snapshot sequence
    for json_data in jsonlist:
        graphName = "ASG"
        G = nx.MultiGraph(name=graphName, data=True, align='vertical')  # undirected
        # json_data = read(path + '\\' + li)
        ners = json_data['ners']
        res = json_data['relations']

        n = len(ners)
        r_id = 0
        n_data[json_data['doc_key']] = n
        for r in res:
            # sub_id, sub_name, sub_type, obj_id, obj_name, obj_type = r[0], ners[r[0]][3], ners[r[0]][2], r[1], ners[r[1]][3], ners[r[1]][2]
            sub_id, sub_name, sub_type, obj_id, obj_name, obj_type = r[0], ners[r[0]], ners[r[0]], r[1], ners[r[1]], ners[r[1]]
            # sub_id, sub_name, sub_type, obj_id, obj_name, obj_type = r[0], ners[r[0]][2], ners[r[0]][2], r[1], ners[r[1]][2], ners[r[1]][2]
            # event_type = no_re_map[r[2]]
            event_type = 'edge'
            # event_type = r[2]

            if not G.has_node(sub_id):
                G.add_node(sub_id, name=sub_name, type=sub_type, time=r_id)
            if not G.has_node(obj_id):
                G.add_node(obj_id, name=obj_name, type=obj_type, time=r_id)

            if not G.has_edge(sub_id, obj_id, event_type):
                G.add_edge(sub_id, obj_id, time=r_id, key=event_type)
            else:
                G[sub_id][obj_id][event_type]['time'] = r_id  # 仅维护同类型事件的最新的一条

            G.nodes()[sub_id]['time'], G.nodes()[obj_id]['time'] = r_id, r_id  # 更新节点时间

        snapshotSeq[graph_id] = G.copy()
        graph_id += 1

    pkl_path = "E:\\CRUcialG\\evaluate\\graph_similartiy\\" + args.scene + '\\' + type_data + '.pkl'

    # specificSnapshotFile = r"D:\APT攻击知识抽取\消融实验\new_pkl\22_change_test_atg.pkl"

    with open(pkl_path, 'wb') as fs:
        pickle.dump(snapshotSeq, fs)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, default='cru_vs_gt')
    args = parser.parse_args()

    assert args.scene in ['cru_vs_gt', 'cru_vs_extractor', 'cru_vs_attackg']

    for t in types_map[args.scene]:
        read_graph(t)
