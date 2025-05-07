import json
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity


def read(json_file):
    data_ = open(json_file, "r", encoding="utf-8")
    docs = json.load(data_)
    return docs

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

    print(len(filelist))

    return sum_similarity/len(filelist), similarity_dict


if __name__ == '__main__':

    test_path = r"E:\DARPA测试\DARPA_test"
    GT_path = r"E:\DARPA测试\DARPA_GT"
    Extractor_path = r'E:\DARPA测试\DARPA_Extractor_change_data'
    Attack_path = r'E:\DARPA测试\DARPA_Attackg_change_data'
    result = r"E:\DARPA测试\test_result1.json"
    f = open(result, 'w', encoding='utf-8')
    pathlist = [test_path, GT_path, Extractor_path, Attack_path]
    # pathlist = [test_path, GT_path]
    filelist = os.listdir(test_path)
    node_dict = {}
    edge_dict = {}
    for path in pathlist:
        node_list = []
        edge_list = []
        for li in filelist:
            json_data = read(path + '\\' + li)
            node_list.append(len(json_data['ner']))
            edge_list.append(len(json_data['relations']))
        node_dict[path.split('\\')[-1]] = node_list
        edge_dict[path.split('\\')[-1]] = edge_list

    json.dump(node_dict, f)
    f.write('\n')
    json.dump(edge_dict, f)
    f.write('\n')

    ASG_GT_dict = {}
    ASG_EXT_dict = {}
    ASG_ATG_dict = {}
    # ASG
    ASG_embedding11 = r"E:\DARPA测试\DARPA_pkl\darpa_embedding_test.pkl"
    GT_embedding11 = r"E:\DARPA测试\DARPA_pkl\darpa_embedding_GT.pkl"

    ASG_GT_dict['AVG_ASG_GT'], ASG_GT_dict['ASG_GT_LIST'] = calculate(ASG_embedding11, GT_embedding11)


    # Extractor
    ASG_embedding = r"E:\DARPA测试\DARPA_pkl\darpa_change_embedding_test.pkl"
    GT_embedding = r"E:\DARPA测试\DARPA_pkl\darpa_change_embedding_GT.pkl"
    Extractor_embedding = r"E:\DARPA测试\DARPA_pkl\darpa_change_embedding_Extractor.pkl"

    ASG_EXT_dict['AVG_ASG_GT'], ASG_EXT_dict['ASG_GT_LIST'] = calculate(ASG_embedding, GT_embedding)
    ASG_EXT_dict['AVG_EXT_GT'], ASG_EXT_dict['EXT_GT_LIST'] = calculate(Extractor_embedding, GT_embedding)


    # Attackg
    ASG_embedding1 = r"E:\DARPA测试\DARPA_pkl\DARPA_change_test_embedding_atg.pkl"
    GT_embedding1 = r"E:\DARPA测试\DARPA_pkl\DARPA_change_GT_embedding_atg.pkl"
    Attack_embedding = r"E:\DARPA测试\DARPA_pkl\DARPA_Attackg_change_embedding_data.pkl"

    ASG_ATG_dict['AVG_ASG_GT'], ASG_ATG_dict['ASG_GT_LIST'] = calculate(ASG_embedding1, GT_embedding1)
    ASG_ATG_dict['AVG_ATG_GT'], ASG_ATG_dict['ATG_GT_LIST'] = calculate(Attack_embedding, GT_embedding1)

    json.dump(ASG_GT_dict, f)
    f.write('\n')
    json.dump(ASG_EXT_dict, f)
    f.write('\n')
    json.dump(ASG_ATG_dict, f)