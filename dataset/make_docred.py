import json

def read(json_file):
    docs = [json.loads(line) for line in open(json_file)]
    return docs

def ca_sentid(start, js):
    sum_len = 0
    for i in range(len(js["sentences"])):
        sum_len += len(js["sentences"][i])
        if start < sum_len:
            return i

def get_ner(ner, sent):
    name = ""
    for i in range(ner[0], ner[1] + 1):
        if i < ner[1]:
            name += (sent[i] + ' ')
        else:
            name += sent[i]
    return name

def get_pos(ner, js, id):
    sum = 0
    for i in range(len(js["sentences"])):
        if i < id:
            sum += len(js["sentences"][i])
        else:
            break
    start = ner[0] - sum
    end = ner[1] - sum + 1
    return [start, end]


def change_ner(ner, js, sent):
    n_dict = dict()
    n_dict["type"] = ner[2]
    n_dict["sent_id"] = ca_sentid(ner[0], js)
    n_dict["global_pos"] = [ner[0], ner[0]]
    n_dict["name"] = get_ner(ner, sent)
    n_dict["pos"] = get_pos(ner, js, n_dict["sent_id"])
    # print(n_dict)
    return n_dict

if __name__ == "__main__":

    json_path = ""    # sciERC
    data_path = ""    # DocRED
    f = open(data_path, 'w', encoding='utf-8')
    data_js = read(json_path)

    data_list = []
    for js in data_js:
        doc_dict = dict()
        doc_dict["title"] = js["doc_key"]
        doc_dict["sents"] = js["sentences"]
        doc_dict["vertexSet"] = []
        doc_dict["labels"] = []
        ner_list = []
        sent = []
        for s in js["sentences"]:
            for w in s:
                sent.append(w)
        for s_ner in js["ner"]:
            for ner in s_ner:
                ner_list.append(change_ner(ner, js, sent))
        print(ner_list)
        for ner in ner_list:
            doc_dict["vertexSet"].append([ner])
        print(doc_dict["vertexSet"])
        '''global_pos = []
        for n_co in js["clusters"]:
            clusters = []
            for co in n_co:
                for ner in ner_list:
                    if co[0] == ner["global_pos"][0]:
                        # print(ner)
                        global_pos.append(ner["global_pos"][0])
                        clusters.append(ner)
            doc_dict["vertexSet"].append(clusters)
        for ner in ner_list:
            sy_n = []
            if ner["global_pos"][0] not in global_pos:
                sy_n.append(ner)
                doc_dict["vertexSet"].append(sy_n)
        '''

        for i in range(len(doc_dict["vertexSet"])):
            for j in range(len(doc_dict["vertexSet"][i])):
                doc_dict["vertexSet"][i][j]["index"] = str(i) + "_" + str(j)
        print(doc_dict["vertexSet"])

        for n_re in js["relations"]:
            for re in n_re:
                relation = dict()
                relation["r"] = re[4]
                relation["evidence"] = []
                full_evidence = []
                for i in range(len(doc_dict["vertexSet"])):
                    for j in range(len(doc_dict["vertexSet"][i])):
                        if doc_dict["vertexSet"][i][j]["global_pos"][0] == re[0]:
                            relation["h"] = i
                            relation["evidence"].append(doc_dict["vertexSet"][i][j]["sent_id"])
                        if doc_dict["vertexSet"][i][j]["global_pos"][0] == re[2]:
                            relation["t"] = i
                            relation["evidence"].append(doc_dict["vertexSet"][i][j]["sent_id"])
                for i in range(relation["evidence"][0], relation["evidence"][1] + 1):
                    full_evidence.append(i)
                relation["evidence"] = full_evidence
                if len(relation) == 4:
                    doc_dict["labels"].append(relation)
        print(doc_dict["labels"])
        print(doc_dict)
        data_list.append(doc_dict)
    json.dump(data_list, f)


