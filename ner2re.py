import os
import json
import argparse

def read(json_file):
    docs = [json.loads(line, strict=False) for line in open(json_file)]
    return docs

def calculate_sentence(start, data):
    for j in data:
        sum_len = 0
        for k in range(len(data["sentences"])):
            sum_len += len(data["sentences"][k])
            if start < sum_len:
                return k

def get_document_ner(data_js, w_s):
    new_entity_js = []
    title_list = []
    data_txt_dict = {}
    for i in range(0, len(data_js)):
        temp = data_js[i]["doc_key"].rsplit("-", 1)[0]
        # temp = data_js[i]["doc_key"]
        if temp not in title_list:
            data_txt_dict[temp] = []
            title_list.append(temp)
        data_txt_dict[temp].append(data_js[i])

    for k, v in data_txt_dict.items():
        if "-" in v[0]["doc_key"] and int(v[-1]["doc_key"].rsplit("-", 1)[1]) != len(v) - 1:
            print(str(k) + "find error")
            continue
        else:
            new_data = dict()
            sentences = []
            win_id = 0

            for win in v:
                if win_id == 0:
                    for sent in win["sentences"]:
                        sentences.append(sent)
                else:
                    sentences.append(win["sentences"][w_s - 1])
                win_id += 1
            new_data["doc_key"] = k
            new_data["sentences"] = sentences
            new_data["ner"] = [[] for i in range(len(sentences))]
 
            win_id = 0
            sum_ner = []
            for win in v:
                if win_id == 0:
                    for sent_ner in win["predicted_ner"]:
                        for ner in sent_ner:
                            sum_ner.append(ner)
                else:
                    for sent_ner in win["predicted_ner"]:
                        for ner in sent_ner:
                            new_ner = []
                            ner_h = ner[0]
                            ner_t = ner[1]
                            for i in range(win_id):
                                ner_h += len(sentences[i])
                                ner_t += len(sentences[i])
                            new_ner = [ner_h] + [ner_t] + [ner[2]] + [ner[3]]
                            sum_ner.append(new_ner)
                win_id += 1

            sum_sentence = []
            for sent in sentences:
                for word in sent:
                    sum_sentence.append(word)

            for ner in sum_ner:
                ner_span = ""
                for i in range(ner[0], ner[1] + 1):
                    ner_span += sum_sentence[i]
                    if i != ner[1]:
                        ner_span += ' '
                ner.append(ner_span)

            for ner in sum_ner:
                new_data["ner"][calculate_sentence(ner[0], new_data)].append(ner)

            new_entity_js.append(new_data)

    return new_entity_js

def choose_ner(ner_js, final_path):
    f1 = open(final_path, "w", encoding='utf-8')
    final_sum_js = []
    for data in ner_js:
        new_data = dict()
        new_data["doc_key"] = data["doc_key"]
        new_data["sentences"] = data["sentences"]
        new_ner = [[] for i in range(len(data["ner"]))]
        for sent_id in range(len(data["ner"])):
            ners_list = []
            for ner_id in range(len(data["ner"][sent_id])):
                ner_list = []
                for i in range(data["ner"][sent_id][ner_id][0], data["ner"][sent_id][ner_id][1] + 1):
                    ner_list.append(i)
                ner_list.append(data["ner"][sent_id][ner_id][2])
                ner_list.append(data["ner"][sent_id][ner_id][3])
                ner_list.append(data["ner"][sent_id][ner_id][4])
                ners_list.append(ner_list)
            for x in ners_list:
                final_ner = [x[0]] + [x[-4]] + [x[-3]] + [x[-2]] + [x[-1]]
                for y in ners_list:
                    if list(set(x[:-3]) & set(y[:-3])):
                        if x[-2] < y[-2]:
                            flag = 0
                            break
                        else:
                            flag = 1
                    else:
                        flag = 1
                if flag == 1:
                    new_ner[sent_id].append(final_ner)
                else:
                    continue
        new_data["ner"] = new_ner
        json.dump(new_data, f1)
        f1.write('\n')
        final_sum_js.append(new_data)
    return final_sum_js


def give_window(x, y):
    window_list = []
    if y < x - 1:
        small_list = []
        for x2 in range(0, y+1):
            small_list.append(x2)
        window_list.append(small_list)
    else:
        for i2 in range(0, y - x + 2):
            small_list = []
            for x1 in range(i2, i2 + x):
                small_list.append(x1)
            window_list.append(small_list)
    return window_list

def get_sentence_length(list):
    sentence_len = {}
    for sentence in range(0, len(list)):
        sentence_len[sentence] = len(list[sentence])
    return sentence_len

def calculate_sentence2(x,dic):
    merge = 0
    for i in range(0,x):
        merge += dic[i]
    return merge

def window_qiefen(final_ner_js, w_s):
    win_sum_js = []
    for i in range(0, len(final_ner_js)):
        len_1 = len(final_ner_js[i]["sentences"])
        new_list = give_window(w_s, len_1 - 1)
        i1 = 0
        for list1 in new_list:
            new_json = {}
            new_json["doc_key"] = final_ner_js[i]["doc_key"] + "-" + str(i1)
            new_json["sentences"] = []
            new_json["ner"] = []
            i1 += 1
            sentence_len = get_sentence_length(final_ner_js[i]["sentences"])

            for e in list1:
                small = min(list1)
                before_length = calculate_sentence2(small, sentence_len)
                new_json["sentences"].append(final_ner_js[i]["sentences"][e])
                entity_small = []
                for entity in final_ner_js[i]["ner"][e]:
                    entity1 = [0, 0, "0"]
                    entity1[0] = entity[0]
                    entity1[1] = entity[1]
                    entity1[2] = entity[2]
                    entity1[0] -= before_length
                    entity1[1] -= before_length
                    entity_small.append(entity1)
                new_json["ner"].append(entity_small)

            win_sum_js.append(new_json)

    return win_sum_js

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
    return n_dict

def make_ner2re_data(re_path, w_f_j):
    f = open(re_path, 'w', encoding='utf-8')
    data_list = []
    for js in w_f_j:
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

        for ner in ner_list:
            doc_dict["vertexSet"].append([ner])

        for i in range(len(doc_dict["vertexSet"])):
            for j in range(len(doc_dict["vertexSet"][i])):
                doc_dict["vertexSet"][i][j]["index"] = str(i) + "_" + str(j)

        data_list.append(doc_dict)
    json.dump(data_list, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_ner_result', type=str, default=None, required=True,
                        help="result from ner model")
    parser.add_argument('--data_final_ner_result', type=str, default=None, required=True,
                        help="ner screening results")
    parser.add_argument('--sentence_window', type=int, default=8,
                        help="window for article segmentation")
    parser.add_argument('--data_re', type=str, default=None, required=True,
                        help="relations prediction dataset")

    args = parser.parse_args()

    win_data_js = read(args.data_ner_result)
    win_size = args.sentence_window

    new_js= get_document_ner(win_data_js, win_size)

    final_ner_js = choose_ner(new_js, args.data_final_ner_result)

    win_final_ner_js = window_qiefen(final_ner_js, win_size)

    make_ner2re_data(args.data_re, win_final_ner_js)
