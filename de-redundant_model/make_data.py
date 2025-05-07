from nltk.tokenize import word_tokenize, sent_tokenize
import json
import os


path = r'E:\A-post-graduate\AI样本生成\校企CTI报告\filter-校企txt'
txtList = os.listdir(path)

for li in txtList:
    data_json = {}
    data_json["cluster"] = []
    data_json["sentences"] = []
    data_json["ner"] = []
    data_json["relations"] = []
    sentences = []
    with open(path + '\\' + li, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for l in lines:
        if l != None and l != '\n':
            l = l.replace('\n', '')
            sentences.append(l)
    # print(sentences)
    words = [word_tokenize(s) for s in sentences]
    # print(words)
    count = 0
    for w in words:
        for i in range(len(w)):
            w[i] = w[i] + " : " + str(count)
            count = count + 1
    # print(words)
    data_json["sentences"] = words
    print(data_json)
    f2 = open(r'E:\A-post-graduate\AI样本生成\校企CTI报告\filter-校企txt\校企.json', 'a', encoding='utf-8')
    json.dump(data_json, f2)
    f2.write('\n')
