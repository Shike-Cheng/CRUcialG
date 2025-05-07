from entity.utils import convert_dataset_to_samples, batchify, NpEncoder
from entity.models import EntityModel
import argparse
from shared.const import task_ner_labels, get_labelmap
import os
from shared.data_structures import Dataset
import json
from entity.utils import ioc_reg
from nltk.tokenize import word_tokenize
import time
import psutil

def init_sentences(txt_path, txt_name):

    sentences = []
    with open(txt_path + '\\' + txt_name, 'r', encoding='utf-8',errors='replace') as f:
        lines = f.readlines()
    for l in lines:
        if l != None and l != '\n':
            if len(word_tokenize(l)) >= 3:
                l = l.replace('\n', '')
                sentences.append(l)
    words = []
    for s in sentences:
        final_word = []
        for w in word_tokenize(s):
            if len(w) > 50:
                final_word.append(w[:25])
                final_word.append(w[25:])
            else:
                final_word.append(w)
        # print(word_tokenize(s))
        words.append(final_word)
    # words = [word_tokenize(s) for s in sentences]
    return words


def give_window(x, y):
    window_list = []
    if y < x - 1:
        small_list = []
        for x2 in range(0, y+1):
            small_list.append(x2)
        window_list.append(small_list)
        # print("window_list", window_list)
    else:
        for i2 in range(0, y - x + 2):
            small_list = []
            for x1 in range(i2, i2 + x):
                small_list.append(x1)
            window_list.append(small_list)
    return window_list

def get_json_data(txt_path, win_size, json_path):
    fileList = os.listdir(txt_path)
    f = open(json_path, "w", encoding='utf-8')
    for li in fileList:
        # print(li)
        sum_sentences = init_sentences(txt_path, li)
        print(li)
        len_1 = len(sum_sentences)
        if len_1 == 0:
            continue
        new_list = give_window(win_size, len_1 - 1)
        i1 = 0
        for list1 in new_list:
            new_json = {}
            new_json["doc_key"] = li[0:-4] + "-" + str(i1)
            new_json["sentences"] = []
            i1 += 1
            for e in list1:
                new_json["sentences"].append(sum_sentences[e])
            new_json["ner"] = [[] for i in range(len(new_json["sentences"]))]
            new_json["relations"] = [[] for i in range(len(new_json["sentences"]))]
            json.dump(new_json, f)
            f.write('\n')
    f.close()


def reg_neridentity(batche, pred_ner):
    new_pred_ner = [[] for i in range(len(pred_ner))]
    sample_id = 0
    for sample, preds in zip(batche, pred_ner):
        a = len(preds)
        b = len(sample['spans_label'])
        span_id = 0
        new_preds = []
        for span in sample["spans"]:
            if preds[span_id] == 0:
                token_span = ""
                for i in range(span[0], span[1]+1):
                    token_span += sample["tokens"][i]
                flag, label = ioc_reg(token_span)
                if flag:
                    new_preds.append(ner_label2id[label])
                else:
                    new_preds.append(preds[span_id])
            else:
                new_preds.append(preds[span_id])
            span_id += 1
        c = len(new_preds)
        new_pred_ner[sample_id] = new_preds
        sample_id += 1
    return new_pred_ner

def output_ner_predictions(model, batches, dataset, output_file):
    """
    Save the prediction as a json file
    """
    ner_result = {}
    ner_prob = {}
    span_hidden_table = {}
    tot_pred_ett = 0
    for i in range(len(batches)):
        # print(batches[i])
        output_dict = model.run_batch(batches[i], training=False)
        pred_ner = output_dict['pred_ner']
        pred_prob = output_dict['ner_probs']
        pred_ner = reg_neridentity(batches[i], pred_ner)
        for sample, preds, probs in zip(batches[i], pred_ner, pred_prob):
            off = sample['sent_start_in_doc'] - sample['sent_start']
            k = sample['doc_key'] + '-' + str(sample['sentence_ix'])
            ner_result[k] = []
            ner_prob[k] = []
            for span, pred, prob in zip(sample['spans'], preds, probs):
                span_id = '%s::%d::(%d,%d)' % (sample['doc_key'], sample['sentence_ix'], span[0] + off, span[1] + off)
                if pred == 0:
                    continue
                ner_result[k].append([span[0] + off, span[1] + off, ner_id2label[pred], prob[pred]])
                ner_prob[k].append(prob[pred])
            tot_pred_ett += len(ner_result[k])

    #print(ner_result)
    js = dataset.js
    for i, doc in enumerate(js):
        doc["predicted_ner"] = []
        doc["predicted_prob"] = []
        doc["predicted_relations"] = []
        for j in range(len(doc["sentences"])):
            k = doc['doc_key'] + '-' + str(j)
            if k in ner_result:
                doc["predicted_ner"].append(ner_result[k])
                doc["predicted_prob"].append(ner_prob[k])
            else:
                doc["predicted_ner"].append([])

            doc["predicted_relations"].append([])

        js[i] = doc

    with open(output_file, 'w') as f:
        f.write('\n'.join(json.dumps(doc, cls=NpEncoder) for doc in js))


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default=None, required=True, choices=['ace04', 'ace05', 'scierc', 'CTI', 'easyCTI'])
    parser.add_argument('--sentence_window', type=int, default=8,
                        help="window for article segmentation")
    parser.add_argument('--data_txt_dir', type=str, default=None, required=True,
                        help="path to the CTI txt")
    parser.add_argument('--data_dir', type=str, default=None, required=True,
                        help="path to the preprocessed dataset")
    parser.add_argument('--output_dir', type=str, default='entity_output',
                        help="output directory of the entity model")
    parser.add_argument('--model', type=str, default='bert-base-uncased',
                        help="the base model name (a huggingface model)")
    parser.add_argument('--bert_model_dir', type=str, default=None, help="the base model directory")
    parser.add_argument('--test_pred_filename', type=str, default="ent_pred_test1.json",
                        help="the prediction filename for the test set")
    parser.add_argument('--use_albert', action='store_true',
                        help="whether to use ALBERT model")
    parser.add_argument('--context_window', type=int, required=True, default=None,
                        help="the context window size W for the entity model")
    parser.add_argument('--max_span_length', type=int, default=8,
                        help="spans w/ length up to max_span_length are considered as candidates")
    parser.add_argument('--eval_batch_size', type=int, default=32,
                        help="batch size during inference")

    args = parser.parse_args()
    ner_label2id, ner_id2label = get_labelmap(task_ner_labels[args.task])

    num_ner_labels = len(task_ner_labels[args.task]) + 1

    args.bert_model_dir = args.output_dir
    # args.test_data = os.path.join(args.data_dir, 'test.json')

    model = EntityModel(args, num_ner_labels=num_ner_labels)

    get_json_data(args.data_txt_dir, args.sentence_window, args.data_dir)
    test_data = Dataset(args.data_dir)

    # prediction_file = os.path.join(args.output_dir, args.test_pred_filename)

    test_samples, test_ner = convert_dataset_to_samples(test_data, args.max_span_length, ner_label2id=ner_label2id,
                                                        context_window=args.context_window)
    test_batches = batchify(test_samples, args.eval_batch_size)

    output_ner_predictions(model, test_batches, test_data, output_file=args.test_pred_filename)

    end_time = time.time()
    execution_time = end_time - start_time
    cpu_usage = psutil.cpu_percent(interval=1)
