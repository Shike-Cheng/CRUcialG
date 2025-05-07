import torch
from prepro import read_docred
from apex import amp
from transformers import AutoConfig, AutoModel, AutoTokenizer
import os
from utils import set_seed, collate_fn, collate_fn_kd, label_collate_fn, get_label_input_ids
from torch.utils.data import DataLoader
from model import DocREModel_KD
import json
from train import evaluate, report
import argparse
import time
import psutil

def main():

    start_time = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./dataset/docred", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)

    parser.add_argument("--test_file", default="test.json", type=str)

    parser.add_argument("--load_path", default="", type=str)

    parser.add_argument("--load_pretrained", default="", type=str)

    parser.add_argument("--output_name", default="result.json", type=str)

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")

    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--test_batch_size", default=8, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--num_labels", default=4, type=int,
                        help="Max number of labels in prediction.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--classifier_lr", default=1e-4, type=float,
                        help="The initial learning rate for Classifier.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--drop_prob", default=0.0, type=float,
                        help="Negative Sample Discard rate.")
    parser.add_argument("--gamma_pos", default=1.0, type=float,
                        help="Gamma for positive class")
    parser.add_argument("--gamma_neg", default=1.0, type=float,
                        help="Gamma for negative class")
    parser.add_argument("--drop_FP", default=0.0, type=float,
                        help="Potential FP Discard rate.")
    parser.add_argument("--drop_FN", default=0.0, type=float,
                        help="Potential FN Discard rate.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--start_steps", default=-1, type=int)
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=97,
                        help="Number of relation types in dataset.")

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device


    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )
    suffix = '.{}.pt'.format(args.model_name_or_path)
    read = read_docred

    test_file = os.path.join(args.data_dir, args.test_file)
    test_features = read(test_file, tokenizer, max_seq_length=args.max_seq_length)
    torch.save(test_features, os.path.join(args.data_dir, args.test_file + suffix))
    print('Created and saved new test features')

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type
    label_features = get_label_input_ids(args.data_dir, tokenizer)

    label_loader = DataLoader(label_features, batch_size=36, shuffle=False, collate_fn=label_collate_fn,
                              drop_last=False)

    set_seed(args)
    model = DocREModel_KD(args, config, model, num_labels=args.num_labels)
    model.to(0)

    model = amp.initialize(model, opt_level="O1", verbosity=0)
    model.load_state_dict(torch.load(args.load_path), strict=False)
    # test_score, test_output = evaluate(args, model, test_features, label_loader, tag="test")
    #print(test_output)
    pred, logits = report(args, model, test_features, label_loader)
    output_dir = os.path.join(args.data_dir, args.output_name)
    with open(output_dir, "w") as fh:
        json.dump(pred, fh)
    end_time = time.time()
    execution_time = end_time - start_time
    cpu_usage = psutil.cpu_percent(interval=1)
if __name__ == "__main__":
    main()
