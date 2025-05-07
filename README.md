# GRUcialG

## Configurations and Required Packages
| 1    | 1    |
| ---- | ---- |
| CPU       | Intel i9-13900K CPU (24 cores and 3.00 GHz)    |
| Memory    | 64G                                            |
| GPU       | NVIDIA GeForce RTX 4090                        |

- python=3.7.16
- CUDA=11.0
- torch==1.7.1+cu110
- numpy==1.21.6
- allennlp==1.2.0
More information on the version of the environment configuration can be found in the GRUcialG.yaml file.

## Dataset and Model
The 10,607 reports we collected can be found on [Google Cloud Drive](https://drive.google.com/file/d/1xYeR9TbEQEgEAwrZHRXQ24ssntse71TJ/view?usp=sharing). In addition, we have also published a dataset of 110 manually labelled CTI reports, which you can find at ‘dataset/cti’.



## Training and Testing
### NER Model

#### train

```shell
python ner/run_entity.py \
	--do_train --do_eval --eval_test \
	--learning_rate=1e-5 \
	--task_learning_rate=5e-4 \
	--train_batch_size=16 \
	--context_window 300 \
	--task CTI \
	--data_dir ner/dataset \
	--model ner/allenai/enscibert_scivocab_uncased \
	--output_dir ner/dataset/output-en-test
```

#### predict_ners

```shell
python ner/predict_ner.py --task CTI --data_txt_dir ner/test_CTI --sentence_window 8 --data_dir ner/test_result/test_ner.json --output_dir ner/dataset/output-en-test --model ner/allenai/enscibert_scivocab_uncased --context_window 300 --test_pred_filename ner/test_result/test_ner_result.json
```

####  ner2re

```shell
python ner2re.py --data_ner_result ner/test_result/test_ner_result.json --data_final_ner_result test_result/test_ner_result.json --sentence_window 8 --data_re relation/test_result/test_to_predict_re.json
```

### relation

#### train

```shell
python relation/train.py \
	--data_dir relation/dataset/ 
	--transformer_type scibert \
	--model_name_or_path enscibert_scivocab_uncased \
	--save_path relation/checkpoints/scibert-0907ctimodel.pt \
	--save_last relation/checkpoints/scibert-0907ctimodel-last.pt \
	--train_file train_annotated.json \
	--dev_file dev.json \
	--test_file test.json \
	--train_batch_size 1 \
	--test_batch_size 1 \
	--gradient_accumulation_steps 1 \
	--num_labels 4 \
	--learning_rate 3e-5 \
	--classifier_lr 1e-4 \
	--max_grad_norm 1.0 \
	--drop_prob 0.0 \
	--warmup_ratio 0.06 \
	--num_train_epochs 50.0 \
	--seed 9900 \
	--num_class 11
```

#### predict_re

```shell
python relation/predict_re.py --data_dir relation/test_result/ --transformer_type scibert --model_name_or_path enscibert_scivocab_uncased --load_path relation/checkpoints/scibert-0907ctimodel.pt --test_file test_to_predict_re.json  --output_name test_to_predict_re_result.json --test_batch_size 1 --gradient_accumulation_steps 1 --num_labels 4 --learning_rate 3e-5 --max_grad_norm 1.0 --drop_prob 0.0 --warmup_ratio 0.06 --seed 9900 --num_class 11

```

### gennerate_graph

```shell
python graph_gennerator.py --sentence_window 8 --data_re relation/test_result/test_to_predict_re.json --data_re_result relation/test_result/test_to_predict_re_result.json --data_ner_result test_result/test_ner_result.json --data_output_json test_result/test_result.json
```

### graph2repair 

#### train

```shell
python graph_repair/trans_CTI.py
```

```shell
python -u -W ignore graph_repair/train_CTI.py \
--path graph_repair/data_preprocessed/99_train \
--train --num_workers 4 \
--batch_size 4 \
--lr 0.001 \
--epochs 10 \
--shuffle --deq_coeff 0.9 \
--save --name 99_train_epoch10_1gpu \
--num_flow_layer 12 \
--nhid 128 \
--nout 128 \
--gcn_layer 3 \
--is_bn --divide_loss --st_type exp \
--seed 2019 \
--all_save_prefix ./
```

#### predict_re

```shell
python graph2repair.py --graph_json test_result/test_result.json --graph_to_repair_json test_result/test_result_to_repair.json --graph_to_repair_txt test_result/test_result_to_repair.txt
```

```shell
python graph_repair/predict_relation_stage.py --path test_result/test_result_to_repair.txt --checkpoint_path graph_repair/save_pretrain/exp_ASG_99_train_epoch10_1gpu/checkpoint9 --predict_data_path test_result/test_result_to_repair.json --predict_result_path test_result/test_result_to_repair_result.json --predict_result_txt_path test_result/test_result_to_repair_result.txt --num_flow_layer 12 --nhid 128 --nout 128 --is_bn --gcn_layer 3 --st_type exp --divide_loss
```

#### predict_re_node

```shell
python graph_repair/predict_node_edges.py \
--path graph_repair/test_result/TTP_to_repair_re_result.txt \
--checkpoint_path graph_repair/save_pretrain/exp_ASG_99_train_epoch10_1gpu/checkpoint9  \
--predict_data_path graph_repair/test_result/TTP_to_repair_re_result.json \
--predict_result_path graph_repair/test_result/TTP_to_repair_final_result.json \
--num_flow_layer 12 \
--nhid 128 \
--nout 128 \
--is_bn --gcn_layer 3 \
--st_type exp --divide_loss
```

#### ASG_Generation
```shell
python ASG_gengerator.py --graph_generator_json test_result/test_result_to_repair_result.json --asg_reconstruction_json test_result/test_ASG_result.json --asg_reconstruction_graph test_result/
```
