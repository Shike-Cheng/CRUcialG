# !/usr/bin/env python
# -*-coding:utf-8 -*-

from time import time
import argparse
import numpy as np
import math
import os
import sys
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils import *

from model_CTI import GraphFlowModel
from dataloader import PretrainZinkDataset
import environment as env
import warnings
from logmaking import make_print_to_file
warnings.filterwarnings("ignore", category=UserWarning)


def save_model(model, optimizer, args, var_list, epoch=None):
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as f:
        json.dump(argparse_dict, f)

    epoch = str(epoch) if epoch is not None else ''
    latest_save_path = os.path.join(args.save_path, 'checkpoint')
    final_save_path = os.path.join(args.save_path, 'checkpoint%s' % epoch)
    torch.save({
        **var_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        final_save_path
    )

    # save twice to maintain a latest checkpoint
    torch.save({
        **var_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        latest_save_path
    )


def restore_model(model, args, epoch=None):
    if epoch is None:
        restore_path = os.path.join(args.save_path, 'checkpoint')
        print('restore from the latest checkpoint')
    else:
        restore_path = os.path.join(args.save_path, 'checkpoint%s' % str(epoch))
        print('restore from checkpoint%s' % str(epoch))

    checkpoint = torch.load(restore_path)
    model.load_state_dict(checkpoint['model_state_dict'])


def read_molecules(path):
    print('reading data from %s' % path)
    node_features = np.load(path + '_np_node.npy')
    adj_features = np.load(path + '_np_adj.npy')
    mol_sizes = np.load(path + '_np_mol.npy')

    f = open(path + '_config.txt', 'r')
    data_config = eval(f.read())
    f.close()

    fp = open(path + '_graph.txt')
    all_smiles = []
    for smiles in fp:
        all_smiles.append(smiles.strip())
    fp.close()

    return node_features, adj_features, mol_sizes, data_config, all_smiles


class Trainer(object):
    def __init__(self, dataloader, data_config, args, all_train_smiles=None):
        self.dataloader = dataloader
        self.data_config = data_config
        self.args = args
        self.all_train_smiles = all_train_smiles

        self.max_size = self.data_config['max_size']
        self.node_dim = self.data_config['node_dim'] - 1  # exclude padding dim.
        self.bond_dim = self.data_config['bond_dim']

        self._model = GraphFlowModel(self.max_size, self.node_dim, self.bond_dim, self.args.edge_unroll, self.args)
        self._optimizer = optim.Adam(filter(lambda p: p.requires_grad, self._model.parameters()),
                                     lr=self.args.lr, weight_decay=self.args.weight_decay)

        self.best_loss = 100.0
        self.start_epoch = 0
        if self.args.cuda:
            self._model = self._model.cuda()

    def initialize_from_checkpoint(self, gen=False):
        checkpoint = torch.load(self.args.init_checkpoint)
        self._model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if not gen:
            self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.best_loss = checkpoint['best_loss']
            self.start_epoch = checkpoint['cur_epoch'] + 1
        print('initialize from %s Done!' % self.args.init_checkpoint)

    def generate_molecule(self, num=10, epoch=None, out_path=None, mute=False,
                          save_good_mol=False):
        self._model.eval()
        all_smiles = []
        pure_valids = []
        appear_in_train = 0.
        start_t = time()
        cnt_mol = 0
        cnt_gen = 0

        while cnt_mol < num:
            smiles, no_resample, num_atoms = self._model.generate(self.args.temperature, mute=mute,
                                                                  max_atoms=self.args.max_atoms, cnt=cnt_gen)
            cnt_gen += 1

            if num_atoms < self.args.min_atoms:
                x1 = 1
                # print('#atoms of generated molecule less than %d, discarded!' % self.args.min_atoms)
            else:
                cnt_mol += 1

                if cnt_mol % 100 == 0:
                    print('cur cnt mol: %d' % cnt_mol)

                all_smiles.append(smiles)
                pure_valids.append(no_resample)
                if self.all_train_smiles is not None and smiles in self.all_train_smiles:
                    appear_in_train += 1.0

        assert cnt_mol == num, 'number of generated molecules does not equal num'

        unique_smiles = list(set(all_smiles))
        unique_rate = len(unique_smiles) / num
        pure_valid_rate = sum(pure_valids) / num
        novelty = 1. - (appear_in_train / num)

        if epoch is None:
            print(
                'Time of generating (%d/%d) molecules(#atoms>=%d): %.5f | unique rate: %.5f | valid rate: %.5f | novelty: %.5f' % (
                num,
                cnt_gen, self.args.min_atoms, time() - start_t, unique_rate, pure_valid_rate, novelty))
        else:
            print(
                'Time of generating (%d/%d) molecules(#atoms>=%d): %.5f at epoch :%d | unique rate: %.5f | valid rate: %.5f | novelty: %.5f' % (
                num,
                cnt_gen, self.args.min_atoms, time() - start_t, epoch, unique_rate, pure_valid_rate, novelty))

        if out_path is not None and self.args.save:
            fp = open(out_path, 'w')
            cnt = 0
            for i in range(len(all_smiles)):
                fp.write(all_smiles[i] + '\n')
                cnt += 1
            fp.close()
            print('writing %d smiles into %s done!' % (cnt, out_path))
        return (unique_rate, pure_valid_rate, novelty)

    def fit(self, mol_out_dir=None):
        t_total = time()
        total_loss = []
        best_loss = self.best_loss
        start_epoch = self.start_epoch
        all_unique_rate = []
        all_valid_rate = []
        all_novelty_rate = []
        print('start fitting.')
        for epoch in range(self.args.epochs):
            epoch_loss = self.train_epoch(epoch + start_epoch)
            total_loss.append(epoch_loss)

            mol_save_path = os.path.join(mol_out_dir,
                                         'epoch%d.txt' % (epoch + start_epoch)) if mol_out_dir is not None else None
            print(mol_save_path)
            cur_unique, cur_valid, cur_novelty = self.generate_molecule(num=10, epoch=epoch + start_epoch,
                                                                        out_path=mol_save_path, mute=True)

            all_unique_rate.append(cur_unique)
            all_valid_rate.append(cur_valid)
            all_novelty_rate.append(cur_novelty)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                if self.args.save:
                    var_list = {'cur_epoch': epoch + start_epoch,
                                'best_loss': best_loss,
                                }
                    save_model(self._model, self._optimizer, self.args, var_list, epoch=epoch + start_epoch)

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time() - t_total))
        if mol_out_dir is not None and self.args.save:
            all_unique_rate = np.array(all_unique_rate)
            all_valid_rate = np.array(all_valid_rate)
            all_novelty_rate = np.array(all_novelty_rate)
            print('saving unique and valid array...')
            np.save(os.path.join(mol_out_dir, 'unique'), all_unique_rate)
            np.save(os.path.join(mol_out_dir, 'valid'), all_valid_rate)
            np.save(os.path.join(mol_out_dir, 'novelty'), all_novelty_rate)

    def train_epoch(self, epoch_cnt):
        t_start = time()
        batch_losses = []
        self._model.train()
        batch_cnt = 0
        epoch_example = 0
        for i_batch, batch_data in enumerate(self.dataloader):
            batch_time_s = time()

            self._optimizer.zero_grad()

            batch_cnt += 1
            inp_node_features = batch_data['node']  # (B, N, node_dim)
            inp_adj_features = batch_data['adj']  # (B, 4, N, N)
            if self.args.cuda:
                inp_node_features = inp_node_features.cuda()
                inp_adj_features = inp_adj_features.cuda()
            if self.args.deq_type == 'random':
                # print("inp_node_features")
                # print(inp_node_features.size())
                # print("inp_adj_features")
                # print(inp_adj_features.size())
                out_z, out_logdet, ln_var = self._model(inp_node_features, inp_adj_features)
                loss = self._model.log_prob(out_z, out_logdet)

                # TODO: add mask for different molecule size, i.e. do not model the distribution over padding nodes.

            elif self.args.deq_type == 'variational':
                out_z, out_logdet, out_deq_logp, out_deq_logdet = self._model(inp_node_features, inp_adj_features)
                ll_node, ll_edge, ll_deq_node, ll_deq_edge = self._model.log_prob(out_z, out_logdet, out_deq_logp,
                                                                                  out_deq_logdet)
                loss = -1. * ((ll_node - ll_deq_node) + (ll_edge - ll_deq_edge))
            else:
                raise ValueError('unsupported dequantization method: (%s)' % self.deq_type)

            loss.backward()
            self._optimizer.step()

            batch_losses.append(loss.item())

            if batch_cnt % self.args.show_loss_step == 0 or (epoch_cnt == 0 and batch_cnt <= 100):
                # print(out_z[0][0])
                epoch_example = [out_z[0][0], out_z[1][0]]
                print('epoch: %d | step: %d | time: %.5f | loss: %.5f | ln_var: %.5f' % (
                epoch_cnt, batch_cnt, time() - batch_time_s, batch_losses[-1], ln_var))

        epoch_loss = sum(batch_losses) / len(batch_losses)
        print(epoch_example)
        print('Epoch: {: d}, loss {:5.5f}, epoch time {:.5f}'.format(epoch_cnt, epoch_loss, time() - t_start))
        return epoch_loss


if __name__ == '__main__':
    make_print_to_file(path='./logs')
    parser = argparse.ArgumentParser(description='GraphFlow model')

    # ******data args******
    parser.add_argument('--dataset', type=str, default='ASG', help='dataset')
    parser.add_argument('--path', type=str, help='path of dataset', required=True)

    parser.add_argument('--batch_size', type=int, default=32, help='batch_size.')
    parser.add_argument('--edge_unroll', type=int, default=10, help='max edge to model for each node in bfs order.')
    parser.add_argument('--shuffle', action='store_true', default=False, help='shuffle data for each epoch')
    parser.add_argument('--num_workers', type=int, default=10, help='num works to generate data.')

    # ******model args******
    parser.add_argument('--name', type=str, default='base',
                        help='model name, crucial for test and checkpoint initialization')
    parser.add_argument('--deq_type', type=str, default='random', help='dequantization methods.')
    parser.add_argument('--deq_coeff', type=float, default=0.9,
                        help='dequantization coefficient.(only for deq_type random)')
    parser.add_argument('--num_flow_layer', type=int, default=6,
                        help='num of affine transformation layer in each timestep')
    parser.add_argument('--gcn_layer', type=int, default=3, help='num of rgcn layers')
    # TODO: Disentangle num of hidden units for gcn layer, st net layer.
    parser.add_argument('--nhid', type=int, default=128, help='num of hidden units of gcn')
    parser.add_argument('--nout', type=int, default=128, help='num of out units of gcn')

    parser.add_argument('--st_type', type=str, default='sigmoid',
                        help='architecture of st net, choice: [sigmoid, exp, softplus, spine]')

    # ******for sigmoid st net only ******
    parser.add_argument('--sigmoid_shift', type=float, default=2.0, help='sigmoid shift on s.')

    # ******for exp st net only ******

    # ******for softplus st net only ******

    # ******optimization args******
    parser.add_argument('--all_save_prefix', type=str, default='./', help='path of save prefix')
    parser.add_argument('--train', action='store_true', default=False, help='do training.')
    parser.add_argument('--save', action='store_true', default=False, help='Save model.')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--learn_prior', action='store_true', default=False, help='learn log-var of gaussian prior.')

    parser.add_argument('--seed', type=int, default=2019, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--is_bn', action='store_true', default=False, help='batch norm on node embeddings.')
    parser.add_argument('--is_bn_before', action='store_true', default=False,
                        help='batch norm on node embeddings on st-net input.')
    parser.add_argument('--scale_weight_norm', action='store_true', default=False,
                        help='apply weight norm on scale factor.')
    parser.add_argument('--divide_loss', action='store_true', default=False, help='divide loss by length of latent.')
    parser.add_argument('--init_checkpoint', type=str, default=None,
                        help='initialize from a checkpoint, if None, do not restore')

    parser.add_argument('--show_loss_step', type=int, default=100)

    # ******generation args******
    parser.add_argument('--temperature', type=float, default=0.75, help='temperature for normal distribution')
    parser.add_argument('--min_atoms', type=int, default=10,
                        help='minimum #atoms of generated mol, otherwise the mol is simply discarded')
    parser.add_argument('--max_atoms', type=int, default=300, help='maximum #atoms of generated mol')
    parser.add_argument('--gen_num', type=int, default=10,
                        help='num of molecules to generate on each call to train.generate')
    parser.add_argument('--gen', action='store_true', default=False, help='generate')
    parser.add_argument('--gen_out_path', type=str, help='output path for generated mol')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.save:
        checkpoint_dir = args.all_save_prefix + 'save_pretrain/%s_%s_%s' % (
        args.st_type, args.dataset, args.name)
        args.save_path = checkpoint_dir

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
    set_seed(args.seed, args.cuda)

    assert (args.train and not args.gen) or (args.gen and not args.train), 'please specify either train or gen mode'
    node_features, adj_features, mol_sizes, data_config, all_smiles = read_molecules(args.path)
    train_dataloader = DataLoader(PretrainZinkDataset(node_features, adj_features, mol_sizes),
                                  batch_size=args.batch_size,
                                  shuffle=args.shuffle,
                                  num_workers=args.num_workers)

    trainer = Trainer(train_dataloader, data_config, args, all_train_smiles=all_smiles)
    if args.init_checkpoint is not None:
        trainer.initialize_from_checkpoint(gen=args.gen)

    if args.train:
        if args.save:
            mol_out_dir = os.path.join(checkpoint_dir, 'asg')

            if not os.path.exists(mol_out_dir):
                os.makedirs(mol_out_dir)
        else:
            mol_out_dir = None

        trainer.fit(mol_out_dir=mol_out_dir)

    if args.gen:
        print('start generating...')
        trainer.generate_molecule(num=args.gen_num, out_path=args.gen_out_path, mute=False)
