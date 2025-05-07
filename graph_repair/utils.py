# coding=utf-8
"""
Anonymous author
part of codes are taken from gcpn/graphRNN's open-source code.
Description: load raw smiles, construct node/edge matrix.
"""

import sys
import os

import numpy as np
import networkx as nx
import random

from rdkit import Chem
from rdkit.Chem import rdmolops

import torch
import torch.nn.functional as F

def mol_to_nx(mol):
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   symbol=atom.GetSymbol(),
                   formal_charge=atom.GetFormalCharge(),
                   implicit_valence=atom.GetImplicitValence(),
                   ring_atom=atom.IsInRing(),
                   degree=atom.GetDegree(),
                   hybridization=atom.GetHybridization())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType())
    return G

    
def get_maxlen_of_bfs_queue(path):
    """
    Calculate the maxlen of bfs queue.
    """
    fp = open(path, 'r')
    max_all = []
    cnt = 0
    for smiles in fp:
        cnt += 1
        if cnt % 10000 == 0:
            print('cur cnt %d' % cnt)
        smiles = smiles.strip()
        mol = Chem.MolFromSmiles(smiles)
        #adj = construct_adj_matrix(mol)
        graph = mol_to_nx(mol)
        N = len(graph.nodes)
        for i in range(N):
            start = i
            order, max_ = bfs_seq(graph, start)
            max_all.append(max_)
    print(max(max_all))


def set_seed(seed, cuda=False):

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)        
        
    print('set seed for random numpy and torch')


def save_one_mol(path, smile, cur_iter=None, score=None):
    """
    save one molecule
    mode: append
    """
    cur_iter = str(cur_iter)

    fp = open(path, 'a')
    fp.write('%s  %s  %s\n' % (cur_iter, smile, str(score)))
    fp.close()


def save_one_reward(path, reward, score, loss, cur_iter):
    """
    save one iter reward/score
    mode: append
    """
    fp = open(path, 'a')
    fp.write('cur_iter: %d | reward: %.5f | score: %.5f | loss: %.5f\n' % (cur_iter, reward, score, loss))
    fp.close()

def save_one_optimized_molecule(path, org_smile, optim_smile, optim_plogp, cur_iter, ranges, sim):
    """
    path: save path
    org_smile: molecule to be optimized
    org_plogp: original plogp
    optim_smile: with shape of (4, ), containing optimized smiles with similarity constrained 0(0.2/0.4/0.6) 
    optim_plogp:  corespongding plogp 
    cur_iter: molecule index

    """
    start = ranges[0]
    end = ranges[1]
    fp1 = open(path + '/sim0_%d_%d' % (ranges[0], ranges[1]), 'a')
    fp2 = open(path + '/sim2_%d_%d' % (ranges[0], ranges[1]), 'a')
    fp3 = open(path + '/sim4_%d_%d' % (ranges[0], ranges[1]), 'a')
    fp4 = open(path + '/sim6_%d_%d' % (ranges[0], ranges[1]), 'a')
    out_string1 = '%d|%s||%s|%.5f|%.5f\n' % (cur_iter, org_smile, optim_smile[0], optim_plogp[0], sim[0])
    out_string2 = '%d|%s||%s|%.5f|%.5f\n' % (cur_iter, org_smile, optim_smile[1], optim_plogp[1], sim[1])
    out_string3 = '%d|%s||%s|%.5f|%.5f\n' % (cur_iter, org_smile, optim_smile[2], optim_plogp[2], sim[2])
    out_string4 = '%d|%s||%s|%.5f|%.5f\n' % (cur_iter, org_smile, optim_smile[3], optim_plogp[3], sim[3])

    fp1.write(out_string1)
    fp2.write(out_string2)
    fp3.write(out_string3)
    fp4.write(out_string4)
    #fp.write('cur_iter: %d | reward: %.5f | score: %.5f | loss: %.5f\n' % (cur_iter, reward, score, loss))
    fp1.close()
    fp2.close()
    fp3.close()
    fp4.close()


def update_optim_dict(optim_dict, org_smile, cur_smile, imp, sim):
    if imp <= 0. or sim == 1.0:
        return optim_dict
    
    else:
        if org_smile not in optim_dict:
            optim_dict[org_smile] = [['', -100, -1], ['', -100, -1], ['', -100, -1], ['', -100, -1]]
        if sim >= 0.:
            if imp > optim_dict[org_smile][0][1]:
                optim_dict[org_smile][0][0] = cur_smile
                optim_dict[org_smile][0][1] = imp
                optim_dict[org_smile][0][2] = sim

        if sim >= 0.2:
            if imp > optim_dict[org_smile][1][1]:
                optim_dict[org_smile][1][0] = cur_smile
                optim_dict[org_smile][1][1] = imp
                optim_dict[org_smile][1][2] = sim

        if sim >= 0.4:
            if imp > optim_dict[org_smile][2][1]:
                optim_dict[org_smile][2][0] = cur_smile
                optim_dict[org_smile][2][1] = imp
                optim_dict[org_smile][2][2] = sim

        if sim >= 0.6:
            if imp > optim_dict[org_smile][3][1]:
                optim_dict[org_smile][3][0] = cur_smile
                optim_dict[org_smile][3][1] = imp
                optim_dict[org_smile][3][2] = sim  
        return optim_dict                          


def update_total_optim_dict(total_optim_dict, optim_dict):
    all_keys = list(optim_dict.keys())
    for key in all_keys:
        if key not in total_optim_dict:
            total_optim_dict[key] = [['', -100, -1], ['', -100, -1], ['', -100, -1], ['', -100, -1]]
        
        if optim_dict[key][0][1] > total_optim_dict[key][0][1]:
            total_optim_dict[key][0][0] = optim_dict[key][0][0]
            total_optim_dict[key][0][1] = optim_dict[key][0][1]
            total_optim_dict[key][0][2] = optim_dict[key][0][2]

        if optim_dict[key][1][1] > total_optim_dict[key][1][1]:
            total_optim_dict[key][1][0] = optim_dict[key][1][0]
            total_optim_dict[key][1][1] = optim_dict[key][1][1]
            total_optim_dict[key][1][2] = optim_dict[key][1][2]

        if optim_dict[key][2][1] > total_optim_dict[key][2][1]:
            total_optim_dict[key][2][0] = optim_dict[key][2][0]
            total_optim_dict[key][2][1] = optim_dict[key][2][1]
            total_optim_dict[key][2][2] = optim_dict[key][2][2]

        if optim_dict[key][3][1] > total_optim_dict[key][3][1]:
            total_optim_dict[key][3][0] = optim_dict[key][3][0]
            total_optim_dict[key][3][1] = optim_dict[key][3][1]
            total_optim_dict[key][3][2] = optim_dict[key][3][2]
    return total_optim_dict


# stage_event_flow = [[x,x,x], []] # flag event definition

class Reasonable_judgment(object):
    def __init__(self, nodes, edges, no_re_map):
        self.edges = edges
        self.node_type = nodes
        self.no_re_map = no_re_map
        sys.setrecursionlimit(5000)

    def find_object(self, behavior, start):
        for i in range(start, len(self.edges)):
            if self.edges[i][0] == self.edges[start][0]:
                if self.node_type[self.edges[i][1]] == behavior[1][0:2] and self.no_re_map[self.edges[i][2]] == behavior[2]:
                    return True, i
        return False, None

    def find_subject(self, behavior, start):
        for i in range(start, len(self.edges)):
            # print(self.node_type[self.edges[i][0]])
            if self.node_type[self.edges[i][0]] == behavior[0][0:2]:
                return True, i
        return False, None

    def factorial(self, behavior, start, flag):
        if flag == 1:
            return True, start
        elif start == len(self.edges):
            return False, start
        else:
            flag1, start1 = self.find_subject(behavior, start)
            if flag1:
                flag2, start2 = self.find_object(behavior, start1)
                if flag2:
                    return self.factorial(behavior, start2, 1)
                else:
                    return self.factorial(behavior, start1 + 1, 0)
            else:
                return False, start

    def knowledge_judge(self):
        sum_flag = 0   
        stage_count = 0
        ti_me = 0 
        sum_flow_index = []
        sum_edge_time_list = [[] for i in range(len(stage_event_flow))]
        for stage in stage_event_flow:
            flow_count = 0
            flow_start_time = ti_me               
            flow_flag_time = []
            flow_index_dict = {}
            edge_time_list = [[] for i in range(len(stage))]
            for flow in stage:                         
                # current_node = self.edges[ti_me][0]
                behavior_count = 0
                # flow_start_time = ti_me                            
                for behavior in flow:                          
                    be_start_time = flow_start_time
                    if be_start_time == len(self.edges) and behavior_count != len(flow):
                        break
                    else:
                        be_flag, be_start_time = self.factorial(behavior, be_start_time, 0)
                    '''if behavior[0][0:2] != self.node_type[self.edges[be_start_time][0]]:
                        be_flag, be_start_time = self.factorial(behavior, be_start_time, 0)
                    else:
                        be_flag, be_start_time = self.find_object(behavior, be_start_time)'''

                    if be_flag:
                        # print(be_start_time)
                        edge_time_list[flow_count].append(be_start_time)
                        behavior_count += 1
                        flow_start_time = be_start_time + 1
                        continue

                if behavior_count == len(flow):
                    flow_flag_time.append(flow_start_time)
                    flow_index_dict[flow_start_time] = flow_count

                flow_start_time = ti_me
                flow_count += 1

            if len(flow_flag_time) != 0:
                sum_flag += 1
                ti_me = min(flow_flag_time)
                sum_flow_index.append(flow_index_dict[ti_me])
                sum_edge_time_list[stage_count] = edge_time_list[flow_index_dict[ti_me]]

            stage_count += 1

        if sum_flag == 4:
            return True, sum_flag, sum_flow_index, sum_edge_time_list
        else:
            return False, sum_flag, sum_flow_index, sum_edge_time_list
