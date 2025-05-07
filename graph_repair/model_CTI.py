# !/usr/bin/env python
# -*-coding:utf-8 -*-

import os
import sys
import numpy as np
import math
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from MaskGAF import MaskedGraphAF

from rdkit import Chem
import environment as env
# from environment import check_valency, convert_radical_electrons_to_hydrogens
from utils import save_one_mol, save_one_reward


class GraphFlowModel(nn.Module):
    """
    Reminder:
        self.args: deq_coeff
                   deq_type

    Args:


    Returns:

    """

    def __init__(self, max_size, node_dim, bond_dim, edge_unroll, args):
        super(GraphFlowModel, self).__init__()
        self.max_size = max_size
        self.node_dim = node_dim
        self.bond_dim = bond_dim
        self.edge_unroll = edge_unroll
        self.args = args

        ###Flow hyper-paramters
        self.num_flow_layer = self.args.num_flow_layer
        self.nhid = self.args.nhid
        self.nout = self.args.nout

        self.node_masks = None
        if self.node_masks is None:
            self.node_masks, self.adj_masks, \
            self.link_prediction_index, self.flow_core_edge_masks = self.initialize_masks(max_node_unroll=self.max_size,
                                                                                          max_edge_unroll=self.edge_unroll)

        self.latent_step = self.node_masks.size(
            0)  # (max_size) + (max_edge_unroll - 1) / 2 * max_edge_unroll + (max_size - max_edge_unroll) * max_edge_unroll
        self.latent_node_length = self.max_size * self.node_dim
        self.latent_edge_length = (self.latent_step - self.max_size) * self.bond_dim
        print('latent node length: %d' % self.latent_node_length)
        print('latent edge length: %d' % self.latent_edge_length)

        self.constant_pi = nn.Parameter(torch.Tensor([3.1415926535]), requires_grad=False)
        # learnable
        if self.args.learn_prior:
            self.prior_ln_var = nn.Parameter(torch.zeros([1]))  # log(1^2) = 0
            nn.init.constant_(self.prior_ln_var, 0.)
        else:
            self.prior_ln_var = nn.Parameter(torch.zeros([1]), requires_grad=False)

        self.dp = False
        num_gpus = torch.cuda.device_count()
        if num_gpus >= 1:
            self.dp = True
            print('using %d GPUs' % num_gpus)

        self.flow_core = MaskedGraphAF(self.node_masks, self.adj_masks,
                                       self.link_prediction_index,
                                       num_flow_layer=self.num_flow_layer,
                                       graph_size=self.max_size,
                                       num_node_type=self.node_dim,
                                       num_edge_type=self.bond_dim,
                                       args=self.args,
                                       nhid=self.nhid,
                                       nout=self.nout)
        if self.dp:
            self.flow_core = nn.DataParallel(self.flow_core)

    def forward(self, inp_node_features, inp_adj_features):
        """
        Args:
            inp_node_features: (B, N, 9)
            inp_adj_features: (B, 4, N, N)

        Returns:
            z: [(B, node_num*9), (B, edge_num*4)]
            logdet:  ([B], [B])
        """
        # TODO: add dropout/normalize

        # inp_node_features_cont = inp_node_features #(B, N, 9) #! this is buggy. shallow copy
        inp_node_features_cont = inp_node_features.clone()  # (B, N, 9)

        inp_adj_features_cont = inp_adj_features[:, :, self.flow_core_edge_masks].clone()  # (B, 4, edge_num)
        inp_adj_features_cont = inp_adj_features_cont.permute(0, 2, 1).contiguous()  # (B, edge_num, 4)

        if self.args.deq_type == 'random':
            # TODO: put the randomness on GPU.!
            inp_node_features_cont += self.args.deq_coeff * torch.rand(
                inp_node_features_cont.size()).cuda()  # (B, N, 9)
            inp_adj_features_cont += self.args.deq_coeff * torch.rand(
                inp_adj_features_cont.size()).cuda()  # (B, edge_num, 4)

        elif self.args.deq_type == 'variational':
            # TODO: add variational deq.
            raise ValueError('current unsupported method: %s' % self.args.deq_type)
        else:
            raise ValueError('unsupported dequantization type (%s)' % self.args.deq_type)

        z, logdet = self.flow_core(inp_node_features, inp_adj_features,
                                   inp_node_features_cont, inp_adj_features_cont)

        if self.args.deq_type == 'random':
            return z, logdet, self.prior_ln_var

        elif self.args.deq_type == 'variational':
            # TODO: try variational dequantization
            return z, logdet, deq_logp, deq_logdet

    def reinforce_forward(self, temperature=0.75, mute=False, batch_size=32, max_size_rl=48, in_baseline=None,
                          cur_iter=None):
        """
        Fintuning model using reinforce algorithm
        Args:
            temperature: generation temperature
            batch_size: batch_size for collecting data
            max_size_rl: maximal num of atoms allowed for generation

        Returns:

        """
        # assert cur_baseline is not None
        # num2bond = {0: Chem.rdchem.BondType.SINGLE, 1: Chem.rdchem.BondType.DOUBLE, 2: Chem.rdchem.BondType.TRIPLE}
        num2bond = {0:"RD",1:"WR",2:"EX",3:"UK",4:"CD",5:"FR",6:"IJ",7:"ST",8:"RF"}
        num2bond_symbol = {0:"RD",1:"WR",2:"EX",3:"UK",4:"CD",5:"FR",6:"IJ",7:"ST",8:"RF"}
        num2atom = {0: "MP", 1: "TP", 2: "MF", 3: "SF", 4: "TF", 5: "SO"}
        # num2bond_symbol = {0: '=', 1: '==', 2: '==='}
        # [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
        # num2atom = {0: 6, 1: 7, 2: 8, 3: 9, 4: 15, 5: 16, 6: 17, 7: 35, 8: 53}

        # num2symbol = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'P', 5: 'S', 6: 'Cl', 7: 'Br', 8: 'I'}

        prior_node_dist = torch.distributions.normal.Normal(torch.zeros([self.node_dim]).cuda(),
                                                            temperature * torch.ones([self.node_dim]).cuda())
        prior_edge_dist = torch.distributions.normal.Normal(torch.zeros([self.bond_dim]).cuda(),
                                                            temperature * torch.ones([self.bond_dim]).cuda())

        node_inputs = {}
        node_inputs['node_features'] = []
        node_inputs['adj_features'] = []
        node_inputs['node_features_cont'] = []
        node_inputs['rewards'] = []
        node_inputs['baseline_index'] = []

        adj_inputs = {}
        adj_inputs['node_features'] = []
        adj_inputs['adj_features'] = []
        adj_inputs['edge_features_cont'] = []
        adj_inputs['index'] = []
        adj_inputs['rewards'] = []
        adj_inputs['baseline_index'] = []

        # if in_baseline is not None:
        #    in_baseline = torch.Tensor(in_baseline).float().cuda()
        reward_baseline = torch.zeros([max_size_rl + 5, 2]).cuda()
        # reward_baseline[:, 0] += 1. #TODO: make sure all element in [:,0] is none zero

        max_action_size = 25 * \
                          (int(max_size_rl + (self.edge_unroll - 1) * self.edge_unroll / 2 + (
                                      max_size_rl - self.edge_unroll) * self.edge_unroll))

        batch_length = 0
        total_node_step = 0
        total_edge_step = 0

        per_mol_reward = []
        per_mol_property_score = []

        ### gather training data from generation
        self.eval()  # ! very important. Because we use batch normalization, training mode will result in unrealistic molecules

        with torch.no_grad():
            while total_node_step + total_edge_step < max_action_size and batch_length < batch_size:

                traj_node_inputs = {}
                traj_node_inputs['node_features'] = []
                traj_node_inputs['adj_features'] = []
                traj_node_inputs['node_features_cont'] = []
                traj_node_inputs['rewards'] = []
                traj_node_inputs['baseline_index'] = []
                traj_adj_inputs = {}
                traj_adj_inputs['node_features'] = []
                traj_adj_inputs['adj_features'] = []
                traj_adj_inputs['edge_features_cont'] = []
                traj_adj_inputs['index'] = []
                traj_adj_inputs['rewards'] = []
                traj_adj_inputs['baseline_index'] = []

                step_cnt = 1.0

                cur_node_features = torch.zeros([1, max_size_rl, self.node_dim]).cuda()
                cur_adj_features = torch.zeros([1, self.bond_dim, max_size_rl, max_size_rl]).cuda()

                # rw_mol = Chem.RWMol()
                mol = None
                # mol_size = mol.GetNumAtoms()

                is_continue = True
                total_resample = 0
                each_node_resample = np.zeros([max_size_rl])

                step_num_data_edge = 0

                for i in range(max_size_rl):

                    step_num_data_edge = 0  # generating new node and its edges. Not sure if this will add into the final mol.

                    if not is_continue:
                        break
                    if i < self.edge_unroll:
                        edge_total = i  # edge to sample for current node
                        start = 0
                    else:
                        edge_total = self.edge_unroll
                        start = i - self.edge_unroll

                    # first generate node
                    ## reverse flow
                    latent_node = prior_node_dist.sample().view(1, -1)  # (1, 9)
                    if self.dp:
                        latent_node = self.flow_core.module.reverse(cur_node_features, cur_adj_features,
                                                                    latent_node, mode=0).view(-1)  # (9, )
                    else:
                        latent_node = self.flow_core.reverse(cur_node_features, cur_adj_features,
                                                             latent_node, mode=0).view(-1)  # (9, )
                    ## node/adj postprocessing
                    # print(latent_node.shape) #(38, 9)
                    feature_id = torch.argmax(latent_node).item()
                    total_node_step += 1
                    node_feature_cont = torch.zeros([1, self.node_dim]).cuda()
                    node_feature_cont[0, feature_id] = 1.0

                    # update traj inputs for node_id
                    traj_node_inputs['node_features'].append(
                        cur_node_features.clone())  # (1, max_size_rl, self.node_dim)
                    traj_node_inputs['adj_features'].append(
                        cur_adj_features.clone())  # (1, self.bond_dim, max_size_rl, max_size_rl)
                    traj_node_inputs['node_features_cont'].append(node_feature_cont)  # (1, self.node_dim)
                    traj_node_inputs['rewards'].append(torch.full(size=(1, 1), fill_value=step_cnt).cuda())  # (1, 1)
                    traj_node_inputs['baseline_index'].append(
                        torch.full(size=(1,), fill_value=step_cnt).long().cuda())  # (1, 1)

                    # step_cnt += 1

                    # print(num2symbol[feature_id])
                    cur_node_features[0, i, feature_id] = 1.0
                    cur_adj_features[0, :, i, i] = 1.0
                    # rw_mol.AddAtom(Chem.Atom(num2atom[feature_id]))

                    # then generate edges
                    if i == 0:
                        is_connect = True
                    else:
                        is_connect = False
                    # cur_mol_size = mol.GetNumAtoms
                    for j in range(edge_total):
                        valid = False
                        resample_edge = 0
                        invalid_bond_type_set = set()
                        while not valid:
                            if len(invalid_bond_type_set) < 9 and resample_edge <= 9:  # haven't sampled all possible bond type or is not stuck in the loop
                                latent_edge = prior_edge_dist.sample().view(1, -1)  # (1, 4)
                                if self.dp:
                                    latent_edge = self.flow_core.module.reverse(cur_node_features, cur_adj_features,
                                                                                latent_edge,
                                                                                mode=1, edge_index=torch.Tensor(
                                            [[j + start, i]]).long().cuda()).view(-1)  # (4, )
                                else:
                                    latent_edge = self.flow_core.reverse(cur_node_features, cur_adj_features,
                                                                         latent_edge, mode=1,
                                                                         edge_index=torch.Tensor(
                                                                             [[j + start, i]]).long().cuda()).view(-1)  # (4, )
                                # if mol.GetNumAtoms() < self.args.min_atoms and j == (edge_total - 1): # molecule is small and current edge is last
                                #    latent_edge[-1] = -999999. #Do argmax in first 3 dimension
                                edge_discrete_id = torch.argmax(latent_edge).item()
                            else:
                                if not mute:
                                    print('have tried all possible bond type, use virtual bond.')
                                assert resample_edge > 9 or len(invalid_bond_type_set) == 9
                                edge_discrete_id = 9  # we have no choice but to choose not to add edge between (i, j+start)
                            total_edge_step += 1
                            edge_feature_cont = torch.zeros([1, self.bond_dim]).cuda()
                            edge_feature_cont[0, edge_discrete_id] = 1.0

                            # update traj inputs for edge_id
                            traj_adj_inputs['node_features'].append(
                                cur_node_features.clone())  # 1, max_size_rl, self.node_dim
                            traj_adj_inputs['adj_features'].append(
                                cur_adj_features.clone())  # 1, self.bond_dim, max_size_rl, max_size_rl
                            traj_adj_inputs['edge_features_cont'].append(edge_feature_cont)  # 1, self.bond_dim
                            traj_adj_inputs['index'].append(
                                torch.Tensor([[j + start, i]]).long().cuda().view(1, -1))  # (1, 2)
                            step_num_data_edge += 1  # add one edge data, not sure if this should be added to the final train data

                            cur_adj_features[0, edge_discrete_id, i, j + start] = 1.0
                            cur_adj_features[0, edge_discrete_id, j + start, i] = 1.0
                            # if edge_discrete_id == 3:  # virtual edge
                            #     valid = True  # virtual edge is alway valid
                            # else:  # single/double/triple bond
                            #     rw_mol.AddBond(i, j + start, num2bond[edge_discrete_id])
                            #     valid = env.check_valency(rw_mol)
                            #     if valid:
                            #         is_connect = True
                            #         # print(num2bond_symbol[edge_discrete_id])
                            #     else:  # backtrack
                            #         rw_mol.RemoveBond(i, j + start)
                            #         cur_adj_features[0, edge_discrete_id, i, j + start] = 0.0
                            #         cur_adj_features[0, edge_discrete_id, j + start, i] = 0.0
                            #         total_resample += 1.0
                            #         each_node_resample[i] += 1.0
                            #         resample_edge += 1
                            #
                            #         invalid_bond_type_set.add(edge_discrete_id)
                            valid = True
                            if valid:
                                is_connect = True
                                    # print(num2bond_symbol[edge_discrete_id])
                            else:  # backtrack
                                # rw_mol.RemoveBond(i, j + start)
                                cur_adj_features[0, edge_discrete_id, i, j + start] = 0.0
                                cur_adj_features[0, edge_discrete_id, j + start, i] = 0.0
                                total_resample += 1.0
                                each_node_resample[i] += 1.0
                                resample_edge += 1
                                invalid_bond_type_set.add(edge_discrete_id)
                            if valid:
                                traj_adj_inputs['rewards'].append(
                                    torch.full(size=(1, 1), fill_value=step_cnt).cuda())  # (1, 1)
                                traj_adj_inputs['baseline_index'].append(
                                    torch.full(size=(1,), fill_value=step_cnt).long().cuda())  # (1)
                                # step_cnt += 1
                            else:
                                if self.args.penalty:
                                    # todo: the step_reward can be tuned here
                                    traj_adj_inputs['rewards'].append(
                                        torch.full(size=(1, 1), fill_value=-1.).cuda())  # (1, 1) invalid edge penalty
                                    traj_adj_inputs['baseline_index'].append(
                                        torch.full(size=(1,), fill_value=step_cnt).long().cuda())  # (1,)
                                    # TODO: check baselien of invalid step, maybe we do not add baseline for invalid step
                                else:
                                    traj_adj_inputs['node_features'].pop(-1)
                                    traj_adj_inputs['adj_features'].pop(-1)
                                    traj_adj_inputs['edge_features_cont'].pop(-1)
                                    traj_adj_inputs['index'].pop(-1)
                                    step_num_data_edge -= 1  # if we do not penalize invalid edge, pop train data, decrease counter by 1

                    if is_connect:  # new generated node has at least one bond with previous node, do not stop generation, backup mol from rw_mol to mol
                        is_continue = True
                        # mol = rw_mol.GetMol()

                    else:
                        is_continue = False
                    step_cnt += 1

                # TODO: check the last iter of generation
                # (Thinking)
                # The last node was not added. So after we generate the second to last
                # node and all its edges, the rest adjacent matrix and node features should all be zero
                # But current implementation append
                # num_atoms = mol.GetNumAtoms()
                # assert num_atoms <= max_size_rl

                # TODO: check if we should discard the small molecule
                # if num_atoms < self.args.min_atoms:
                #    print('#atoms of generated molecule less than %d, discarded!' % self.args.min_atoms)
                #    continue
                # else:
                #    batch_length += 1
                #    print('generating %d-th molecule done!' % batch_length)
                batch_length += 1
                # print('generating %d-th molecule with %d atoms' % (batch_length, num_atoms))

                if num_atoms < max_size_rl:
                    # ! this implementation is buggy. we only mask the last node feature cont
                    # ! But we ignore the non-zero node features in generating edges
                    # ! this pattern will make model not to generated any edges between
                    # ! the new-generated isolated node and exsiting subgraph.
                    # ! this may be the biggest bug in Reinforce algorithm!!!!!
                    # ! since the final iter/(step) has largest reward....!!!!!!!
                    # ! work around1: add a counter and mask out all node feautres in generating edges of last iter.
                    # ! work around2: do not append any data if the isolated node is not connected to subgraph.
                    # currently use work around2

                    # pop all the reinforce train-data add by at the generating the last isolated node and its edge
                    ## pop node
                    traj_node_inputs['node_features'].pop(-1)
                    traj_node_inputs['adj_features'].pop(-1)
                    traj_node_inputs['node_features_cont'].pop(-1)
                    traj_node_inputs['rewards'].pop(-1)
                    traj_node_inputs['baseline_index'].pop(-1)

                    ## pop adj
                    for pop_cnt in range(step_num_data_edge):
                        traj_adj_inputs['node_features'].pop(-1)
                        traj_adj_inputs['adj_features'].pop(-1)
                        traj_adj_inputs['edge_features_cont'].pop(-1)
                        traj_adj_inputs['index'].pop(-1)
                        traj_adj_inputs['rewards'].pop(-1)
                        traj_adj_inputs['baseline_index'].pop(-1)

                    # generated molecule doesn't reach the maximal size
                    # last node's feature should be all zero
                    # traj_node_inputs['node_features_cont'][-1] = torch.zeros([1, self.node_dim]).cuda()

                reward_valid = 2
                reward_property = 0
                reward_length = 0
                # TODO: check if it is necessary to penal the small molecule, current we do not use it
                # if num_atoms < 10:
                #    reward_length -= 1.
                # elif num_atoms >= 20
                #    reward_length += 1
                # elif num_atoms >= 30:
                #    reward_length += 4
                # reward_length = 0  # current we do not use this reward
                flag_steric_strain_filter = True
                flag_zinc_molecule_filter = True

                assert mol is not None, 'mol is None...'
                final_valid = env.check_chemical_validity(mol)

                assert final_valid is True, 'warning: use valency check during generation but the final molecule is invalid!!!'
                if not final_valid:
                    reward_valid -= 5  # this is useless, because this case will not occur in our implementation
                else:
                    final_mol = env.convert_radical_electrons_to_hydrogens(mol)
                    s = Chem.MolToSmiles(final_mol, isomericSmiles=True)
                    print(s)
                    final_mol = Chem.MolFromSmiles(s)
                    # mol filters with negative rewards
                    if not env.steric_strain_filter(final_mol):  # passes 3D conversion, no excessive strain
                        reward_valid -= 1  # TODO: check the magnitude of this reward.
                        flag_steric_strain_filter = False
                    if not env.zinc_molecule_filter(final_mol):  # does not contain any problematic functional groups
                        reward_valid -= 1
                        flag_zinc_molecule_filter = False

                    # todo: add arg for property_type here
                    property_type = self.args.property
                    assert property_type in ['qed',
                                             'plogp'], 'unsupported property optimization, choices are [qed, plogp]'

                    try:
                        if property_type == 'qed':
                            score = env.qed(final_mol)
                            reward_property += (score * self.args.qed_coeff)
                            if score > 0.945:
                                save_one_mol_path = os.path.join(self.args.save_path, 'good_mol_qed.txt')
                                save_one_mol(save_one_mol_path, s, cur_iter=cur_iter, score=score)

                        elif property_type == 'plogp':
                            score = env.penalized_logp(final_mol)

                            # TODO: design stable reward....
                            if self.args.reward_type == 'exp':
                                reward_property += (np.exp(score / self.args.exp_temperature) - self.args.exp_bias)
                            elif self.args.reward_type == 'linear':
                                reward_property += (score * self.args.plogp_coeff)
                            # elif self.args.reward_type == 'exp_with_linear':
                            #    reward_property += (score * self.args.plogp_coeff + np.exp(score / self.args.exp_temperature))
                            # elif self.args.reward_type == 'neg_exp': # not working well
                            #    if score < 0:
                            #        tmp_score = -2
                            #    else:
                            #        tmp_score = np.exp(score / self.args.exp_temperature - 1.)
                            #    reward_property += (tmp_score)
                            # score_tmp = score * self.args.plogp_coeff
                            # reward_property += (np.exp(score_tmp / self.args.exp_temperature) - 4.0)

                            if score > 4.0:
                                save_one_mol_path = os.path.join(self.args.save_path, 'good_mol_plogp.txt')
                                save_one_mol(save_one_mol_path, s, cur_iter=cur_iter, score=score)
                    except:
                        print('generated mol does not pass env.qed/plogp')
                        # reward_property -= 2.0
                        # TODO: check what we should do if the model do not pass qed/plogp test
                        # workaround1: add a extra penalty reward
                        # workaround2: discard the molecule.

                reward_final_total = reward_valid + reward_property + reward_length
                # reward_final_total = reward_property
                per_mol_reward.append(reward_final_total)
                per_mol_property_score.append(reward_property)

                # todo: add arg for reward decay weight here
                reward_decay = self.args.reward_decay

                node_inputs['node_features'].append(torch.cat(traj_node_inputs['node_features'],
                                                              dim=0))  # append tensor of shape (max_size_rl, max_size_rl, self.node_dim)
                node_inputs['adj_features'].append(torch.cat(traj_node_inputs['adj_features'],
                                                             dim=0))  # append tensor of shape (max_size_rl, bond_dim, max_size_rl, max_size_rl)
                node_inputs['node_features_cont'].append(
                    torch.cat(traj_node_inputs['node_features_cont'], dim=0))  # append tensor of shape (max_size_rl, 9)

                traj_node_inputs_baseline_index = torch.cat(traj_node_inputs['baseline_index'], dim=0)  # (max_size_rl)
                traj_node_inputs_rewards = torch.cat(traj_node_inputs['rewards'],
                                                     dim=0)  # tensor of shape (max_size_rl, 1)
                traj_node_inputs_rewards[traj_node_inputs_rewards > 0] = \
                    reward_final_total * torch.pow(reward_decay, step_cnt - 1. - traj_node_inputs_rewards[
                        traj_node_inputs_rewards > 0])
                node_inputs['rewards'].append(
                    traj_node_inputs_rewards)  # append tensor of shape (max_size_rl, 1)
                node_inputs['baseline_index'].append(traj_node_inputs_baseline_index)

                for ss in range(traj_node_inputs_rewards.size(0)):
                    reward_baseline[traj_node_inputs_baseline_index[ss]][0] += 1.0
                    reward_baseline[traj_node_inputs_baseline_index[ss]][1] += traj_node_inputs_rewards[ss][0]

                adj_inputs['node_features'].append(
                    torch.cat(traj_adj_inputs['node_features'], dim=0))  # (step, max_size_rl, self.node_dim)
                adj_inputs['adj_features'].append(
                    torch.cat(traj_adj_inputs['adj_features'], dim=0))  # (step, bond_dim, max_size_rl, max_size_rl)
                adj_inputs['edge_features_cont'].append(
                    torch.cat(traj_adj_inputs['edge_features_cont'], dim=0))  # (step, 4)
                adj_inputs['index'].append(torch.cat(traj_adj_inputs['index'], dim=0))  # (step, 2)

                traj_adj_inputs_baseline_index = torch.cat(traj_adj_inputs['baseline_index'],
                                                           dim=0)  # (step)
                traj_adj_inputs_rewards = torch.cat(traj_adj_inputs['rewards'], dim=0)
                traj_adj_inputs_rewards[traj_adj_inputs_rewards > 0] = \
                    reward_final_total * torch.pow(reward_decay,
                                                   step_cnt - 1. - traj_adj_inputs_rewards[traj_adj_inputs_rewards > 0])
                adj_inputs['rewards'].append(traj_adj_inputs_rewards)
                adj_inputs['baseline_index'].append(traj_adj_inputs_baseline_index)

                for ss in range(traj_adj_inputs_rewards.size(0)):
                    reward_baseline[traj_adj_inputs_baseline_index[ss]][0] += 1.0
                    reward_baseline[traj_adj_inputs_baseline_index[ss]][1] += traj_adj_inputs_rewards[ss][0]

        self.train()
        # TODO: check whether to finetuning batch normalization
        for module in self.modules():
            if isinstance(module, torch.nn.modules.BatchNorm1d):
                module.eval()

        for i in range(reward_baseline.size(0)):
            if reward_baseline[i, 0] == 0:
                reward_baseline[i, 0] += 1.

        reward_baseline_per_step = reward_baseline[:, 1] / reward_baseline[:, 0]  # (max_size_rl, )
        # TODO: check the baseline for invalid edge penalty step....
        # panelty step do not have property reward. So its magnitude may be quite different from others.

        if in_baseline is not None:
            #    #print(reward_baseline_per_step.size())
            assert in_baseline.size() == reward_baseline_per_step.size()
            reward_baseline_per_step = reward_baseline_per_step * (
                        1. - self.args.moving_coeff) + in_baseline * self.args.moving_coeff
            print('calculating moving baseline per step')

        node_inputs_node_features = torch.cat(node_inputs['node_features'], dim=0)  # (total_size, max_size_rl, 9)
        node_inputs_adj_features = torch.cat(node_inputs['adj_features'],
                                             dim=0)  # (total_size, 4, max_size_rl, max_size_rl)
        node_inputs_node_features_cont = torch.cat(node_inputs['node_features_cont'], dim=0)  # (total_size, 9)
        node_inputs_rewards = torch.cat(node_inputs['rewards'], dim=0).view(-1)  # (total_size,)
        node_inputs_baseline_index = torch.cat(node_inputs['baseline_index'], dim=0).long()  # (total_size,)
        node_inputs_baseline = torch.index_select(reward_baseline_per_step, dim=0,
                                                  index=node_inputs_baseline_index)  # (total_size, )

        adj_inputs_node_features = torch.cat(adj_inputs['node_features'], dim=0)  # (total_size, max_size_rl, 9)
        adj_inputs_adj_features = torch.cat(adj_inputs['adj_features'],
                                            dim=0)  # (total_size, 4, max_size_rl, max_size_rl)
        adj_inputs_edge_features_cont = torch.cat(adj_inputs['edge_features_cont'], dim=0)  # (total_size, 4)
        adj_inputs_index = torch.cat(adj_inputs['index'], dim=0)  # (total_size, 2)
        adj_inputs_rewards = torch.cat(adj_inputs['rewards'], dim=0).view(-1)  # (total_size,)
        adj_inputs_baseline_index = torch.cat(adj_inputs['baseline_index'], dim=0).long()  # (total_size,)
        adj_inputs_baseline = torch.index_select(reward_baseline_per_step, dim=0,
                                                 index=adj_inputs_baseline_index)  # (total_size, )

        if self.args.deq_type == 'random':
            # ! here comes the randomness from torch cpu
            # we put the device in torch.rand so it is genereated directly on gpu
            node_inputs_node_features_cont += self.args.deq_coeff * torch.rand(
                node_inputs_node_features_cont.size(),
                device='cuda:%d' % (node_inputs_node_features_cont.get_device()))  # (total_size, 9)
            adj_inputs_edge_features_cont += self.args.deq_coeff * torch.rand(
                adj_inputs_edge_features_cont.size(),
                device='cuda:%d' % (adj_inputs_edge_features_cont.get_device()))  # (total_size, 4)

        elif self.args.deq_type == 'variational':
            # TODO: add variational deq.
            raise ValueError('current unsupported method: %s' % self.args.deq_type)
        else:
            raise ValueError('unsupported dequantization type (%s)' % self.args.deq_type)

        if self.dp:
            node_function = self.flow_core.module.forward_rl_node
            edge_function = self.flow_core.module.forward_rl_edge
        else:
            node_function = self.flow_core.forward_rl_node
            edge_function = self.flow_core.forward_rl_edge

        z_node, logdet_node = node_function(node_inputs_node_features, node_inputs_adj_features,
                                            node_inputs_node_features_cont)  # (total_step, 9), (total_step, )

        z_edge, logdet_edge = edge_function(adj_inputs_node_features, adj_inputs_adj_features,
                                            adj_inputs_edge_features_cont,
                                            adj_inputs_index)  # (total_step, 4), (total_step, )

        node_total_length = z_node.size(0) * float(self.node_dim)
        edge_total_length = z_edge.size(0) * float(self.bond_dim)

        # logdet_node = logdet_node - self.latent_node_length  # calculate probability of a region from probability density, minus constant has no effect on optimization
        # logdet_edge = logdet_edge - self.latent_edge_length  # calculate probability of a region from probability density, minus constant has no effect on optimization

        ll_node = -1 / 2 * (
                    torch.log(2 * self.constant_pi) + self.prior_ln_var + torch.exp(-self.prior_ln_var) * (z_node ** 2))
        ll_node = ll_node.sum(-1)  # (B)
        ll_edge = -1 / 2 * (
                    torch.log(2 * self.constant_pi) + self.prior_ln_var + torch.exp(-self.prior_ln_var) * (z_edge ** 2))
        ll_edge = ll_edge.sum(-1)  # (B)

        ll_node += logdet_node  # ([B])
        ll_edge += logdet_edge  # ([B])

        # TODO: check baseline of penalty step(invalid edge)
        # TODO: check whether moving baseline is better than batch average.
        ll_node = ll_node * (node_inputs_rewards - node_inputs_baseline)
        ll_edge = ll_edge * (adj_inputs_rewards - adj_inputs_baseline)

        if self.args.deq_type == 'random':
            if self.args.divide_loss:
                # print(ll_node.size())
                # print(ll_edge.size())
                return -((ll_node.sum() + ll_edge.sum()) / (
                            node_total_length + edge_total_length) - 1.0), per_mol_reward, per_mol_property_score, reward_baseline_per_step
            else:
                # ! useless
                return -torch.sum(ll_node + ll_edge) / batch_length  # scalar

    def generate_one_molecule_with_latent_provided(self, latent_manipulated, latent_org, max_step, max_atoms=48,
                                                   temperature=0.75):
        """
        Generated a molecule with initial latent provided.
        If latent violate the valency, resample latent.
        """
        generate_start_t = time()
        max_step = np.random.randint(23)
        with torch.no_grad():
            num2bond = {0: Chem.rdchem.BondType.SINGLE, 1: Chem.rdchem.BondType.DOUBLE, 2: Chem.rdchem.BondType.TRIPLE}
            num2bond_symbol = {0: '=', 1: '==', 2: '==='}
            # [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
            num2atom = {0: 6, 1: 7, 2: 8, 3: 9, 4: 15, 5: 16, 6: 17, 7: 35, 8: 53}
            num2symbol = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'P', 5: 'S', 6: 'Cl', 7: 'Br', 8: 'I'}

            prior_node_dist = torch.distributions.normal.Normal(torch.zeros([self.node_dim]).cuda(),
                                                                temperature * torch.ones([self.node_dim]).cuda())
            prior_edge_dist = torch.distributions.normal.Normal(torch.zeros([self.bond_dim]).cuda(),
                                                                temperature * torch.ones([self.bond_dim]).cuda())
            org_latent_node = latent_org[:342]
            org_latent_edge = latent_org[342:]
            m_latent_node = latent_manipulated[:342]  # (self.max_size * 9)
            m_latent_edge = latent_manipulated[342:]  # (latent_size - self.max_size * 9)
            assert m_latent_node.size(0) == (self.max_size * 9)
            assert m_latent_edge.size(0) == ((11 * 6 + (38 - 12) * 12)) * 4
            cur_node_features = torch.zeros([1, max_atoms, self.node_dim]).cuda()
            cur_adj_features = torch.zeros([1, self.bond_dim, max_atoms, max_atoms]).cuda()

            rw_mol = Chem.RWMol()  # editable mol
            mol = None

            total_edge_index = -1

            is_continue = True
            total_resample = 0
            each_node_resample = np.zeros([max_atoms])
            for i in range(max_atoms):
                if not is_continue:
                    break
                if i < self.edge_unroll:
                    edge_total = i  # edge to sample for current node
                    start = 0
                else:
                    edge_total = self.edge_unroll
                    start = i - self.edge_unroll
                # first generate node
                ## reverse flow
                if i < self.max_size and i < max_step:  # node less than 38, use pre latent
                    latent_node = org_latent_node[self.node_dim * i: self.node_dim * (i + 1)].clone().view(1,
                                                                                                           -1)  # (1, 9)
                elif i < self.max_size and i >= max_step:
                    latent_node = m_latent_node[self.node_dim * i: self.node_dim * (i + 1)].clone().view(1,
                                                                                                         -1)  # (1, 9)
                    # latent_node += 0.5 * prior_node_dist.sample().view(1, -1) #(1, 9)

                    # print('using predefined node latent for node %d' % i)
                else:
                    print('sample node latent for node %d' % i)
                    latent_node = prior_node_dist.sample().view(1, -1)  # (1, 9)

                if self.dp:
                    latent_node = self.flow_core.module.reverse(cur_node_features, cur_adj_features,
                                                                latent_node, mode=0).view(-1)  # (9, )
                else:
                    latent_node = self.flow_core.reverse(cur_node_features, cur_adj_features,
                                                         latent_node, mode=0).view(-1)  # (9, )
                ## node/adj postprocessing
                # print(latent_node.shape) #(38, 9)
                feature_id = torch.argmax(latent_node).item()
                # print(num2symbol[feature_id])
                cur_node_features[0, i, feature_id] = 1.0
                cur_adj_features[0, :, i, i] = 1.0
                rw_mol.AddAtom(Chem.Atom(num2atom[feature_id]))

                # then generate edges
                if i == 0:
                    is_connect = True
                else:
                    is_connect = False
                # cur_mol_size = mol.GetNumAtoms
                for j in range(edge_total):
                    first = True
                    total_edge_index += 1
                    valid = False
                    resample_edge = 0
                    invalid_bond_type_set = set()
                    while not valid:
                        # TODO: add cache. Some atom can not get the right edge type and is stuck in the loop
                        # TODO: add cache. invalid bond set
                        if len(invalid_bond_type_set) < 3 and resample_edge <= 50:  # haven't sampled all possible bond type or is not stuck in the loop
                            if first:
                                first = False
                                if i < self.max_size and i < max_step:
                                    latent_edge = org_latent_edge[self.bond_dim * total_edge_index: self.bond_dim * (
                                                total_edge_index + 1)].clone().view(1, -1)  # (1, 4)
                                elif i < self.max_size and i >= max_step:
                                    latent_edge = m_latent_edge[self.bond_dim * total_edge_index: self.bond_dim * (
                                                total_edge_index + 1)].clone().view(1, -1)  # (1, 4)
                                    # latent_edge += 0.5 * prior_edge_dist.sample().view(1, -1) #(1, 4)
                                else:
                                    print('sample edge latent for edge %d' % total_edge_index)
                                    latent_edge = prior_edge_dist.sample().view(1, -1)  # (1, 4)
                            else:
                                print('did not use predefined edge latent for generating %d-th edge' % total_edge_index)
                                latent_edge = prior_edge_dist.sample().view(1, -1)  # (1, 4)

                            if self.dp:
                                latent_edge = self.flow_core.module.reverse(cur_node_features, cur_adj_features,
                                                                            latent_edge,
                                                                            mode=1, edge_index=torch.Tensor(
                                        [[j + start, i]]).long().cuda()).view(-1)  # (4, )
                            else:
                                latent_edge = self.flow_core.reverse(cur_node_features, cur_adj_features, latent_edge,
                                                                     mode=1, edge_index=torch.Tensor(
                                        [[j + start, i]]).long().cuda()).view(-1)  # (4, )
                            edge_discrete_id = torch.argmax(latent_edge).item()
                        else:
                            print('have tried all possible bond type, use virtual bond.')
                            assert resample_edge > 50 or len(invalid_bond_type_set) == 3
                            edge_discrete_id = 3  # we have no choice but to choose not to add edge between (i, j+start)
                        cur_adj_features[0, edge_discrete_id, i, j + start] = 1.0
                        cur_adj_features[0, edge_discrete_id, j + start, i] = 1.0
                        if edge_discrete_id == 3:  # virtual edge
                            valid = True
                        else:  # single/double/triple bond
                            rw_mol.AddBond(i, j + start, num2bond[edge_discrete_id])
                            valid = env.check_valency(rw_mol)
                            if valid:
                                is_connect = True
                                # print(num2bond_symbol[edge_discrete_id])
                            else:  # backtrack
                                rw_mol.RemoveBond(i, j + start)
                                cur_adj_features[0, edge_discrete_id, i, j + start] = 0.0
                                cur_adj_features[0, edge_discrete_id, j + start, i] = 0.0
                                total_resample += 1.0
                                each_node_resample[i] += 1.0
                                resample_edge += 1

                                invalid_bond_type_set.add(edge_discrete_id)
                if is_connect:  # new generated node has at least one bond with previous node, do not stop generation, backup mol from rw_mol to mol
                    is_continue = True
                    mol = rw_mol.GetMol()

                else:
                    is_continue = False

            # mol = rw_mol.GetMol() # mol backup
            assert mol is not None, 'mol is None...'

            # final_valid = check_valency(mol)
            final_valid = env.check_chemical_validity(mol)
            assert final_valid is True, 'warning: use valency check during generation but the final molecule is invalid!!!'

            final_mol = env.convert_radical_electrons_to_hydrogens(mol)
            smiles = Chem.MolToSmiles(final_mol, isomericSmiles=True)
            assert '.' not in smiles, 'warning: use is_connect to check stop action, but the final molecule is disconnected!!!'

            final_mol = Chem.MolFromSmiles(smiles)

            # mol = convert_radical_electrons_to_hydrogens(mol)
            num_atoms = final_mol.GetNumAtoms()
            num_bonds = final_mol.GetNumBonds()

            pure_valid = 0
            if total_resample == 0:
                pure_valid = 1.0
            print('smiles: %s | #atoms: %d | #bonds: %d | #resample: %.5f | time: %.5f |' % (
            smiles, num_atoms, num_bonds, total_resample, time() - generate_start_t))
            return smiles, pure_valid, num_atoms

    def rule_check(self, i,j2,edge_list, cur_adj_features):
        reasonal_list = [[1,3,1],[1,4,1],[1,5,1],[2,3,1],[2,4,1],[2,5,1],[1,3,2],[1,4,2],[1,5,2],[2,3,2],[2,4,2],[2,5,2],[1,3,3],[1,4,3],[1,5,3],[2,3,3],[2,4,3],[2,5,3],[1,3,4],[1,4,4],[1,5,4],[2,3,4],[2,4,4],[2,5,4],[1,1,4],[2,2,4],[1,2,4],[2,1,4],[1,3,5],[1,4,5],[1,5,5],[2,3,5],[2,4,5],[2,5,5],[1,1,6],[2,2,6],[2,1,6],[1,2,6],[1,1,7],[2,2,7],[2,1,7],[1,2,7],[1,6,8],[2,6,8],[6,1,9],[6,2,9],[3,1,1],[4,1,1],[5,1,1],[3,2,1],[4,2,1],[5,2,1],[3,1,2],[4,1,2],[5,1,2],[3,2,2],[4,2,2],[5,2,2],[3,1,3],[4,1,3],[5,1,3],[3,2,3],[4,2,3],[5,2,3],[3,1,4],[4,1,4],[5,1,4],[3,2,4],[4,2,4],[5,2,4],[3,1,5],[4,1,5],[5,1,5],[3,2,5],[4,2,5],[5,2,5],[6,1,8],[6,2,8],[1,6,9],[2,6,9]]

        file_list = [3,4,5]
        process_list = [1,2]
        socket_list = [6]
        file_action = [1,2,3,4,5]
        process_action = [1,2,3,4,5,6,7,8,9]
        flag = True
        if edge_list not in reasonal_list:
            flag = False

        if ((edge_list[0] in file_list) or (edge_list[1] in file_list)) and (edge_list[2] in file_action) :
            if edge_list[0] in file_list:
                file_node = i
            else:
                file_node = j2
            for i1 in range(0,self.max_size):
                if cur_adj_features[0,3,file_node,i1] == 1.0:
                    flag = False

        if ((edge_list[0] in process_list) != (edge_list[1] in process_list)) and (edge_list[2] in process_action) :
            if edge_list[0] in process_list:
                p_node = i
            else:
                p_node = j2
            for i1 in range(0,self.max_size):
                if cur_adj_features[0,3,p_node,i1] == 1.0:
                    flag = False
        if ((edge_list[0] in process_list) and (edge_list[1] in process_list)) and (edge_list[2] in process_action) :
            p_node_1 = i
            p_node_2 = j2
            for i1 in range(0,self.max_size):
                if cur_adj_features[0,3,p_node_1,i1] == 1.0 or cur_adj_features[0,3,p_node_2,i1] == 1.0:
                    flag = False

        if ((edge_list[0] in process_list) and (edge_list[1] in process_list)) and (edge_list[2]==6):
            p_node = i
            for i1 in range(0,self.max_size):
                for i2 in process_action:
                    if cur_adj_features[0,i2-1,p_node,i1] == 1.0:
                        flag = False

        if ((edge_list[0] in process_list) and (edge_list[1] in process_list)) and (edge_list[2] in process_action):
            p_node_1 = i
            p_node_2 = j2
            for i1 in range(0,self.max_size):
                if cur_adj_features[0,5,p_node_1,p_node_2] == 1.0:
                    flag = False
        return flag

    def generate(self, temperature=0.75, mute=False, max_atoms=48, cnt=None):
        """
        inverse flow to generate molecule
        Args:
            temp: temperature of normal distributions, we sample from (0, temp^2 * I)
        """
        generate_start_t = time()
        with torch.no_grad():
            # num2bond = {0: Chem.rdchem.BondType.SINGLE, 1: Chem.rdchem.BondType.DOUBLE, 2: Chem.rdchem.BondType.TRIPLE}
            # num2bond_symbol = {0: '=', 1: '==', 2: '==='}
            # # [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
            # num2atom = {0: 6, 1: 7, 2: 8, 3: 9, 4: 15, 5: 16, 6: 17, 7: 35, 8: 53}
            # num2symbol = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'P', 5: 'S', 6: 'Cl', 7: 'Br', 8: 'I'}
            num2bond = {0: "RD", 1: "WR", 2: "EX", 3: "UK", 4: "CD", 5: "FR", 6: "IJ", 7: "ST", 8: "RF", 9: "none"}
            num2bond_symbol = {0: "RD", 1: "WR", 2: "EX", 3: "UK", 4: "CD", 5: "FR", 6: "IJ", 7: "ST", 8: "RF", 9: "none"}
            num2atom = {0: "MP", 1: "TP", 2: "MF", 3: "SF", 4: "TF", 5: "SO"}
            reasonal_list = [[1,3,1],[1,4,1],[1,5,1],[2,3,1],[2,4,1],[2,5,1],[1,3,2],[1,4,2],[1,5,2],[2,3,2],[2,4,2],[2,5,2],[1,3,3],[1,4,3],[1,5,3],[2,3,3],[2,4,3],[2,5,3],[1,3,4],[1,4,4],[1,5,4],[2,3,4],[2,4,4],[2,5,4],[1,1,4],[2,2,4],[1,2,4],[2,1,4],[1,3,5],[1,4,5],[1,5,5],[2,3,5],[2,4,5],[2,5,5],[1,1,6],[2,2,6],[2,1,6],[1,2,6],[1,1,7],[2,2,7],[2,1,7],[1,2,7],[1,6,8],[2,6,8],[6,1,9],[6,2,9],[3,1,1],[4,1,1],[5,1,1],[3,2,1],[4,2,1],[5,2,1],[3,1,2],[4,1,2],[5,1,2],[3,2,2],[4,2,2],[5,2,2],[3,1,3],[4,1,3],[5,1,3],[3,2,3],[4,2,3],[5,2,3],[3,1,4],[4,1,4],[5,1,4],[3,2,4],[4,2,4],[5,2,4],[3,1,5],[4,1,5],[5,1,5],[3,2,5],[4,2,5],[5,2,5],[6,1,8],[6,2,8],[1,6,9],[2,6,9]]
            reasonal_list_right = [[1,3,1],[1,4,1],[1,5,1],[2,3,1],[2,4,1],[2,5,1],[1,3,2],[1,4,2],[1,5,2],[2,3,2],[2,4,2],[2,5,2],[1,3,3],[1,4,3],[1,5,3],[2,3,3],[2,4,3],[2,5,3],[1,3,4],[1,4,4],[1,5,4],[2,3,4],[2,4,4],[2,5,4],[1,1,4],[2,2,4],[1,2,4],[2,1,4],[1,3,5],[1,4,5],[1,5,5],[2,3,5],[2,4,5],[2,5,5],[1,1,6],[2,2,6],[2,1,6],[1,2,6],[1,1,7],[2,2,7],[2,1,7],[1,2,7],[1,6,8],[2,6,8],[6,1,9],[6,2,9]]
            reasonal_list_reverse = [[3,1,1],[4,1,1],[5,1,1],[3,2,1],[4,2,1],[5,2,1],[3,1,2],[4,1,2],[5,1,2],[3,2,2],[4,2,2],[5,2,2],[3,1,3],[4,1,3],[5,1,3],[3,2,3],[4,2,3],[5,2,3],[3,1,4],[4,1,4],[5,1,4],[3,2,4],[4,2,4],[5,2,4],[3,1,5],[4,1,5],[5,1,5],[3,2,5],[4,2,5],[5,2,5],[6,1,8],[6,2,8],[1,6,9],[2,6,9]]
            prior_node_dist = torch.distributions.normal.Normal(torch.zeros([self.node_dim]).cuda(),
                                                                temperature * torch.ones([self.node_dim]).cuda())
            prior_edge_dist = torch.distributions.normal.Normal(torch.zeros([self.bond_dim]).cuda(),
                                                                temperature * torch.ones([self.bond_dim]).cuda())

            cur_node_features = torch.zeros([1, max_atoms, self.node_dim]).cuda()
            cur_adj_features = torch.zeros([1, self.bond_dim, max_atoms, max_atoms]).cuda()

            original_graph = ""
            mol = None
            # mol_size = mol.GetNumAtoms()

            is_continue = True
            total_resample = 0
            each_node_resample = np.zeros([max_atoms])
            node_cnt = 0
            edge_cnt = 0
            for i in range(max_atoms):

                if not is_continue:
                    break
                if i < self.edge_unroll:  # max edge to model for each node in bfs order.
                    edge_total = i  # edge to sample for current node
                    start = 0
                else:
                    edge_total = self.edge_unroll
                    start = i - self.edge_unroll

                if i == 0:
                    cur_node_features[0,0,0] =1.0
                    feature_id = 0
                    original_graph = original_graph + " *" + num2atom[0]
                    node_cnt += 1
                else:
                    latent_node = prior_node_dist.sample().view(1,-1)
                    if self.dp:
                        latent_node = self.flow_core.module.reverse(cur_node_features, cur_adj_features,latent_node, mode=0).view(-1)
                    else:
                        latent_node = self.flow_core.reverse(cur_node_features, cur_adj_features,latent_node, mode=0).view(-1)
                    ## node/adj postprocessing
                    # print(latent_node.shape) #(38, 9)
                    feature_id = torch.argmax(latent_node).item()
                    # print(latent_node.size())#torch.Size([6])
                    # print(num2symbol[feature_id])
                    cur_node_features[0, i, feature_id] = 1.0
                    # cur_adj_features[0, :, i, i] = 1.0
                    # rw_mol.AddAtom(Chem.Atom(num2atom[feature_id]))
                    original_graph = original_graph + " *" +num2atom[feature_id]
                    node_cnt += 1

                if i == 0:
                    is_connect = True
                else:
                    is_connect = False
                # cur_mol_size = mol.GetNumAtoms

                for j in range(edge_total):
                    # print("edge_total "+str(edge_total))
                    valid = False
                    resample_edge = 0
                    invalid_bond_type_set = set()
                    while not valid:
                        # TODO: add cache. Some atom can not get the right edge type and is stuck in the loop
                        # TODO: add cache. invalid bond set

                        if len(invalid_bond_type_set) < 9 and resample_edge <= 100:  # haven't sampled all possible bond type or is not stuck in the loop
                            latent_edge = prior_edge_dist.sample().view(1,-1)
                            if self.dp:
                                latent_edge = self.flow_core.module.reverse(cur_node_features, cur_adj_features,latent_edge,mode=1, edge_index=torch.Tensor([[j + start, i]]).long().cuda()).view(-1)
                            else:
                                latent_edge = self.flow_core.reverse(cur_node_features, cur_adj_features, latent_edge,mode=1, edge_index=torch.Tensor([[j + start, i]]).long().cuda()).view(-1)  # (4, )
                            edge_discrete_id = torch.argmax(latent_edge).item()
                            if edge_discrete_id == 9:
                                sorted_indices = torch.argsort(latent_edge, descending=True)

                                edge_discrete_id = sorted_indices[1].item()
                        else:
                            # if not mute:
                            #     print('have tried all possible bond type, use virtual bond.')
                            assert resample_edge > 100 or len(invalid_bond_type_set) == 9
                            edge_discrete_id = 9

                        if edge_discrete_id == 9:
                            valid = True
                        else:
                            edge_cnt += 1
                            for x1 in range(0,6):
                                if cur_node_features[0,i,x1] == 1:
                                    node1 = x1
                            for x2 in range(0,6):
                                if cur_node_features[0,j+start,x2] == 1:
                                    node2 = x2
                            edge_list = [node1+1,node2+1,edge_discrete_id+1]
                            if edge_list in reasonal_list:
                                check_flag = self.rule_check(i,j+start,edge_list,cur_adj_features)
                                rechoose_cnt = 1
                                while (not check_flag) and (rechoose_cnt != 9):
                                    rechoose_cnt += 1
                                    total_resample += 1.0
                                    each_node_resample[i] += 1.0
                                    resample_edge += 1.0
                                    invalid_bond_type_set.add(edge_discrete_id)
                                    top_values, top_indices = torch.topk(latent_edge[:-1], k=9, largest=True)
                                    edge_discrete_id = top_indices[rechoose_cnt-1].item()
                                    edge_list[2] = edge_discrete_id+1
                                    check_flag = self.rule_check(i,j+start,edge_list,cur_adj_features)
                                if check_flag == False:
                                    edge_discrete_id = 9
                                    edge_cnt -= 1
                                    valid = True
                                else:
                                    if ((" " + str(j+start) + " " + str(i) + " " + num2bond[edge_discrete_id]) or (" " + str(i) + " " + str(j+start) + " " + num2bond[edge_discrete_id])) not in (original_graph):
                                        if ([node1+1,node2+1,edge_discrete_id+1]) in reasonal_list_right:
                                            if ([node1+1,node2+1,edge_discrete_id+1] in [[1,1,6],[1,2,6],[2,1,6],[2,2,6],[1,1,4],[1,2,4],[2,1,4],[2,2,4],[1,1,7],[1,2,7],[2,1,7],[2,2,7]]) and (i > j+start):
                                                original_graph = original_graph + " " + str(j + start) + " " + str(i) + " " + num2bond[edge_discrete_id]
                                                # print(str(j+start)+str(i))
                                            elif (j+start == 0) and (edge_discrete_id != 8):
                                                original_graph = original_graph + " " + str(j + start) + " " + str(i) + " " + num2bond[edge_discrete_id]
                                            else:
                                                original_graph = original_graph + " " + str(i) + " " + str(j + start) + " " + num2bond[edge_discrete_id]
                                                # print( str(i) + " " + str(j + start) + " " + num2bond[edge_discrete_id])
                                        elif ([node1+1,node2+1,edge_discrete_id+1]) in reasonal_list_reverse:
                                            original_graph = original_graph + " " + str(j+start) + " " + str(i) + " " + num2bond[edge_discrete_id]
                                    valid = True
                                    is_connect = True
                                cur_adj_features[0, edge_discrete_id, i, j + start] = 1.0
                                cur_adj_features[0, edge_discrete_id, j + start, i] = 1.0
                            else:
                                edge_cnt -= 1
                                # cur_adj_features[0, edge_discrete_id, i, j + start] = 0.0
                                # cur_adj_features[0, edge_discrete_id, j + start, i] = 0.0
                                total_resample += 1.0
                                each_node_resample[i] += 1.0
                                resample_edge += 1
                                invalid_bond_type_set.add(edge_discrete_id)

                if is_connect:  # new generated node has at least one bond with previous node, do not stop generation, backup mol from rw_mol to mol
                    is_continue = True
                else:
                    is_continue = False

            # mol = rw_mol.GetMol() # mol backup
            assert original_graph is not None, 'mol is None...'

            final_valid = True
            assert final_valid is True, 'warning: use valency check during generation but the final molecule is invalid!!!'

            pure_valid = 0
            if total_resample == 0:
                pure_valid = 1.0
            if not mute:
                cnt = str(cnt) if cnt is not None else ''
                print('ASG%s: %s | #atoms: %d | #bonds: %d | #resample: %.5f | time: %.5f |' % (
                cnt, original_graph, node_cnt, edge_cnt, total_resample, time() - generate_start_t))

            return original_graph, pure_valid, node_cnt

    # TODO: batched generation, current useless
    def batch_generate(self, batch_size=100, temperature=0.75, mute=False):
        """
        inverse flow to generate one batch molecules
        Args:
            temp: temperature of normal distributions, we sample from (0, temp^2 * I)
            mute: do not print some messages
        Returns:
            generated mol represented by smiles, valid rate (without check)
        """

        generate_start_t = time()
        with torch.no_grad():
            num2bond = {0: Chem.rdchem.BondType.SINGLE, 1: Chem.rdchem.BondType.DOUBLE, 2: Chem.rdchem.BondType.TRIPLE}
            num2bond_symbol = {0: '=', 1: '==', 2: '==='}
            # [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
            num2atom = {0: 6, 1: 7, 2: 8, 3: 9, 4: 15, 5: 16, 6: 17, 7: 35, 8: 53}
            num2symbol = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'P', 5: 'S', 6: 'Cl', 7: 'Br', 8: 'I'}

            prior_node_dist = torch.distributions.normal.Normal(torch.zeros([self.node_dim]).cuda(),
                                                                temperature * torch.ones([self.node_dim]).cuda())
            prior_edge_dist = torch.distributions.normal.Normal(torch.zeros([self.bond_dim]).cuda(),
                                                                temperature * torch.ones([self.bond_dim]).cuda())

            cur_node_features = torch.zeros([batch_size, self.max_size, self.node_dim]).cuda()
            cur_adj_features = torch.zeros([batch_size, self.bond_dim, self.max_size, self.max_size]).cuda()

            rw_mol = Chem.RWMol()  # editable mol
            mol = None
            # mol_size = mol.GetNumAtoms()

            is_continue = [True] * batch_size
            total_resample = [0] * batch_size
            each_node_resample = np.zeros([batch_size, self.max_size])
            for i in range(self.max_size):
                if not is_continue:
                    break
                if i < self.edge_unroll:
                    edge_total = i  # edge to sample for current node
                    start = 0
                else:
                    edge_total = self.edge_unroll
                    start = i - self.edge_unroll
                # first generate node
                ## reverse flow
                a.sample(sample_shape=torch.Size([20]))
                latent_node = prior_node_dist.sample().view(1, -1)  # (1, 9)
                if self.dp:
                    latent_node = self.flow_core.module.reverse(cur_node_features, cur_adj_features,
                                                                latent_node, mode=0).view(-1)  # (9, )
                else:
                    latent_node = self.flow_core.reverse(cur_node_features, cur_adj_features,
                                                         latent_node, mode=0).view(-1)  # (9, )
                ## node/adj postprocessing
                # print(latent_node.shape) #(38, 9)
                feature_id = torch.argmax(latent_node).item()
                # print(num2symbol[feature_id])
                cur_node_features[0, i, feature_id] = 1.0
                cur_adj_features[0, :, i, i] = 1.0
                rw_mol.AddAtom(Chem.Atom(num2atom[feature_id]))

                # then generate edges
                if i == 0:
                    is_connect = True
                else:
                    is_connect = False
                # cur_mol_size = mol.GetNumAtoms
                for j in range(edge_total):
                    valid = False
                    resample_edge = 0
                    invalid_bond_type_set = set()
                    while not valid:
                        # TODO: add cache. Some atom can not get the right edge type and is stuck in the loop
                        # TODO: add cache. invalid bond set
                        if len(invalid_bond_type_set) < 3 and resample_edge <= 50:  # haven't sampled all possible bond type or is not stuck in the loop
                            latent_edge = prior_edge_dist.sample().view(1, -1)  # (1, 4)
                            if self.dp:
                                latent_edge = self.flow_core.module.reverse(cur_node_features, cur_adj_features,
                                                                            latent_edge,
                                                                            mode=1, edge_index=torch.Tensor(
                                        [[j + start, i]]).long().cuda()).view(-1)  # (4, )
                            else:
                                latent_edge = self.flow_core(cur_node_features, cur_adj_features, latent_edge,
                                                             mode=1, edge_index=torch.Tensor(
                                        [[j + start, i]]).long().cuda()).view(-1)  # (4, )
                            edge_discrete_id = torch.argmax(latent_edge).item()
                        else:
                            if not mute:
                                print('have tried all possible bond type, use virtual bond.')
                            assert resample_edge > 50 or len(invalid_bond_type_set) == 3
                            edge_discrete_id = 3  # we have no choice but to choose not to add edge between (i, j+start)
                        cur_adj_features[0, edge_discrete_id, i, j + start] = 1.0
                        cur_adj_features[0, edge_discrete_id, j + start, i] = 1.0
                        if edge_discrete_id == 3:  # virtual edge
                            valid = True
                        else:  # single/double/triple bond
                            rw_mol.AddBond(i, j + start, num2bond[edge_discrete_id])
                            # TODO: check valency
                            valid = env.check_valency(rw_mol)
                            if valid:
                                is_connect = True
                                # print(num2bond_symbol[edge_discrete_id])
                            else:  # backtrack
                                rw_mol.RemoveBond(i, j + start)
                                cur_adj_features[0, edge_discrete_id, i, j + start] = 0.0
                                cur_adj_features[0, edge_discrete_id, j + start, i] = 0.0
                                total_resample += 1.0
                                each_node_resample[i] += 1.0
                                resample_edge += 1

                                invalid_bond_type_set.add(edge_discrete_id)
                if is_connect:  # new generated node has at least one bond with previous node, do not stop generation, backup mol from rw_mol to mol
                    is_continue = True
                    mol = rw_mol.GetMol()

                else:
                    is_continue = False

            # mol = rw_mol.GetMol() # mol backup
            assert mol is not None, 'mol is None...'
            mol = env.convert_radical_electrons_to_hydrogens(mol)
            num_atoms = mol.GetNumAtoms()
            num_bonds = mol.GetNumBonds()

            smiles = Chem.MolToSmiles(mol)
            assert '.' not in smiles, 'warning: use is_connect to check stop action, but the final molecule is disconnected!!!'

            final_valid = env.check_valency(mol)
            assert final_valid is True, 'warning: use valency check during generation but the final molecule is invalid!!!'

            pure_valid = 0
            if total_resample == 0:
                pure_valid = 1.0
            if not mute:
                print('smiles: %s | #atoms: %d | #bonds: %d | #resample: %.5f | time: %.5f |' % (
                smiles, num_atoms, num_bonds, total_resample, time() - generate_start_t))
            return smiles, pure_valid

    def initialize_masks(self, max_node_unroll=38, max_edge_unroll=10):
        """
        Args:
            max node unroll: maximal number of nodes in molecules to be generated (default: 38)
            max edge unroll: maximal number of edges to predict for each generated nodes (default: 12, calculated from zink250K data)
        Returns:
            node_masks: node mask for each step
            adj_masks: adjacency mask for each step
            is_node_update_mask: 1 indicate this step is for updating node features
            flow_core_edge_mask: get the distributions we want to model in adjacency matrix
        """
        num_masks = int(
            max_node_unroll + (max_edge_unroll - 1) * max_edge_unroll / 2 + (max_node_unroll - max_edge_unroll) * (
                max_edge_unroll))
        num_mask_edge = int(num_masks - max_node_unroll)

        node_masks1 = torch.zeros([max_node_unroll, max_node_unroll]).byte()
        adj_masks1 = torch.zeros([max_node_unroll, max_node_unroll, max_node_unroll]).byte()
        node_masks2 = torch.zeros([num_mask_edge, max_node_unroll]).byte()
        adj_masks2 = torch.zeros([num_mask_edge, max_node_unroll, max_node_unroll]).byte()
        # is_node_update_masks = torch.zeros([num_masks]).byte()

        link_prediction_index = torch.zeros([num_mask_edge, 2]).long()

        flow_core_edge_masks = torch.zeros([max_node_unroll, max_node_unroll]).byte()

        # masks_edge = dict()
        cnt = 0
        cnt_node = 0
        cnt_edge = 0
        for i in range(max_node_unroll):
            node_masks1[cnt_node][:i] = 1
            adj_masks1[cnt_node][:i, :i] = 1
            # is_node_update_masks[cnt] = 1
            cnt += 1
            cnt_node += 1

            edge_total = 0
            if i < max_edge_unroll:
                start = 0
                edge_total = i
            else:
                start = i - max_edge_unroll
                edge_total = max_edge_unroll
            for j in range(edge_total):
                if j == 0:
                    node_masks2[cnt_edge][:i + 1] = 1
                    adj_masks2[cnt_edge] = adj_masks1[cnt_node - 1].clone()
                    adj_masks2[cnt_edge][i, i] = 1
                else:
                    node_masks2[cnt_edge][:i + 1] = 1
                    adj_masks2[cnt_edge] = adj_masks2[cnt_edge - 1].clone()
                    adj_masks2[cnt_edge][i, start + j - 1] = 1
                    adj_masks2[cnt_edge][start + j - 1, i] = 1
                cnt += 1
                cnt_edge += 1
        assert cnt == num_masks, 'masks cnt wrong'
        assert cnt_node == max_node_unroll, 'node masks cnt wrong'
        assert cnt_edge == num_mask_edge, 'edge masks cnt wrong'

        cnt = 0
        for i in range(max_node_unroll):
            if i < max_edge_unroll:
                start = 0
                edge_total = i
            else:
                start = i - max_edge_unroll
                edge_total = max_edge_unroll

            for j in range(edge_total):
                link_prediction_index[cnt][0] = start + j
                link_prediction_index[cnt][1] = i
                cnt += 1
        assert cnt == num_mask_edge, 'edge mask initialize fail'

        for i in range(max_node_unroll):
            if i == 0:
                continue
            if i < max_edge_unroll:
                start = 0
                end = i
            else:
                start = i - max_edge_unroll
                end = i
            flow_core_edge_masks[i][start:end] = 1

        node_masks = torch.cat((node_masks1, node_masks2), dim=0)
        adj_masks = torch.cat((adj_masks1, adj_masks2), dim=0)

        node_masks = nn.Parameter(node_masks, requires_grad=False)
        adj_masks = nn.Parameter(adj_masks, requires_grad=False)
        link_prediction_index = nn.Parameter(link_prediction_index, requires_grad=False)
        flow_core_edge_masks = nn.Parameter(flow_core_edge_masks, requires_grad=False)

        return node_masks, adj_masks, link_prediction_index, flow_core_edge_masks

    def log_prob(self, z, logdet, deq_logp=None, deq_logdet=None):

        # TODO: check multivariate gaussian log_prob formula
        logdet[0] = logdet[
                        0] - self.latent_node_length  # calculate probability of a region from probability density, minus constant has no effect on optimization
        logdet[1] = logdet[
                        1] - self.latent_edge_length  # calculate probability of a region from probability density, minus constant has no effect on optimization

        ll_node = -1 / 2 * (
                    torch.log(2 * self.constant_pi) + self.prior_ln_var + torch.exp(-self.prior_ln_var) * (z[0] ** 2))
        ll_node = ll_node.sum(-1)  # (B)

        ll_edge = -1 / 2 * (
                    torch.log(2 * self.constant_pi) + self.prior_ln_var + torch.exp(-self.prior_ln_var) * (z[1] ** 2))
        ll_edge = ll_edge.sum(-1)  # (B)

        ll_node += logdet[0]  # ([B])
        ll_edge += logdet[1]  # ([B])

        if self.args.deq_type == 'random':
            if self.args.divide_loss:
                return -(torch.mean(ll_node + ll_edge) / (self.latent_edge_length + self.latent_node_length))
            else:
                # ! useless
                return -torch.mean(ll_node + ll_edge)  # scalar

        elif self.args.deq_type == 'variational':
            # TODO: finish this part
            assert deq_logp is not None and deq_logdet is not None, 'variational dequantization requires deq_logp/deq_logdet'
            ll_deq_node = deq_logp[0] - deq_logdet[0]  # ()
            # print(ll_deq_node.size())
            ll_deq_edge = deq_logp[1] - deq_logdet[1]
            # print(ll_deq_edge.size())
            return (torch.mean(ll_node), torch.mean(ll_edge), torch.mean(ll_deq_node), torch.mean(ll_deq_edge))




