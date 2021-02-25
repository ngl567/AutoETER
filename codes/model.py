#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

from dataloader import TestDataset

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, type_dim, gamma, gamma_type, gamma_pair,
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.type_dim = type_dim
        self.epsilon = 2.0
        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )

        self.gamma_type = nn.Parameter(
            torch.Tensor([gamma_type]), 
            requires_grad=False
        )

        self.gamma_pair = nn.Parameter(
            torch.Tensor([gamma_pair]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )

        self.type_embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma_type.item() + self.epsilon) / self.type_dim]), 
            requires_grad=False
        )
        
        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim
        # setting type_dim with the hidden_dim
        # self.type_dim = hidden_dim/5
        
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )

        self.type_embedding = nn.Parameter(torch.zeros(nentity, self.type_dim))
        nn.init.uniform_(
            tensor=self.type_embedding, 
            a=-self.type_embedding_range.item(), 
            b=self.type_embedding_range.item()
        )

        self.reltype_embedding = nn.Parameter(torch.zeros(nrelation, self.type_dim))
        nn.init.uniform_(
            tensor=self.reltype_embedding, 
            a=-self.type_embedding_range.item(), 
            b=self.type_embedding_range.item()
        )

        self.norm_vector_embedding = nn.Parameter(torch.zeros(self.nrelation, self.entity_dim))
        nn.init.uniform_(
            tensor = self.norm_vector_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.norm_vectortype_embedding = nn.Parameter(torch.zeros(self.nrelation, self.type_dim))
        nn.init.uniform_(
            tensor = self.norm_vectortype_embedding,
            a=-self.type_embedding_range.item(),
            b=self.type_embedding_range.item()
        )
        
        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
        
        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE', 'AutoETER']:
            raise ValueError('model %s not supported' % model_name)
            
        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')
        
    def forward(self, sample, mode='single', is_train=True):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''

        if is_train == False:
            score_entity, score_type = self.predict(sample, mode)
            return score_entity, score_type

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=sample[:,1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,2]
            ).unsqueeze(1)

            vector_rel = torch.index_select(
                self.norm_vector_embedding,
                dim=0,
                index=sample[:,1]
            ).unsqueeze(1)

            norm = F.normalize(vector_rel, p = 2, dim = -1)
            head = head - torch.sum(head * norm, -1, True) * norm
            tail = tail - torch.sum(tail * norm, -1, True) * norm

            head_type = torch.index_select(
                self.type_embedding, 
                dim=0, 
                index=sample[:,0]
            ).unsqueeze(1)
            
            relation_type = torch.index_select(
                self.reltype_embedding, 
                dim=0, 
                index=sample[:,1]
            ).unsqueeze(1)
            
            tail_type = torch.index_select(
                self.type_embedding, 
                dim=0, 
                index=sample[:,2]
            ).unsqueeze(1)

            vector_reltype = torch.index_select(
                self.norm_vectortype_embedding,
                dim=0,
                index=sample[:,1]
            ).unsqueeze(1)

            norm_type = F.normalize(vector_reltype, p = 2, dim = -1)
            head_type = head_type - torch.sum(head_type * norm_type, -1, True) * norm_type
            tail_type = tail_type - torch.sum(tail_type * norm_type, -1, True) * norm_type
            
        elif mode == 'head-batch':
            tail_part, head_part, positive_pair_sample, negative_pair_sample = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=tail_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part[:, 2]
            ).unsqueeze(1)

            vector_rel = torch.index_select(
                self.norm_vector_embedding,
                dim=0,
                index=tail_part[:,1]
            ).unsqueeze(1)
            '''
            print("head_batch\n")
            print("head size:{}".format(head.shape))
            print("relation size:{}".format(relation.shape))
            print("tail size:{}".format(tail.shape))
            print("vector_rel size:{}".format(vector_rel.shape))
            '''
            norm = F.normalize(vector_rel, p = 2, dim = -1)
            if head.shape[0] != norm.shape[0]:
                head = head.view(-1, norm.shape[0], head.shape[-1])
                norm = norm.view(-1, norm.shape[0], norm.shape[-1])
                head = head - torch.sum(head * norm, -1, True) * norm
                head = head.view(-1, head.shape[-1])
            else:
                head = head - torch.sum(head * norm, -1, True) * norm

            if tail.shape[0] != norm.shape[0]:
                tail = tail.view(-1, norm.shape[0], tail.shape[-1])
                norm = norm.view(-1, norm.shape[0], norm.shape[-1])
                tail = tail - torch.sum(tail * norm, -1, True) * norm
                tail = tail.view(-1, tail.shape[-1])
            else:
                tail = tail - torch.sum(tail * norm, -1, True) * norm
            '''
            print("head size:{}".format(head.shape))
            print("tail size:{}".format(tail.shape))
            print("norm size:{}".format(vector_rel.shape))
            '''
            head_type = torch.index_select(
                self.type_embedding, 
                dim=0, 
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
            relation_type = torch.index_select(
                self.reltype_embedding, 
                dim=0, 
                index=tail_part[:, 1]
            ).unsqueeze(1)

            # head_type in positive instance for triplet loss
            head_type_pos = torch.index_select(
                self.type_embedding, 
                dim=0, 
                index=tail_part[:, 0]
            ).unsqueeze(1)
            
            tail_type = torch.index_select(
                self.type_embedding, 
                dim=0, 
                index=tail_part[:, 2]
            ).unsqueeze(1)

            vector_reltype = torch.index_select(
                self.norm_vectortype_embedding,
                dim=0,
                index=tail_part[:,1]
            ).unsqueeze(1)
            '''
            print("head_batch\n")
            print("head_type size:{}".format(head_type.shape))
            print("relation_type size:{}".format(relation_type.shape))
            print("tail_type size:{}".format(tail_type.shape))
            print("vector_reltype size:{}".format(vector_reltype.shape))
            '''
            norm_type = F.normalize(vector_reltype, p = 2, dim = -1)
            if head_type.shape[0] != norm_type.shape[0]:
                head_type = head_type.view(-1, norm_type.shape[0], head_type.shape[-1])
                norm_type = norm_type.view(-1, norm_type.shape[0], norm_type.shape[-1])
                head_type = head_type - torch.sum(head_type * norm_type, -1, True) * norm_type
                head_type = head_type.view(-1, head_type.shape[-1])
            else:
                head_type = head_type - torch.sum(head_type * norm_type, -1, True) * norm_type

            if tail_type.shape[0] != norm_type.shape[0]:
                tail_type = tail_type.view(-1, norm_type.shape[0], tail_type.shape[-1])
                norm_type = norm_type.view(-1, norm_type.shape[0], norm_type.shape[-1])
                tail_type = tail_type - torch.sum(tail_type * norm_type, -1, True) * norm_type
                tail_type = tail_type.view(-1, tail_type.shape[-1])
            else:
                tail_type = tail_type - torch.sum(tail_type * norm_type, -1, True) * norm_type

            if head_type_pos.shape[0] != norm_type.shape[0]:
                head_type_pos = head_type_pos.view(-1, norm_type.shape[0], head_type_pos.shape[-1])
                norm_type = norm_type.view(-1, norm_type.shape[0], norm_type.shape[-1])
                head_type_pos = head_type_pos - torch.sum(head_type_pos * norm_type, -1, True) * norm_type
                head_type_pos = head_type_pos.view(-1, head_type_pos.shape[-1])
            else:
                head_type_pos = head_type_pos - torch.sum(head_type_pos * norm_type, -1, True) * norm_type

            '''
            print("head_type size:{}".format(head_type.shape))
            print("tail_type size:{}".format(tail_type.shape))
            print("vector_reltype size:{}".format(vector_reltype.shape))
            '''
            headpair_part = positive_pair_sample
            batch_size, negative_sample_size = headpair_part.size(0), headpair_part.size(1)
            
            positive_head_pair = torch.index_select(
                self.type_embedding, 
                dim=0, 
                index=headpair_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            '''
            print("positive_head_pair size:{}".format(positive_head_pair.shape))
            '''
            norm_type = F.normalize(vector_reltype, p = 2, dim = -1)
            if positive_head_pair.shape[0] != norm_type.shape[0]:
                positive_head_pair = positive_head_pair.view(-1, norm_type.shape[0], positive_head_pair.shape[-1])
                norm_type = norm_type.view(-1, norm_type.shape[0], norm_type.shape[-1])
                positive_head_pair = positive_head_pair - torch.sum(positive_head_pair * norm_type, -1, True) * norm_type
                positive_head_pair = positive_head_pair.view(-1, positive_head_pair.shape[-1])
            else:
                positive_head_pair = positive_head_pair - torch.sum(positive_head_pair * norm_type, -1, True) * norm_type
            '''
            print("positive_head_pair size:{}".format(positive_head_pair.shape))
            '''
            negative_headpair_part = negative_pair_sample
            batch_size, negative_sample_size = negative_headpair_part.size(0), negative_headpair_part.size(1)
            
            negative_head_pair = torch.index_select(
                self.type_embedding, 
                dim=0, 
                index=negative_headpair_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            '''
            print("negative_head_pair size:{}".format(negative_head_pair.shape))
            '''
            norm_type = F.normalize(vector_reltype, p = 2, dim = -1)
            if negative_head_pair.shape[0] != norm_type.shape[0]:
                negative_head_pair = negative_head_pair.view(-1, norm_type.shape[0], negative_head_pair.shape[-1])
                norm_type = norm_type.view(-1, norm_type.shape[0], norm_type.shape[-1])
                negative_head_pair = negative_head_pair - torch.sum(negative_head_pair * norm_type, -1, True) * norm_type
                negative_head_pair = negative_head_pair.view(-1, negative_head_pair.shape[-1])
            else:
                negative_head_pair = negative_head_pair - torch.sum(negative_head_pair * norm_type, -1, True) * norm_type
            '''
            print("negative_head_pair size:{}".format(negative_head_pair.shape))
            '''

        elif mode == 'tail-batch':
            head_part, tail_part, positive_pair_sample, negative_pair_sample = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part[:, 0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            vector_rel = torch.index_select(
                self.norm_vector_embedding,
                dim=0,
                index=head_part[:,1]
            ).unsqueeze(1)
            '''
            print("tail_batch\n")
            print("head size:{}".format(head.shape))
            print("relation size:{}".format(relation.shape))
            print("tail size:{}".format(tail.shape))
            print("vector_rel size:{}".format(vector_rel.shape))
            '''
            norm = F.normalize(vector_rel, p = 2, dim = -1)
            if head.shape[0] != norm.shape[0]:
                head = head.view(-1, norm.shape[0], head.shape[-1])
                norm = norm.view(-1, norm.shape[0], norm.shape[-1])
                head = head - torch.sum(head * norm, -1, True) * norm
                head = head.view(-1, head.shape[-1])
            else:
                head = head - torch.sum(head * norm, -1, True) * norm

            if tail.shape[0] != norm.shape[0]:
                tail = tail.view(-1, norm.shape[0], tail.shape[-1])
                norm = norm.view(-1, norm.shape[0], norm.shape[-1])
                tail = tail - torch.sum(tail * norm, -1, True) * norm
                tail = tail.view(-1, tail.shape[-1])
            else:
                tail = tail - torch.sum(tail * norm, -1, True) * norm
            '''
            print("head size:{}".format(head.shape))
            print("tail size:{}".format(tail.shape))
            print("norm size:{}".format(norm.shape))
            '''
            head_type = torch.index_select(
                self.type_embedding, 
                dim=0, 
                index=head_part[:, 0]
            ).unsqueeze(1)
            
            relation_type = torch.index_select(
                self.reltype_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
 
            # tail_type in positive instance for triplet loss
            tail_type_pos = torch.index_select(
                self.type_embedding, 
                dim=0, 
                index=head_part[:, 2]
            ).unsqueeze(1)
           
            tail_type = torch.index_select(
                self.type_embedding, 
                dim=0, 
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            vector_reltype = torch.index_select(
                self.norm_vectortype_embedding,
                dim=0,
                index=head_part[:,1]
            ).unsqueeze(1)

            norm_type = F.normalize(vector_reltype, p = 2, dim = -1)
            if head_type.shape[0] != norm_type.shape[0]:
                head_type = head_type.view(-1, norm_type.shape[0], head_type.shape[-1])
                norm_type = norm_type.view(-1, norm_type.shape[0], norm_type.shape[-1])
                head_type = head_type - torch.sum(head_type * norm_type, -1, True) * norm_type
                head_type = head_type.view(-1, head_type.shape[-1])
            else:
                head_type = head_type - torch.sum(head_type * norm_type, -1, True) * norm_type

            if tail_type.shape[0] != norm_type.shape[0]:
                tail_type = tail_type.view(-1, norm_type.shape[0], tail_type.shape[-1])
                norm_type = norm_type.view(-1, norm_type.shape[0], norm_type.shape[-1])
                tail_type = tail_type - torch.sum(tail_type * norm_type, -1, True) * norm_type
                tail_type = tail_type.view(-1, tail_type.shape[-1])
            else:
                tail_type = tail_type - torch.sum(tail_type * norm_type, -1, True) * norm_type

            if tail_type_pos.shape[0] != norm_type.shape[0]:
                tail_type_pos = tail_type_pos.view(-1, norm_type.shape[0], tail_type_pos.shape[-1])
                norm_type = norm_type.view(-1, norm_type.shape[0], norm_type.shape[-1])
                tail_type_pos = tail_type_pos - torch.sum(tail_type_pos * norm_type, -1, True) * norm_type
                tail_type_pos = tail_type_pos.view(-1, tail_type_pos.shape[-1])
            else:
                tail_type_pos = tail_type_pos - torch.sum(tail_type_pos * norm_type, -1, True) * norm_type
            '''
            print("tail_batch\n")
            print("head_type size:{}".format(head_type.shape))
            print("relation_type size:{}".format(relation_type.shape))
            print("tail_type size:{}".format(tail_type.shape))
            print("vector_reltype size:{}".format(vector_reltype.shape))
            '''
            tailpair_part = positive_pair_sample
            batch_size, negative_sample_size = tailpair_part.size(0), tailpair_part.size(1)
            
            positive_tail_pair = torch.index_select(
                self.type_embedding, 
                dim=0, 
                index=tailpair_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            '''
            print("positive_tail_pair size:{}".format(positive_tail_pair.shape))
            '''
            norm_type = F.normalize(vector_reltype, p = 2, dim = -1)
            if positive_tail_pair.shape[0] != norm_type.shape[0]:
                positive_tail_pair = positive_tail_pair.view(-1, norm_type.shape[0], positive_tail_pair.shape[-1])
                norm_type = norm_type.view(-1, norm_type.shape[0], norm_type.shape[-1])
                positive_tail_pair = positive_tail_pair - torch.sum(positive_tail_pair * norm_type, -1, True) * norm_type
                positive_tail_pair = positive_tail_pair.view(-1, positive_tail_pair.shape[-1])
            else:
                positive_tail_pair = positive_tail_pair - torch.sum(positive_tail_pair * norm_type, -1, True) * norm_type
            '''
            print("positive_tail_pair size:{}".format(positive_tail_pair.shape))
            '''
            negative_tailpair_part = negative_pair_sample
            batch_size, negative_sample_size = negative_tailpair_part.size(0), negative_tailpair_part.size(1)
            
            negative_tail_pair = torch.index_select(
                self.type_embedding, 
                dim=0, 
                index=negative_tailpair_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            '''
            print("negative_tail_pair size:{}".format(negative_tail_pair.shape))
            '''
            norm_type = F.normalize(vector_reltype, p = 2, dim = -1)
            if negative_tail_pair.shape[0] != norm_type.shape[0]:
                negative_tail_pair = negative_tail_pair.view(-1, norm_type.shape[0], negative_tail_pair.shape[-1])
                norm_type = norm_type.view(-1, norm_type.shape[0], norm_type.shape[-1])
                negative_tail_pair = negative_tail_pair - torch.sum(negative_tail_pair * norm_type, -1, True) * norm_type
                negative_tail_pair = negative_tail_pair.view(-1, negative_tail_pair.shape[-1])
            else:
                negative_tail_pair = negative_tail_pair - torch.sum(negative_tail_pair * norm_type, -1, True) * norm_type
            '''
            print("negative_tail_pair size:{}".format(negative_tail_pair.shape))
            '''
        else:
            raise ValueError('mode %s not supported' % mode)
            
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE,
            'AutoETER': self.AutoETER
        }
        
        if mode == 'single':
            if self.model_name == 'AutoETER':
                score_entity, score_type = model_func[self.model_name](head, relation, tail, head_type, relation_type, tail_type, mode)
                return score_entity, score_type
            elif self.model_name in model_func:
                score = model_func[self.model_name](head, relation, tail, mode)
                return score
            else:
                raise ValueError('model %s not supported' % self.model_name)
        elif mode == 'head-batch':
            if self.model_name == 'AutoETER':
                score_entity, score_type = model_func[self.model_name](head, relation, tail, head_type, relation_type, tail_type, mode)
                score_positive_pair, score_negative_pair = self.type_pair(head_type_pos, positive_head_pair, negative_head_pair)

                return score_entity, score_type, score_positive_pair, score_negative_pair, self.gamma_pair

            elif self.model_name in model_func:
                score = model_func[self.model_name](head, relation, tail, mode)
                return score
            else:
                raise ValueError('model %s not supported' % self.model_name)

        else:
            if self.model_name == 'AutoETER':
                score_entity, score_type = model_func[self.model_name](head, relation, tail, head_type, relation_type, tail_type, mode)
                score_positive_pair, score_negative_pair = self.type_pair(tail_type_pos, positive_tail_pair, negative_tail_pair)

                return score_entity, score_type, score_positive_pair, score_negative_pair, self.gamma_pair

            elif self.model_name in model_func:
                score = model_func[self.model_name](head, relation, tail, mode)
                return score
            else:
                raise ValueError('model %s not supported' % self.model_name)
        
        #return score_entity, score_type

    def predict(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=sample[:,1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,2]
            ).unsqueeze(1)

            vector_rel = torch.index_select(
                self.norm_vector_embedding,
                dim=0,
                index=sample[:,1]
            ).unsqueeze(1)

            norm = F.normalize(vector_rel, p = 2, dim = -1)
            head = head - torch.sum(head * norm, -1, True) * norm
            tail = tail - torch.sum(tail * norm, -1, True) * norm

            head_type = torch.index_select(
                self.type_embedding, 
                dim=0, 
                index=sample[:,0]
            ).unsqueeze(1)
            
            relation_type = torch.index_select(
                self.reltype_embedding, 
                dim=0, 
                index=sample[:,1]
            ).unsqueeze(1)
            
            tail_type = torch.index_select(
                self.type_embedding, 
                dim=0, 
                index=sample[:,2]
            ).unsqueeze(1)

            vector_reltype = torch.index_select(
                self.norm_vectortype_embedding,
                dim=0,
                index=sample[:,1]
            ).unsqueeze(1)

            norm_type = F.normalize(vector_reltype, p = 2, dim = -1)
            head_type = head_type - torch.sum(head_type * norm_type, -1, True) * norm_type
            tail_type = tail_type - torch.sum(tail_type * norm_type, -1, True) * norm_type
            
        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=tail_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part[:, 2]
            ).unsqueeze(1)

            vector_rel = torch.index_select(
                self.norm_vector_embedding,
                dim=0,
                index=tail_part[:,1]
            ).unsqueeze(1)
            '''
            print("head_batch\n")
            print("head size:{}".format(head.shape))
            print("relation size:{}".format(relation.shape))
            print("tail size:{}".format(tail.shape))
            print("vector_rel size:{}".format(vector_rel.shape))
            '''
            norm = F.normalize(vector_rel, p = 2, dim = -1)
            if head.shape[0] != norm.shape[0]:
                head = head.view(-1, norm.shape[0], head.shape[-1])
                norm = norm.view(-1, norm.shape[0], norm.shape[-1])
                head = head - torch.sum(head * norm, -1, True) * norm
                head = head.view(-1, head.shape[-1])
            else:
                head = head - torch.sum(head * norm, -1, True) * norm

            if tail.shape[0] != norm.shape[0]:
                tail = tail.view(-1, norm.shape[0], tail.shape[-1])
                norm = norm.view(-1, norm.shape[0], norm.shape[-1])
                tail = tail - torch.sum(tail * norm, -1, True) * norm
                tail = tail.view(-1, tail.shape[-1])
            else:
                tail = tail - torch.sum(tail * norm, -1, True) * norm
            '''
            print("head size:{}".format(head.shape))
            print("tail size:{}".format(tail.shape))
            print("norm size:{}".format(vector_rel.shape))
            '''
            head_type = torch.index_select(
                self.type_embedding, 
                dim=0, 
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
            relation_type = torch.index_select(
                self.reltype_embedding, 
                dim=0, 
                index=tail_part[:, 1]
            ).unsqueeze(1)
            
            tail_type = torch.index_select(
                self.type_embedding, 
                dim=0, 
                index=tail_part[:, 2]
            ).unsqueeze(1)

            vector_reltype = torch.index_select(
                self.norm_vectortype_embedding,
                dim=0,
                index=tail_part[:,1]
            ).unsqueeze(1)
            '''
            print("head_batch\n")
            print("head_type size:{}".format(head_type.shape))
            print("relation_type size:{}".format(relation_type.shape))
            print("tail_type size:{}".format(tail_type.shape))
            print("vector_reltype size:{}".format(vector_reltype.shape))
            '''
            norm_type = F.normalize(vector_reltype, p = 2, dim = -1)
            if head_type.shape[0] != norm_type.shape[0]:
                head_type = head_type.view(-1, norm_type.shape[0], head_type.shape[-1])
                norm_type = norm_type.view(-1, norm_type.shape[0], norm_type.shape[-1])
                head_type = head_type - torch.sum(head_type * norm_type, -1, True) * norm_type
                head_type = head_type.view(-1, head_type.shape[-1])
            else:
                head_type = head_type - torch.sum(head_type * norm_type, -1, True) * norm_type

            if tail_type.shape[0] != norm_type.shape[0]:
                tail_type = tail_type.view(-1, norm_type.shape[0], tail_type.shape[-1])
                norm_type = norm_type.view(-1, norm_type.shape[0], norm_type.shape[-1])
                tail_type = tail_type - torch.sum(tail_type * norm_type, -1, True) * norm_type
                tail_type = tail_type.view(-1, tail_type.shape[-1])
            else:
                tail_type = tail_type - torch.sum(tail_type * norm_type, -1, True) * norm_type
            '''
            print("head_type size:{}".format(head_type.shape))
            print("tail_type size:{}".format(tail_type.shape))
            print("vector_reltype size:{}".format(vector_reltype.shape))
            '''

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part[:, 0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            vector_rel = torch.index_select(
                self.norm_vector_embedding,
                dim=0,
                index=head_part[:,1]
            ).unsqueeze(1)
            '''
            print("tail_batch\n")
            print("head size:{}".format(head.shape))
            print("relation size:{}".format(relation.shape))
            print("tail size:{}".format(tail.shape))
            print("vector_rel size:{}".format(vector_rel.shape))
            '''
            norm = F.normalize(vector_rel, p = 2, dim = -1)
            if head.shape[0] != norm.shape[0]:
                head = head.view(-1, norm.shape[0], head.shape[-1])
                norm = norm.view(-1, norm.shape[0], norm.shape[-1])
                head = head - torch.sum(head * norm, -1, True) * norm
                head = head.view(-1, head.shape[-1])
            else:
                head = head - torch.sum(head * norm, -1, True) * norm

            if tail.shape[0] != norm.shape[0]:
                tail = tail.view(-1, norm.shape[0], tail.shape[-1])
                norm = norm.view(-1, norm.shape[0], norm.shape[-1])
                tail = tail - torch.sum(tail * norm, -1, True) * norm
                tail = tail.view(-1, tail.shape[-1])
            else:
                tail = tail - torch.sum(tail * norm, -1, True) * norm
            '''
            print("head size:{}".format(head.shape))
            print("tail size:{}".format(tail.shape))
            print("norm size:{}".format(norm.shape))
            '''
            head_type = torch.index_select(
                self.type_embedding, 
                dim=0, 
                index=head_part[:, 0]
            ).unsqueeze(1)
            
            relation_type = torch.index_select(
                self.reltype_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
            
            tail_type = torch.index_select(
                self.type_embedding, 
                dim=0, 
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            vector_reltype = torch.index_select(
                self.norm_vectortype_embedding,
                dim=0,
                index=head_part[:,1]
            ).unsqueeze(1)

            norm_type = F.normalize(vector_reltype, p = 2, dim = -1)
            if head_type.shape[0] != norm_type.shape[0]:
                head_type = head_type.view(-1, norm_type.shape[0], head_type.shape[-1])
                norm_type = norm_type.view(-1, norm_type.shape[0], norm_type.shape[-1])
                head_type = head_type - torch.sum(head_type * norm_type, -1, True) * norm_type
                head_type = head_type.view(-1, head_type.shape[-1])
            else:
                head_type = head_type - torch.sum(head_type * norm_type, -1, True) * norm_type

            if tail_type.shape[0] != norm_type.shape[0]:
                tail_type = tail_type.view(-1, norm_type.shape[0], tail_type.shape[-1])
                norm_type = norm_type.view(-1, norm_type.shape[0], norm_type.shape[-1])
                tail_type = tail_type - torch.sum(tail_type * norm_type, -1, True) * norm_type
                tail_type = tail_type.view(-1, tail_type.shape[-1])
            else:
                tail_type = tail_type - torch.sum(tail_type * norm_type, -1, True) * norm_type
            '''
            print("tail_batch\n")
            print("head_type size:{}".format(head_type.shape))
            print("relation_type size:{}".format(relation_type.shape))
            print("tail_type size:{}".format(tail_type.shape))
            print("vector_reltype size:{}".format(vector_reltype.shape))
            '''
            
        else:
            raise ValueError('mode %s not supported' % mode)
            
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE,
            'AutoETER': self.AutoETER
        }
        
        if self.model_name == 'AutoETER':
            score_entity, score_type = model_func[self.model_name](head, relation, tail, head_type, relation_type, tail_type, mode)
            return score_entity, score_type
        elif self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
            return score
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        #return score_entity, score_type
    
    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)
        return score

    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846
        
        #Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head/(self.embedding_range.item()/pi)
        phase_relation = relation/(self.embedding_range.item()/pi)
        phase_tail = tail/(self.embedding_range.item()/pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)            
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim = 2) * self.modulus
        return score

    def AutoETER(self, head, relation, tail, head_type, relation_type, tail_type, mode):
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score_entity = torch.stack([re_score, im_score], dim = 0)
        score_entity = score_entity.norm(dim = 0)

        score_entity = self.gamma.item() - score_entity.sum(dim = 2)

        if mode == 'head-batch':
            score_type = head_type + (relation_type - tail_type)
        else:
            score_type = (head_type + relation_type) - tail_type

        score_type = self.gamma_type.item() - torch.norm(score_type, p=1, dim=2)
        '''
        print("AutoETER: \n score_entity:\t {} \n score_type:\t {}".format(score_entity, score_type))
        '''
        return score_entity, score_type

    def type_pair(self, ent_type, positive_pair, negative_pair):
        score_positive = ent_type - positive_pair
        score_negative = ent_type - negative_pair

        score_positive = torch.norm(score_positive, p=1, dim=2)
        score_negative = torch.norm(score_negative, p=1, dim=2)
        '''
        print("type_pair:\nscore_positive:\t {} \n score_negative:\t {}".format(score_positive, score_negative))
        '''
        return score_positive, score_negative
    
    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''
#        print("\nTrain set start!\n")
        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode, positive_pair_sample, negative_pair_sample = next(train_iterator)
        '''
        print("positive_sample size:\n{}".format(positive_sample.shape))
        print("negative_sample size:\n{}".format(negative_sample.shape))
        print("mode: {}".format(mode))
        print("positive_pair_sample size:\n{}".format(positive_pair_sample.shape))
        print("negative_pair_sample size:\n{}".format(negative_pair_sample.shape))
        '''
        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()
            positive_pair_sample = positive_pair_sample.cuda()
            negative_pair_sample = negative_pair_sample.cuda()

        #print("\nTraining data preparation finished!\n")

        negative_score_entity, negative_score_type, positive_score_pair, negative_score_pair, gamma_pair = model((positive_sample, negative_sample, positive_pair_sample, negative_pair_sample), mode=mode, is_train=True)
        
        #print("\nNegative score finished!\n")

        positive_score_entity, positive_score_type = model(positive_sample, is_train=True)
        positive_score_entity = F.logsigmoid(positive_score_entity).squeeze(dim = 1)
        positive_score_type = F.logsigmoid(positive_score_type).squeeze(dim = 1)

        #print("\n Positive score finished!\n")

        if args.negative_adversarial_sampling:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score_entity = (F.softmax(negative_score_entity * args.adversarial_temperature, dim = 1).detach() 
                              * F.logsigmoid(-negative_score_entity)).sum(dim = 1)
            negative_score_type = (F.softmax(negative_score_type * args.adversarial_temperature, dim = 1).detach() 
                              * F.logsigmoid(-negative_score_type)).sum(dim = 1)

            margin_loss_pair = (F.softmax(negative_score_pair * args.adversarial_temperature, dim = -1).detach() 
                              * torch.max(positive_score_pair - negative_score_pair, -gamma_pair)).sum(dim = -1).mean() + gamma_pair

        else:
            negative_score_entity = F.logsigmoid(-negative_score_entity).mean(dim = 1)
            negative_score_type = F.logsigmoid(-negative_score_type).mean(dim = 1)
            margin_loss_pair = torch.max(positive_score_pair - negative_score_pair, -gamma_pair)
            margin_loss_pair = margin_loss_pair.sum(dim = -1).mean() + gamma_pair

        #print("\nSigmoid score and pair loss!\n")

        if args.uni_weight:
            positive_sample_loss = - positive_score_entity.mean()
            negative_sample_loss = - negative_score_entity.mean() 
            positive_sample_loss_type = - positive_score_type.mean()
            negative_sample_loss_type = - negative_score_type.mean() 
        else:
            positive_sample_loss = - (subsampling_weight * positive_score_entity).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score_entity).sum()/subsampling_weight.sum()
            positive_sample_loss_type = - (subsampling_weight * positive_score_type).sum()/subsampling_weight.sum()
            negative_sample_loss_type = - (subsampling_weight * negative_score_type).sum()/subsampling_weight.sum()

        #print("\nentity and type loss!\n")

        loss = (positive_sample_loss + negative_sample_loss + args.alpha_1 * (positive_sample_loss_type + negative_sample_loss_type) + args.alpha_2 * margin_loss_pair) / 5.0
        
        if args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_embedding.norm(p = 3)**3 + 
                model.relation_embedding.norm(p = 3).norm(p = 3)**3 + 
                model.type_embedding.norm(p = 3).norm(p = 3)**3 + 
                model.reltype_embedding.norm(p = 3).norm(p = 3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        #print("\nbackward started!\n")
            
        loss.backward()

        #print("\nbackward finished!\n")

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'positive_type_loss': positive_sample_loss_type.item(),
            'negative_type_loss': negative_sample_loss_type.item(),
            'margin_loss_pair': margin_loss_pair.item(),
            'loss': loss.item()
        }

        return log
    
    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        model.eval()
        
        if args.countries:
            #Countries S* datasets are evaluated on AUC-PR
            #Process test data for AUC-PR evaluation
            sample = list()
            y_true  = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            #average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}
            
        else:
            #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            #Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'head-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'tail-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )
            
            test_dataset_list = [test_dataloader_head, test_dataloader_tail]
            
            logs = []

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()
                            #positive_pair_sample = positive_pair_sample.cuda()
                            #negative_pair_sample = negative_pair_sample.cuda()

                        batch_size = positive_sample.size(0)

                        score_entity, score_type = model((positive_sample, negative_sample), mode, is_train=False)
                        score = (score_entity + args.alpha_1 * score_type)/2
                        score += filter_bias

                        #Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim = 1, descending=True)

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            #Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

                            #ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            logs.append({
                                'MRR': 1.0/ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics
