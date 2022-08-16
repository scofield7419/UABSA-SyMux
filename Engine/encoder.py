# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SyntaxAwareGCN(nn.Module):
    """
    Simple GCN layer
    """
    def __init__(self, dep_dim, in_features, out_features, pos_dim=None, bias=True):
        super(SyntaxAwareGCN, self).__init__()
        self.dep_dim = dep_dim
        self.pos_dim = pos_dim
        self.in_features = in_features
        self.out_features = out_features

        self.dep_attn = nn.Linear(dep_dim + pos_dim + in_features, out_features)
        self.dep_fc = nn.Linear(dep_dim, out_features)
        self.pos_fc = nn.Linear(pos_dim, out_features)

    def forward(self, text, adj, dep_embed, pos_embed=None):
        """

        :param text: [batch size, seq_len, feat_dim]
        :param adj: [batch size, seq_len, seq_len]
        :param dep_embed: [batch size, seq_len, seq_len, dep_type_dim]
        :param pos_embed: [batch size, seq_len, pos_dim]
        :return: [batch size, seq_len, feat_dim]
        """
        batch_size, seq_len, feat_dim = text.shape

        val_us = text.unsqueeze(dim=2)
        val_us = val_us.repeat(1, 1, seq_len, 1)
        pos_us = pos_embed.unsqueeze(dim=2).repeat(1, 1, seq_len, 1)
        # [batch size, seq_len, seq_len, feat_dim+pos_dim+dep_dim]
        val_sum = torch.cat([val_us, pos_us, dep_embed], dim=-1)

        r = self.dep_attn(val_sum)

        p = torch.sum(r, dim=-1)
        mask = (adj == 0).float() * (-1e30)
        p = p + mask
        p = torch.softmax(p, dim=2)
        p_us = p.unsqueeze(3).repeat(1, 1, 1, feat_dim)

        output = val_us + self.pos_fc(pos_us) + self.dep_fc(dep_embed)
        output = torch.mul(p_us, output)

        output_sum = torch.sum(output, dim=2)

        return r, output_sum, p


class SGCN(nn.Module):
    def __init__(self, opt):
        super(SGCN, self).__init__()
        self.opt = opt
        self.model = nn.ModuleList([SyntaxAwareGCN(opt.dep_dim, opt.bert_dim,
                                                  opt.bert_dim, opt.pos_dim)
                                    for i in range(self.opt.num_layer)])
        self.dep_embedding = nn.Embedding(opt.dep_num, opt.dep_dim, padding_idx=0)

    def forward(self, x, simple_graph, graph, pos_embed=None, output_attention=False):

        dep_embed = self.dep_embedding(graph)

        attn_list = []
        for lagcn in self.model:
            r, x, attn = lagcn(x, simple_graph, dep_embed, pos_embed=pos_embed)
            attn_list.append(attn)

        if output_attention is True:
            return x, r, attn_list
        else:
            return x, r


class SyMuxEncoder(nn.Module):
    def __init__(self, bert, opt):
        super(SyMuxEncoder, self).__init__()
        self.opt = opt
        self.bert = bert
        self.sgcn = SGCN(opt)

        self.fc = nn.Linear(opt.bert_dim*2 + opt.pos_dim, opt.bert_dim*2)
        self.bert_dropout = nn.Dropout(opt.bert_dropout)
        self.output_dropout = nn.Dropout(opt.output_dropout)

        self.pod_embedding = nn.Embedding(opt.pos_num, opt.pos_dim, padding_idx=0)

    def forward(self, input_ids, input_masks, simple_graph, graph, pos=None, output_attention=False):

        pos_embed = self.pod_embedding(pos)
        sequence_output, pooled_output = self.bert(input_ids)
        x = self.bert_dropout(sequence_output)

        lagcn_output = self.sgcn(x, simple_graph, graph, pos_embed, output_attention)

        pos_output = self.local_attn(x, pos_embed, self.opt.num_layer, self.opt.w_size)

        output = torch.cat((lagcn_output[0], pos_output, sequence_output), dim=-1)
        output = self.fc(output)
        output = self.output_dropout(output)
        return output, lagcn_output[1]

    def local_attn(self, x, pos_embed, num_layer, w_size):
        """

        :param x:
        :param pos_embed:
        :return:
        """
        batch_size, seq_len, feat_dim = x.shape
        pos_dim = pos_embed.size(-1)
        output = pos_embed
        for i in range(num_layer):
            val_sum = torch.cat([x, output], dim=-1)  # [batch size, seq_len, feat_dim+pos_dim]
            attn = torch.matmul(val_sum, val_sum.transpose(1, 2))  # [batch size, seq_len, seq_len]
            # pad size = seq_len + (window_size - 1) // 2 * 2
            pad_size = seq_len + w_size * 2
            mask = torch.zeros((batch_size, seq_len, pad_size), dtype=torch.float).to(
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            for i in range(seq_len):
                mask[:, i, i:i + w_size] = 1.0
            pad_attn = torch.full((batch_size, seq_len, w_size), -1e18).to(
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            attn = torch.cat([pad_attn, attn, pad_attn], dim=-1)
            local_attn = torch.softmax(torch.mul(attn, mask), dim=-1)
            local_attn = local_attn[:, :, w_size:pad_size - w_size]  # [batch size, seq_len, seq_len]
            local_attn = local_attn.unsqueeze(dim=3).repeat(1, 1, 1, pos_dim)
            output = output.unsqueeze(dim=2).repeat(1, 1, seq_len, 1)
            output = torch.sum(torch.mul(output, local_attn), dim=2)  # [batch size, seq_len, pos_dim]
        return output