
"""
The MIT License

Copyright (c) 2021 MatNet

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


from ATSPModel_LIB import AddAndInstanceNormalization, FeedForward, MixedScore_MultiHeadAttention


class ATSPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = ATSP_Encoder(**model_params)
        self.decoder = ATSP_Decoder(**model_params)

        self.encoded_row = None
        self.encoded_col = None
        # shape: (batch, node, embedding)

    def pre_forward(self, reset_state):

        problems = reset_state.problems
        # problems.shape: (batch, node, node)

        batch_size = problems.size(0)
        node_cnt = problems.size(1)
        embedding_dim = self.model_params['embedding_dim']

        row_emb = torch.zeros(size=(batch_size, node_cnt, embedding_dim))
        # emb.shape: (batch, node, embedding)
        col_emb = torch.zeros(size=(batch_size, node_cnt, embedding_dim))
        # shape: (batch, node, embedding)

        seed_cnt = self.model_params['one_hot_seed_cnt']
        rand = torch.rand(batch_size, seed_cnt)
        batch_rand_perm = rand.argsort(dim=1)
        rand_idx = batch_rand_perm[:, :node_cnt]

        b_idx = torch.arange(batch_size)[:, None].expand(batch_size, node_cnt)
        n_idx = torch.arange(node_cnt)[None, :].expand(batch_size, node_cnt)
        col_emb[b_idx, n_idx, rand_idx] = 1
        # shape: (batch, node, embedding)

        self.encoded_row, self.encoded_col = self.encoder(row_emb, col_emb, problems)
        # encoded_nodes.shape: (batch, node, embedding)

        self.decoder.set_kv(self.encoded_col)

    def forward(self, state):

        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.current_node is None:
            selected = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))

            # encoded_rows_mean = self.encoded_row.mean(dim=1, keepdim=True)
            # encoded_cols_mean = self.encoded_col.mean(dim=1, keepdim=True)
            # # shape: (batch, 1, embedding)
            encoded_first_row = _get_encoding(self.encoded_row, selected)
            # shape: (batch, pomo, embedding)
            self.decoder.set_q1(encoded_first_row)

        else:
            encoded_current_row = _get_encoding(self.encoded_row, state.current_node)
            # shape: (batch, pomo, embedding)
            all_job_probs = self.decoder(encoded_current_row, ninf_mask=state.ninf_mask)
            # shape: (batch, pomo, job)

            if self.training or self.model_params['eval_type'] == 'softmax':
                while True:  # to fix pytorch.multinomial bug on selecting 0 probability elements
                    with torch.no_grad():
                        selected = all_job_probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                            .squeeze(dim=1).reshape(batch_size, pomo_size)
                        # shape: (batch, pomo)

                    prob = all_job_probs[state.BATCH_IDX, state.POMO_IDX, selected] \
                        .reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)

                    if (prob != 0).all():
                        break
            else:
                selected = all_job_probs.argmax(dim=2)
                # shape: (batch, pomo)
                prob = None

        return selected, prob


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes


########################################
# ENCODER
########################################
class ATSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        encoder_layer_num = model_params['encoder_layer_num']
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, row_emb, col_emb, cost_mat):
        # col_emb.shape: (batch, col_cnt, embedding)
        # row_emb.shape: (batch, row_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)

        for layer in self.layers:
            row_emb, col_emb = layer(row_emb, col_emb, cost_mat)

        return row_emb, col_emb


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.row_encoding_block = EncodingBlock(**model_params)
        self.col_encoding_block = EncodingBlock(**model_params)

    def forward(self, row_emb, col_emb, cost_mat):
        # row_emb.shape: (batch, row_cnt, embedding)
        # col_emb.shape: (batch, col_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        row_emb_out = self.row_encoding_block(row_emb, col_emb, cost_mat)
        col_emb_out = self.col_encoding_block(col_emb, row_emb, cost_mat.transpose(1, 2))

        return row_emb_out, col_emb_out


class EncodingBlock(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.mixed_score_MHA = MixedScore_MultiHeadAttention(**model_params)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)

    def forward(self, row_emb, col_emb, cost_mat):
        # NOTE: row and col can be exchanged, if cost_mat.transpose(1,2) is used
        # input1.shape: (batch, row_cnt, embedding)
        # input2.shape: (batch, col_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(row_emb), head_num=head_num)
        # q shape: (batch, head_num, row_cnt, qkv_dim)
        k = reshape_by_heads(self.Wk(col_emb), head_num=head_num)
        v = reshape_by_heads(self.Wv(col_emb), head_num=head_num)
        # kv shape: (batch, head_num, col_cnt, qkv_dim)

        out_concat = self.mixed_score_MHA(q, k, v, cost_mat)
        # shape: (batch, row_cnt, head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, row_cnt, embedding)

        out1 = self.add_n_normalization_1(row_emb, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3
        # shape: (batch, row_cnt, embedding)


########################################
# Decoder
########################################

class ATSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq_0 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved key, for single-head attention
        self.q1 = None  # saved q1, for multi-head attention

    def set_kv(self, encoded_jobs):
        # encoded_jobs.shape: (batch, job, embedding)
        head_num = self.model_params['head_num']

        self.k = reshape_by_heads(self.Wk(encoded_jobs), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_jobs), head_num=head_num)
        # shape: (batch, head_num, job, qkv_dim)
        self.single_head_key = encoded_jobs.transpose(1, 2)
        # shape: (batch, embedding, job)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']

        self.q1 = reshape_by_heads(self.Wq_1(encoded_q1), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_q0, ninf_mask):
        # encoded_q4.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, job)

        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        #######################################################
        q0 = reshape_by_heads(self.Wq_0(encoded_q0), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        q = self.q1 + q0
        # shape: (batch, head_num, pomo, qkv_dim)

        out_concat = self._multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, job)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, job)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, job)

        return probs

    def _multi_head_attention(self, q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
        # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or pomo
        # k,v shape: (batch, head_num, node, key_dim)
        # rank2_ninf_mask.shape: (batch, node)
        # rank3_ninf_mask.shape: (batch, group, node)

        batch_s = q.size(0)
        n = q.size(2)
        node_cnt = k.size(2)

        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        sqrt_qkv_dim = self.model_params['sqrt_qkv_dim']

        score = torch.matmul(q, k.transpose(2, 3))
        # shape: (batch, head_num, n, node)

        score_scaled = score / sqrt_qkv_dim
        if rank2_ninf_mask is not None:
            score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, node_cnt)
        if rank3_ninf_mask is not None:
            score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, node_cnt)

        weights = nn.Softmax(dim=3)(score_scaled)
        # shape: (batch, head_num, n, node)

        out = torch.matmul(weights, v)
        # shape: (batch, head_num, n, key_dim)

        out_transposed = out.transpose(1, 2)
        # shape: (batch, n, head_num, key_dim)

        out_concat = out_transposed.reshape(batch_s, n, head_num * qkv_dim)
        # shape: (batch, n, head_num*key_dim)

        return out_concat


########################################
# NN SUB FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed
