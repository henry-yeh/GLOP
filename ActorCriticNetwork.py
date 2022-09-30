import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch import tanh
import math
from improvement import utils


class Encoder(nn.Module):
    """
    Encoder of TSP-Net
    """

    def __init__(self,
                 input_dim,
                 embedding_dim,
                 hidden_dim,
                 n_nodes,
                 n_rnn_layers):
        """
        Initialise Encoder
        :param int input_dim: Number of input dimensions
        :param int embedding_dim: Number of embbeding dimensions
        :param int hidden_dim: Number of hidden units of the RNN
        :param int n_layers: Number of RNN layers
        :param int n_nodes: Number of nodes in the TSP
        """
        super(Encoder, self).__init__()
        self.n_rnn_layers = n_rnn_layers
        self.hidden_dim = hidden_dim
        self.n_nodes = n_nodes

        self.embedding = nn.Linear(input_dim, embedding_dim)

        self.g_embedding = nn.Linear(embedding_dim, hidden_dim)
        self.g_embedding1 = nn.Linear(hidden_dim, hidden_dim)
        self.g_embedding2 = nn.Linear(hidden_dim, hidden_dim)

        self.rnn0 = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=n_rnn_layers,
                            batch_first=True)

        self.rnn = nn.LSTM(input_size=embedding_dim,
                           hidden_size=hidden_dim,
                           num_layers=n_rnn_layers,
                           batch_first=True)

        self.h0 = Parameter(torch.zeros(1), requires_grad=False)
        self.c0 = Parameter(torch.zeros(1), requires_grad=False)

        self.rnn0_reversed = nn.LSTM(input_size=embedding_dim,
                                     hidden_size=hidden_dim,
                                     num_layers=n_rnn_layers,
                                     batch_first=True)

        self.rnn_reversed = nn.LSTM(input_size=embedding_dim,
                                    hidden_size=hidden_dim,
                                    num_layers=n_rnn_layers,
                                    batch_first=True)

        self.W_f = nn.Linear(hidden_dim, hidden_dim)
        self.W_b = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, input, hidden=None):
        """
        Encoder: Forward-pass

        :param Tensor input: Graph inputs (bs, n_nodes, 2)
        :param Tensor hidden: hidden vectors passed as inputs from t-1
        """

        batch_size = input.size(0)

        edges = utils.batch_pair_squared_dist(input, input)
        edges.requires_grad = False

        # embedding shared across all nodes
        embedded_input = self.embedding(input)

        if hidden is None:
            h0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(self.n_rnn_layers,
                                                          batch_size,
                                                          self.hidden_dim)
            c0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(self.n_rnn_layers,
                                                          batch_size,
                                                          self.hidden_dim)
        else:
            h0, c0 = hidden
            h0 = h0.detach()
            c0 = h0.detach()

            h0 = h0.unsqueeze(0).repeat(self.n_rnn_layers, 1, 1)
            c0 = c0.unsqueeze(0).repeat(self.n_rnn_layers, 1, 1)

        g_embedding = embedded_input \
            + F.relu(torch.bmm(edges, self.g_embedding(embedded_input)))
        g_embedding = g_embedding \
            + F.relu(torch.bmm(edges, self.g_embedding1(g_embedding)))
        g_embedding = g_embedding \
            + F.relu(torch.bmm(edges, self.g_embedding2(g_embedding)))

        rnn_input = g_embedding
        rnn_input_reversed = torch.flip(g_embedding, [1])

        # first RNN reads the last node on the input
        rnn0_input = rnn_input[:, -1, :].unsqueeze(1)
        self.rnn0.flatten_parameters()
        _, (h0, c0) = self.rnn0(rnn0_input, (h0, c0))
        # second RNN reads the sequence of nodes
        self.rnn.flatten_parameters()
        s_out, s_hidden = self.rnn(rnn_input, (h0, c0))

        # first RNN reads the last node on the input
        rnn0_input_reversed = rnn_input_reversed[:, -1, :].unsqueeze(1)
        self.rnn0_reversed.flatten_parameters()
        _, (h0_r, c0_r) = self.rnn0_reversed(rnn0_input_reversed)
        # second RNN reads the sequence of nodes
        self.rnn_reversed.flatten_parameters()
        s_out_reversed, s_hidden_reversed = self.rnn_reversed(rnn_input_reversed,
                                                              (h0_r, c0_r))

        s_out = tanh(self.W_f(s_out)
                     + self.W_b(torch.flip(s_out_reversed, [1])))

        s_hidden = (s_hidden[0]+s_hidden_reversed[0],
                    s_hidden[1]+s_hidden_reversed[1])

        return s_out, s_hidden, _, g_embedding


class Attention(nn.Module):
    """
    Attention Mechanism of the Pointer-Net
    """

    def __init__(self, hidden_dim, C=10.0, T=1.0):
        super(Attention, self).__init__()
        """
        :param int hidden_dim: Number of hidden units in the query/ref
        """
        self.W1 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim,
                            kernel_size=1, stride=1, bias=False)
        self.W2 = nn.Linear(in_features=hidden_dim,
                            out_features=hidden_dim, bias=False)
        self.V = nn.Linear(in_features=hidden_dim,
                           out_features=1, bias=False)

        self.C = C
        self.T = T

        # Initialize vector V
        torch.nn.init.uniform_(self.V.weight,
                               a=-1.0/math.sqrt(hidden_dim),
                               b=1.0/math.sqrt(hidden_dim))
        self._inf = float('-inf')

    def forward(self,
                ref,
                q,
                mask=None):
        """
        Attention - Forward-pass

        :param Tensor decoder_state: Hidden state h of the decoder
        :param Tensor encoder_outputs: Outputs of the encoder
        :param Boolean mask: Selection mask
        :return: tuple of - (Attentioned hidden state, Alphas)
        """
        # ref: (batch_size, n_nodes, hidden_dim)
        # permute: (batch_size, hidden_dim, n_nodes) for conv1d
        ref = ref.permute(0, 2, 1)
        ref_W1 = self.W1(ref)
        # ref_W1: (batch_size, n_nodes, hidden_dim)
        ref_W1 = ref_W1.permute(0, 2, 1)

        # q_W2: (batch_size, 1, hidden_dim)
        q_W2 = self.W2(q).unsqueeze(1)

        # u_i: (batch_size, n_nodes, 1)
        u_i = self.V(tanh(ref_W1 + q_W2))

        # u_i: (batch_size, n_nodes)
        u_i = u_i.squeeze(-1)

        u_i = u_i.masked_fill_(mask, self._inf)
        # print("u_i after mask", u_i)

        u_i = self.C*tanh(u_i/self.T)
        # probs: (batch_size, n_nodes)
        probs = F.softmax(u_i, dim=1)

        # q_a: (batch, 1, hidden_dim)
        q_a = torch.bmm(probs.unsqueeze(1), ref_W1)
        # hidden_state_dec: (batch, hidden_dim)
        q_a = q_a.squeeze(1)

        return probs, q_a


class Decoder(nn.Module):
    """
    Decoder
    """

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 n_actions):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions

        self.W_0 = nn.Linear(hidden_dim, hidden_dim)
        self.W_1 = nn.Linear(hidden_dim, hidden_dim)

        self.W_star = nn.Linear(hidden_dim, hidden_dim//2)
        self.W_s = nn.Linear(hidden_dim, hidden_dim//2)

        self.att = Attention(hidden_dim)

        self.mask = Parameter(torch.ones(1), requires_grad=False)
        self.runner = Parameter(torch.zeros(1), requires_grad=False)

        self.init_dec = Parameter(torch.FloatTensor(hidden_dim),
                                  requires_grad=False)

        nn.init.uniform_(self.init_dec, -1/math.sqrt(hidden_dim),
                         1/math.sqrt(hidden_dim))

    def forward(self, q, ref, inp, actions=None, g_emb=None, q_star=None):

        batch_size = ref.size(0)
        n_nodes = ref.size(1)

        if g_emb is not None:
            g_emb, _ = torch.max(g_emb, dim=1)

        # mask: (batch, n_nodes) filled with 1's
        mask = self.mask.repeat((batch_size, n_nodes))

        # runner: (input_lenght) tensor filled with 0's
        runner = self.runner.repeat(n_nodes)
        # runner: (input_lenght) tensor from {0 to input_lenght-1}
        for i in range(n_nodes):
            runner.data[i] = i
        # (batch, seq_len) filled with {0,...,seq-len-1}
        runner = runner.unsqueeze(0).expand(batch_size, -1).long()
        if q_star is not None:
            q_star_h, q_star_c = q_star
            q_h, q_c = q

            q_h = torch.cat([self.W_star(q_star_h), self.W_s(q_h)], dim=1)
        else:
            # h,c: input should be tuple(2) of (batch_size, hidden_dim)
            q_h, q_c = q
        # dec_input: (batch_size, embedding_dim)
        dec_input = self.init_dec.unsqueeze(0).expand(batch_size, -1)
        if g_emb is not None:
            h = q_h + g_emb
        else:
            h = q_h
        # lists for the outputs
        probs = []
        pointers = []
        log_probs_pts = []
        entropy = []

        ################### change mask ###################
        mask[:, -1] = 0
        mask[:, 0] = 0
        
        for i in range(self.n_actions):
            if i == 0:
                # if it's the first output mask the last index
                mask[:, -2] = 0
            if i == 1:
                # return the last index
                mask[:, -2] = 1

            h = tanh(self.W_1(h) + self.W_0(dec_input))

            prob, _ = self.att(ref, h, torch.eq(mask, 0))

            # # Masking selected inputs
            # masked_outs: (batch, seq_len)
            masked_prob = prob*mask

            c = torch.distributions.Categorical(masked_prob)
            if actions is None:
                indices = c.sample()
                log_probs_idx = c.log_prob(indices)
                dist_entropy = c.entropy()
            else:
                indices = actions[:, i]
                log_probs_idx = c.log_prob(indices)
                dist_entropy = c.entropy()

            repeat_indices = indices.unsqueeze(1).expand(-1, n_nodes)

            # 1-pointers probs indices i.e. if idx= 4 and len = 5
            # one_hot_pointers[0] = [0, 0 , 0 , 0 , 1]
            # one_hot_pointers: (batch_size, seq_len)
            one_pointers = (runner == repeat_indices).float()
            lower_pointers = (runner <= repeat_indices).float()

            # Update mask to ignore seen indices
            # (mask gets updated from 1 --> 0 for seen indices)
            # mask: (batch_size, seq_len)
            mask = mask * (1 - lower_pointers)

            # embbeding mask: boolean (batch size, seq_len, embbeding_dim)
            # True for the pointed input False otherwise
            one_pointers = one_pointers.unsqueeze(2)
            dec_input_mask = one_pointers.expand(-1,
                                                 -1,
                                                 self.hidden_dim).bool()
            masked_dec_input = inp[dec_input_mask.data]
            dec_input = masked_dec_input.view(batch_size, self.hidden_dim)

            # outputs: list of softmax outputs of size (1, batch_size, seq_len)
            probs.append(prob.unsqueeze(0))
            pointers.append(indices.unsqueeze(1))
            log_probs_pts.append(log_probs_idx.unsqueeze(1))
            entropy.append(dist_entropy.unsqueeze(1))

        probs = torch.cat(probs).permute(1, 0, 2)

        # pointers: index outputs (batch_size, n_actions)
        pointers = torch.cat(pointers, 1)
        log_probs_pts = torch.cat(log_probs_pts, 1)
        entropies = torch.cat(entropy, 1)

        return probs, pointers, log_probs_pts, entropies


class ActorCriticNetwork(nn.Module):

    """
    ActorCritic-Net
    """

    def __init__(self,
                 input_dim,
                 embedding_dim,
                 hidden_dim,
                 n_nodes,
                 n_rnn_layers,
                 n_actions,
                 graph_ref=False):
        """
        :param int embedding_dim: Number of embbeding dimensions
        :param int hidden_dim: Encoder/Decoder hidden units
        :param int lstm_layers: Number of LSTM layers
        :param bool bidir: Bidirectional
        :param bool batch_first: Batch first in the LSTM
        """

        super(ActorCriticNetwork, self).__init__()

        self.encoder = Encoder(input_dim,
                               embedding_dim,
                               hidden_dim,
                               n_nodes,
                               n_rnn_layers)

        self.encoder_star = Encoder(input_dim,
                                    embedding_dim,
                                    hidden_dim,
                                    n_nodes,
                                    n_rnn_layers)

        self.decoder_a = Decoder(embedding_dim,
                                 hidden_dim,
                                 n_actions)

        self.W_star = nn.Linear(hidden_dim, hidden_dim//2)
        self.W_s = nn.Linear(hidden_dim, hidden_dim//2)

        self.decoder_c = nn.Sequential(
                         nn.Linear(hidden_dim, hidden_dim),
                         nn.ReLU(),
                         nn.Linear(hidden_dim, 1))
        self.graph_ref = graph_ref

    def forward(self, inputs, inputs_star, hidden=None, actions=None):

        _, s_hidden_star, _, _ = self.encoder_star(inputs_star, hidden)

        s_out, s_hidden, _, g_embedding = self.encoder(inputs, hidden)

        # enc_h: get the last layer of the LSTM encoder
        enc_h = (s_hidden[0][-1], s_hidden[1][-1])
        enc_h_star = (s_hidden_star[0][-1], s_hidden_star[1][-1])

        probs, pts, log_probs_pts, entropies = self.decoder_a(enc_h,
                                                              s_out,
                                                              s_out,
                                                              actions,
                                                              g_embedding,
                                                              enc_h_star)

        v_g = torch.mean(g_embedding, dim=1).squeeze(1)
        h_v = torch.cat([self.W_star(enc_h_star[0]), self.W_s(enc_h[0])],
                        dim=1)
        v = self.decoder_c(v_g + h_v)

        return probs, pts, log_probs_pts, v, entropies, enc_h
