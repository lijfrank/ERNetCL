import torch
import torch.nn.functional as F
import math
import copy
from torch import nn
from torch.autograd import Variable

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MyAttention(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(MyAttention, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
    def forward(self, features_src, key_padding_mask): 
        features = features_src
        for mod in self.layers:
            features = mod(features, key_padding_mask)
        return features
    
class MyAttentionLayer(nn.Module):
    def __init__(self, feature_size, nheads=4, dropout=0.2, norm_first=True, residual=True, no_cuda=False):
        super(MyAttentionLayer, self).__init__() 
        self.no_cuda = no_cuda
        self.residual = residual
        self.norm_first = norm_first
        self.multihead_attn = nn.MultiheadAttention(feature_size, nheads)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(feature_size)

    def forward(self, features_src, key_padding_mask):
        key_padding_mask = key_padding_mask.transpose(0,1)
        
        features = features_src
        if self.residual:
            if self.norm_first:
                features = features_src + self.att(self.norm(features), key_padding_mask)
            else:
                features = self.norm(features_src + self.att(features, key_padding_mask))
        else:
            if self.norm_first:
                features = self.att(self.norm(features), key_padding_mask)
            else:
                features = self.norm(self.att(features, key_padding_mask))
        return features
    
    def att(self, features, key_padding_mask):
        features = self.multihead_attn(features, features, features, key_padding_mask=key_padding_mask)[0]
        return self.dropout(features)

class MyRecurrent(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(MyRecurrent, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
    def forward(self, features_src, seq_lengths): 
        features = features_src
        for mod in self.layers:
            features = mod(features, seq_lengths)
        return features
    
class MyRecurrentLayer(nn.Module):
    def __init__(self, feature_size, rnn_model, dropout=0.2, norm_first=True, residual=True, no_cuda=False):
        super(MyRecurrentLayer, self).__init__() 
        self.no_cuda = no_cuda
        self.residual = residual
        self.rnn_model = rnn_model
        self.norm_first = norm_first
        if self.rnn_model == 'LSTM':
            self.rnn = nn.LSTM(input_size=feature_size, hidden_size=feature_size, num_layers=1, bidirectional=True)
        elif self.rnn_model == 'GRU':
            self.rnn = nn.GRU(input_size=feature_size, hidden_size=feature_size, num_layers=1, bidirectional=True)
        else:
            print('Base model must be one of LSTM/GRU/Linear')
            raise NotImplementedError
        self.linear_rnn = nn.Linear(2*feature_size, feature_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(feature_size)

    def forward(self, features_src, seq_lengths):
        features = features_src
        if self.residual:
            if self.norm_first:
                features = features_src + self.rnn_(self.norm(features), seq_lengths)
            else:
                features = self.norm(features_src + self.rnn_(features, seq_lengths))
        else:
            if self.norm_first:
                features = self.rnn_(self.norm(features), seq_lengths)
            else:
                features = self.norm(self.rnn_(features, seq_lengths))
        return features
    
    def rnn_(self, features, seq_lengths):
        features_ = nn.utils.rnn.pack_padded_sequence(features, seq_lengths.cpu(), enforce_sorted=False)
        self.rnn.flatten_parameters()
        features_rnn = self.rnn(features_)[0]
        features_rnn = nn.utils.rnn.pad_packed_sequence(features_rnn)[0]

        features = self.linear_rnn(features_rnn)
        return self.dropout(features)


class MyDialogue(nn.Module):
    def __init__(self, rnn_model='LSTM', rnn_layer=2, attention_head=8, attention_layer=6, norm_first=True, use_residual=False, input_size=None, input_in_size=None, feature_mode=None, n_speakers=2, use_speaker=False, n_classes=7, dropout=0.2, cuda_flag=False, ran_mode=None):
        super(MyDialogue, self).__init__()
        self.no_cuda = cuda_flag
        self.rnn_model = rnn_model
        self.n_speakers = n_speakers
        self.feature_mode = feature_mode
        self.use_speaker = use_speaker
        self.ran_mode = ran_mode

        if feature_mode == 'concat4':
           input_size = 4*input_size
        elif feature_mode == 'concat2':
            input_size = 2*input_size
        else:
            input_size = input_size

        self.linear_in = nn.Linear(input_size, input_in_size)
        self.dropout_in = nn.Dropout(dropout)
        self.speaker_embeddings = nn.Embedding(n_speakers, input_in_size)

        myattentionlayer = MyAttentionLayer(feature_size=input_in_size, nheads=attention_head, dropout=dropout, norm_first=norm_first, residual=use_residual, no_cuda=cuda_flag)
        self.myattention = MyAttention(myattentionlayer, num_layers=attention_layer)

        myrecurrentlayer = MyRecurrentLayer(feature_size=input_in_size, rnn_model=rnn_model, dropout=dropout, norm_first=norm_first, residual=use_residual, no_cuda=cuda_flag)
        self.myrecurrent = MyRecurrent(myrecurrentlayer, rnn_layer)
        
        if ran_mode == 'dran':
            self.linear_cat = nn.Linear(2*input_in_size, input_in_size)

        self.smax_fc = nn.Linear(input_in_size, n_classes)
        # self.smax_fc = nn.Linear(input_in_size, n_classes)

    def forward(self, r1, r2, r3, r4, qmask, umask, seq_lengths):
        if self.feature_mode == 'concat4':
           features = torch.cat([r1, r2, r3, r4], axis=-1)
        elif self.feature_mode == 'concat2':
            features = torch.cat([r1, r2], axis=-1)
        elif self.feature_mode == 'sum4':
            features = (r1 + r2 + r3 + r4)/4
        elif self.feature_mode == 'r1':
            features = r1
        elif self.feature_mode == 'r2':
            features = r2
        elif self.feature_mode == 'r3':
            features = r3
        elif self.feature_mode == 'r4':
            features = r4
        features = self.dropout_in(self.linear_in(features))

        if self.use_speaker:
            spk_idx = torch.argmax(qmask, dim=-1).cuda() if not self.no_cuda else torch.argmax(qmask, dim=-1)
            spk_emb_vector = self.speaker_embeddings(spk_idx)
            features = features + spk_emb_vector

        features_rnn = self.myrecurrent(features, seq_lengths=seq_lengths)
        features_att = self.myattention(features_rnn, key_padding_mask=umask)
        features_cat = features_att

        prob = self.smax_fc(features_cat)
        
        return prob
