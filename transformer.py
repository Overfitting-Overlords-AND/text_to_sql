import torch
import torch.nn as nn
from positional_encoding import PositionalEncoding
from encoder_layer import EncoderLayer
from decoder_layer import DecoderLayer

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.segment_embedding = nn.Embedding(2, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_src_mask(self, srcq, srcc):
        return (torch.cat((srcq, srcc), dim=0) != 3).unsqueeze(1).unsqueeze(2) # pad_id=3

    def generate_tgt_mask(self, tgt):
        tgt_mask = (tgt != 3).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        return tgt_mask & nopeak_mask

    def forward(self, srcq, srcc, tgt):
        src_mask = self.generate_src_mask(srcq, srcc)
        tgt_mask = self.generate_tgt_mask(tgt)
        q_emb = self.segment_embedding(torch.tensor(0)) + self.encoder_embedding(srcq)
        c_emb = self.segment_embedding(torch.tensor(1)) + self.encoder_embedding(srcc)
        src_embedded = self.dropout(self.positional_encoding(torch.cat((q_emb, c_emb), dim=1)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output