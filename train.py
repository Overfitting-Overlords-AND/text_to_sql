import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import constants
from transformer import Transformer

src_vocab_size = constants.VOCAB_SIZE
tgt_vocab_size = constants.VOCAB_SIZE
d_model = constants.D_MODEL
num_heads = constants.NUM_HEADS
num_layers = constants.NUM_LAYERS
d_ff = constants.D_FF
max_seq_length = constants.MAX_SEQ_LENGTH
dropout = constants.DROPOUT
batch_size = constants.BATCH_SIZE

transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

# Generate random sample data
srcq_data = torch.randint(1, src_vocab_size, (batch_size, max_seq_length))
srcc_data = torch.randint(1, src_vocab_size, (batch_size, max_seq_length))
tgt_data = torch.randint(1, tgt_vocab_size, (batch_size, max_seq_length))

criterion = nn.CrossEntropyLoss(ignore_index=3)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train()

for epoch in range(100):
    optimizer.zero_grad()
    output = transformer(srcq_data, srcc_data, tgt_data[:, :-1])
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")