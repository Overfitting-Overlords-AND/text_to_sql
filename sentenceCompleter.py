import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from transformer import Transformer
import sentencepiece as spm
import constants
import utilities
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

def generate(question, context):

  sp = spm.SentencePieceProcessor(model_file='text_to_sql.model')
  question = torch.tensor(sp.encode_as_ids(question, add_bos=True)).long().unsqueeze(0)
  print("question shape: ", question.shape)
  context = torch.tensor(sp.encode_as_ids(context, add_bos=True)).long().unsqueeze(0)
  print("context shape: ", context.shape)
  transformer = Transformer(constants.VOCAB_SIZE, constants.VOCAB_SIZE, constants.D_MODEL, constants.NUM_HEADS, constants.NUM_LAYERS, constants.D_FF, constants.MAX_SEQ_LENGTH, constants.DROPOUT)
  transformer.eval()
  utilities.load_latest_checkpoint(transformer)
  
  answer = torch.tensor([[1]]) # <s>=1
  print("answer shape: ", answer.shape)
  for _ in range(constants.MAX_SEQ_LENGTH):
    with torch.no_grad():
      logits = transformer(question, context, answer) 
      logits = logits[:, -1, :] / 1.0
      probs = torch.nn.functional.softmax(logits, dim=-1)
      next = torch.multinomial(probs, num_samples=1)
      if next.item() == 2: break # </s>=2
      answer = torch.cat([answer, next], dim=1)
  
  # attn_probs =  transformer.decoder_layers[0].self_attn.attn_probs 
  # sns.heatmap(attn_probs[0][0], cmap="viridis")
  # plt.show()
  
  output = sp.decode(answer.tolist()[0])
  print(f"{output}")
  return { "sql" : output } 

