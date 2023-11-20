import torch
# from datasets import load_dataset
import sentencepiece as spm
import json
import constants
import random

class TextToSqlData(torch.utils.data.Dataset):
  def __init__(self):
    # self.dataset = load_dataset(name,split=mode)
    with open('./sql_create_context_V4.json', 'r') as file:
      self.dataset = json.load(file) 
    self.sp = spm.SentencePieceProcessor(model_file='text_to_sql.model') 

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    record = self.dataset[idx]
    # record = self.dataset[random.randint(0, 4)]
    encoded_question = self.sp.encode_as_ids(record["question"], add_bos=True, add_eos=True)[0:constants.MAX_SEQ_LENGTH]
    encoded_context = self.sp.encode_as_ids(record["context"])[0:constants.MAX_SEQ_LENGTH]
    encoded_answer = self.sp.encode_as_ids(record["answer"], add_bos=True, add_eos=True)[0:constants.MAX_SEQ_LENGTH]
    # return torch.tensor(encoded_question)
    return torch.tensor(encoded_question), torch.tensor(encoded_context), torch.tensor(encoded_answer)
  
  def collate(self, batch):
    q = torch.nn.utils.rnn.pad_sequence([torch.cat((item[0],item[1]), dim=0) for item in batch], batch_first=True, padding_value=3)
    # c = torch.nn.utils.rnn.pad_sequence([item[1] for item in batch], batch_first=True, padding_value=3)
    a = torch.nn.utils.rnn.pad_sequence([item[2] for item in batch], batch_first=True, padding_value=3)
    return q, a
  
if __name__ == '__main__':
  data = TextToSqlData()
  print(data.__getitem__(0))