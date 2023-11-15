import torch
# from datasets import load_dataset
import sentencepiece as spm
import json

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
    encoded_question = self.sp.encode_as_ids(record["question"], add_bos=True, add_eos=True)
    encoded_context = self.sp.encode_as_ids(record["context"], add_bos=True, add_eos=True)
    encoded_answer = self.sp.encode_as_ids(record["answer"], add_bos=True, add_eos=True)
    # return torch.tensor(encoded_question)
    return torch.tensor(encoded_question), torch.tensor(encoded_context), torch.tensor(encoded_answer)
  
  # def collate_function(self, batch):
  #   return torch.nn.utils.rnn.pad_sequence([item for item in batch], batch_first=True, padding_value=3)

if __name__ == '__main__':
  data = TextToSqlData()
  print(data.__getitem__(0))