import os
import numpy as np
import pandas as pd
from datasets import DatasetDict
from transformers import (BertTokenizer, 
                          BertModelWithHeads,
                          )
from transformers.adapters.composition import Fuse
import torch


"""
# Old: NLI task
def predict(premise, hypothesis):
  encoded = tokenizer(premise, hypothesis, return_tensors="pt")
  # if torch.cuda.is_available():
  #  encoded.to("cuda")
  logits = model(**encoded)[0]
  tanh = torch.tanh(logits)
  pred_class = torch.argmax(logits).item()
  print("sigmoid:", torch.sigmoid(logits))
  return pred_class
"""

def predict(premise):
    encoded = tokenizer(premise, return_tensors="pt")
    logits = model(**encoded)[0]
    tanh = torch.tanh(logits)
    pred_class = torch.argmax(logits).item()
    # print("sigmoid:", torch.sigmoid(logits))
    return pred_class



model_name = "bert-base-multilingual-cased"
tokenizer = BertTokenizer.from_pretrained(model_name, output_attentions=True)
model = BertModelWithHeads.from_pretrained(model_name)


adapter_name = "classifier_adapter"

# adapter_path = os.path.join(os.getcwd(), "models", mode_name)
adapter_path = os.path.join(os.getcwd(), "models", adapter_name)
model.load_adapter(adapter_path)
model.set_active_adapters(adapter_name)


texts = [(0, "приказ о смене паспорта"), (0, "увольнение в связи со сменой места жительства"), (1, "услуги смп"),
         (1, "бронхит смп"), (2, "экспедитор выплачивает стоимость утраченного товара с ндс"), (4, "нормативы сброса ндс")]

k = 1
for l, tx in texts:
    prd = predict(tx)
    print(k, "true:", l, "predict:", prd)
    k += 1