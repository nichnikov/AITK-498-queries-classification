import os
import numpy as np
import pandas as pd
from datasets import DatasetDict
from config import PATH
from transformers import (BertTokenizer, 
                          BertModelWithHeads,
                          )
from transformers.adapters.composition import Fuse
import torch

def predict(premise):
    encoded = tokenizer(premise,  max_length=512, return_tensors="pt")
    logits = model(**encoded)[0]
    # tanh = torch.tanh(logits)
    pred_class = torch.argmax(logits).item()
    # print("sigmoid:", torch.sigmoid(logits))
    return pred_class



model_name = "bert-base-multilingual-cased"
tokenizer = BertTokenizer.from_pretrained(model_name, output_attentions=True)
model = BertModelWithHeads.from_pretrained(model_name)


adapter_name = "classifier_adapter"

adapter_path = os.path.join(os.getcwd(), "models", adapter_name)
model.load_adapter(adapter_path)
model.set_active_adapters(adapter_name)

dataset_df = pd.read_csv(os.path.join(PATH, "data", "test_dataset_lb2int.csv"), sep="\t")
test_dataset = list(dataset_df.itertuples(index=False))


result = []
true_pred = 0
not_other_pred = 0
k = 1
for l_str, true_label, text in test_dataset:
    predict_label = predict(text)
    if predict_label == true_label:
        true_pred += 1
    if predict_label != 4:
        not_other_pred += 1
    print("true_pred:", true_pred, "not_other_pred:", not_other_pred)
    print(k, "true:", true_label, "predict:", predict_label)
    k += 1
    
print("true_pred:", true_pred, "precision:", true_pred/not_other_pred)
print("not_other_pred:", not_other_pred, "recall:", not_other_pred/len(test_dataset))