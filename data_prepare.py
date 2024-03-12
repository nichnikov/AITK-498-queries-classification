import os
from datasets import Dataset, DatasetDict
import pandas as pd

fns = ["homonymy_1.csv", "homonymy_2.csv", "homonymy_3.csv", "homonymy_4.csv"]

queries_df = pd.DataFrame()
for fn in fns:
    temp_df = pd.read_csv(os.path.join(os.getcwd(), "data", fn), sep="\t")
    queries_df = pd.concat([queries_df, temp_df], axis=0)
    
# queries_df = pd.read_csv(os.path.join(os.getcwd(), "data", "train_dataset_lb2int.csv"), sep="\t")
print(set(list(queries_df["label"])))

# добавление числовых классов
queries_df.rename(columns={"label": "label_str"}, inplace=True)
queries_with_lb_df = pd.DataFrame()
for num, lb in enumerate(sorted(set(list(queries_df["label_str"])))):
    temp_df = queries_df[queries_df["label_str"] == lb]
    temp_df["label"] = num
    queries_with_lb_df = pd.concat([queries_with_lb_df, temp_df], axis=0)

queries_with_lb_df.to_csv(os.path.join(os.getcwd(), "data", "queries_lb2int.tsv"), sep="\t", index=False)
print(queries_with_lb_df)

dtset = Dataset.from_pandas(queries_with_lb_df)
train_data = DatasetDict({"train": dtset})
print(train_data)
train_data.save_to_disk(os.path.join(os.getcwd(), "data", "train.data"))