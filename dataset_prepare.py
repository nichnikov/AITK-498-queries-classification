import os
from datasets import Dataset, DatasetDict
import pandas as pd

fns = ["tech_queries_clearing.csv", "expert_queries_clearing.csv"]

queries_df = pd.DataFrame()
for num, fn in enumerate(fns):
    temp_df = pd.read_csv(os.path.join(os.getcwd(), "data", fn), sep="\t")
    temp_df["label"] = num
    queries_df = pd.concat([queries_df, temp_df], axis=0)
    

queries_df.rename(columns={"Query": "text"}, inplace=True)
queries_df.to_csv(os.path.join(os.getcwd(), "data", "queries_lb2int.tsv"), sep="\t", index=False)

print(queries_df)
# Разбивка на тренировочную и валидационную выборки:
queries_df = queries_df.sample(frac=1)
val_quantity = queries_df.shape[0]//10

val_queries_df = queries_df[:val_quantity]
train_queries_df = queries_df[val_quantity:]

dataset = DatasetDict()
for nm, df in [("val", val_queries_df), ("train", train_queries_df)]:
    dtset = Dataset.from_pandas(df)
    dataset[nm] = dtset
    # dataset = DatasetDict({nm: dtset})
    print(dataset)

dataset.save_to_disk(os.path.join(os.getcwd(), "data", "train_val.data"))