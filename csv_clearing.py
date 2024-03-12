import os
import re
import pandas as pd

ex_queries_df = pd.read_csv(os.path.join("data", "expert_queries.csv"), sep="\t")
ex_queries_df["Query"] = ex_queries_df["Query"].str.replace("\n", " ")
ex_queries_df.to_csv(os.path.join("data", "expert_queries_clearing.csv"), sep="\t", index=False)

tech_queries_df = pd.read_csv(os.path.join("data", "tech_queries.csv"), sep="\t")
tech_queries_df["Query"] = tech_queries_df["Query"].str.replace("\n", " ")
tech_queries_df.to_csv(os.path.join("data", "tech_queries_clearing.csv"), sep="\t", index=False)