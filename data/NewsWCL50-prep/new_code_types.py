import pandas as pd
import os
import re

FOLDER = "C:\\Users\\annaz\\PycharmProjects\\newsalyze-backend_2\\data\\NewsWCL50-prep\\2020_annot"
OLD_CODES = "C:\\Users\\annaz\\PycharmProjects\\newsalyze-backend_2\\data\\NewsWCL50-prep\\aggr_m_conceptcategorization.csv"
NEW_CODES = "C:\\Users\\annaz\\PycharmProjects\\newsalyze-backend_2\\data\\NewsWCL50-prep\\aggr_m_conceptcategorization_2020.csv"

DOC, CODE, SEGMENT, COMMENT, BEGINNING, END = "Document_name", "Code", "Segment", "Comment", "Beginning", "End"
TOPIC = "topic_id"

combined_df = pd.DataFrame()
last_id = 0
for file_name in os.listdir(FOLDER):
    local_df = pd.read_csv(os.path.join(FOLDER, file_name))
    local_df.index = list(range(last_id, last_id + len(local_df)))
    combined_df = combined_df.append(local_df)

combined_df.columns = [re.sub(r'\W%', "", c).replace(" ", "_") for c in list(combined_df.columns)]
combined_df[TOPIC] = [int(v.split("_")[0]) for v in list(combined_df[DOC].values)]

codes_df = combined_df[[TOPIC, CODE]].drop_duplicates()
codes_df[CODE] = [re.sub(r'´', "", v) for v in list(codes_df[CODE].values)]

old_code_types_df = pd.read_csv(OLD_CODES)
old_code_types_df[CODE] = [re.sub(r'´', "", v) for v in list(old_code_types_df[CODE].values)]

merged_df = codes_df.merge(old_code_types_df, how="left", on=[TOPIC, CODE])
merged_df.fillna("", inplace=True)
merged_df_ = merged_df[~merged_df[CODE].str.contains('Properties')]
merged_df_.to_csv(NEW_CODES, index=False)
print()
