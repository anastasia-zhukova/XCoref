import pandas as pd
from cdcr.structures.candidate import Candidate
import progressbar

DOC_ID = "doc_id"
SENT_ID = "sent_id"
HEAD_ID = "head_id"
BEGIN_ID = "begin_id"
END_ID = "end_id"
PHRASE = "phrase"
# COREF = "coref"


def create_cand_table(cand_set, suffix="", drop_duplicates=False):
    df = make_table(suffix)

    if not len(cand_set):
        return df

    widgets = [
        progressbar.FormatLabel("PROGRESS: Saved row with candidate info from %(value)d (%(percentage)d %%)"
                                "candidate groups (in: %(elapsed)s).")
    ]
    bar = progressbar.ProgressBar(widgets=widgets,
                                  maxval=len(cand_set)).start()

    for i, cand_group in enumerate(cand_set):
        for cand in cand_group:
            df = df.append(one_row_table(cand, df))
        bar.update(i + 1)
    bar.finish()

    if drop_duplicates:
        return df.drop_duplicates(subset=[t+suffix for t in [PHRASE, DOC_ID, SENT_ID, HEAD_ID]])

    return df


def create_cand_short_table(val, suffix):
    df = make_table(suffix)
    return one_row_table(val, df)


def make_table(suffix):
    return pd.DataFrame(columns=[col + suffix for col in [PHRASE, DOC_ID, SENT_ID, BEGIN_ID, END_ID, HEAD_ID]])


def one_row_table(val, df):
    if type(val) == Candidate:
        return pd.DataFrame({col: val for col, val in zip(list(df.columns),
                                                          [val.text, val.document.id, val.sentence.index,
                                                           val.tokens[0].index, val.tokens[-1].index,
                                                           val.head_token.index])}, index=[val.id])
    if type(val) == list:
        return pd.DataFrame({col: val for col, val in zip(list(df.columns), val)}, index=["test"])
