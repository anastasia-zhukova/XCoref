import pandas as pd
import os
import json
import sys
import re
from cdcr.config import ROOT_ABS_DIR
import warnings
import logging
import string
import csv
from datetime import datetime
import progressbar
import numpy as np
import copy
import difflib

logging.basicConfig(level=logging.INFO)
sys.path.append(f"{ROOT_ABS_DIR}/newstsc")
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
pd.options.mode.chained_assignment = None

from cdcr.config import ORIGINAL_DATA_PATH, DATA_PATH
from cdcr.pipeline import Pipeline
from cdcr.pipeline.modules import NewsPleaseReader
from cdcr.pipeline.modules import Preprocessor
from cdcr.util.cache import Cache


# --- SET THIS --- #
ANNOT_FOLDER = "2020_annot"
CATEGORIES_FILE = "aggr_m_conceptcategorization_2020.csv"
# -----------------#

# init table
DOC, CODE, SEGMENT, COMMENT, BEGINNING, END = "Document_name", "Code", "Segment", "Comment", "Beginning", "End"
COUNT, FOUND = "count", "found"

# final table
ID, CODE_TYPE, CODE_NAME, CODE_MENTION, TARGET_CONCEPT = 'id', 'code_type', 'code_name', 'code_mention', 'target_concept'
EVENT_ID, PUBL_ID, SENT, PARAGRAPH, START, END_FINAL = 'event_id', 'publisher_id', "sentence", "paragraph", "start", "end"

assoctable = {}
assoctable[r'Properties\Affection\Affection or empathy'] = 'Affection'
assoctable[r'Properties\Importance\Important & Intense'] = 'Importance'
assoctable[r'Properties\Other\Other'] = 'Other'
assoctable[r'Properties\Behaviour & character\Trustworthy'] = 'Trustworthiness'
assoctable[r'Properties\Violence & Unfair\Perpetrator & Aggressor (giving violence)'] = 'Aggressor'
assoctable[r'Properties\Behaviour & character\SHOWN as Not reasonable/Stupid'] = 'Unreason'
assoctable[r'Properties\Difficulty\Difficult'] = 'Difficulty'
assoctable[r'Properties\Behaviour & character\SHOWN as Reasonable/Smart'] = 'Reason'
assoctable[r'Properties\Power & Strength\Powerful & Leading'] = 'Power'
assoctable[r'Properties\Law\Lawful'] = 'Lawfulness'
assoctable[r'Properties\Violence & Unfair\Victim (receiving violence)'] = 'Victim'
assoctable[r'Properties\Quality\Bad Quality'] = 'Poor quality'
assoctable[r'Properties\Economy\Positive - economy'] = 'Economy positive'
assoctable[r'Properties\Economy\Negative - economy'] = 'Economy negative'
assoctable[r'Properties\Power & Strength\Weak & Following'] = 'Weakness'
assoctable[r'Properties\Behaviour & character\Not trustworthy'] = 'No trustworthiness'
assoctable[r'Properties\Quality\Bad Quality'] = 'Poor quality'
assoctable[r'Properties\Affection\Refusal'] = 'Refusal'
assoctable[r'Properties\Law\Unlawful'] = 'Unlawfulness'
assoctable[r'Properties\Importance\Unimportant'] = 'Unimportance'
assoctable[r'Properties\Safety\Unsafe'] = 'Unsafety'
assoctable[r'Properties\Other\Positive'] = 'Positive'
assoctable[r'Properties\Other\Negative'] = 'Negative'
assoctable[r'Properties\Behaviour & character\Fair'] = 'Fairness'
assoctable[r'Properties\Behaviour & character\Fair '] = 'Fairness'
assoctable[r'Properties\Behaviour & character\Confident'] = 'Confidence'
assoctable[r'Properties\Quality\Functioning'] = 'Good quality'
assoctable[r'Properties\Honor & patriotic values\Honor'] = 'Honor'
assoctable[r'Properties\Safety\Safe'] = 'Safety'
assoctable[r'Properties\Honor & patriotic values\Dishonor'] = 'Dishonor'
assoctable[r'Properties\Difficulty\Easy'] = 'Easiness'

DATA_ROOT_FOLDER = os.path.join(DATA_PATH, "NewsWCL50-prep")
DATAFOLDER = os.path.join(DATA_ROOT_FOLDER, ANNOT_FOLDER)


def convert_paragraphs_to_sents(doc):
    paragraphs = doc.fulltext.split("\n")
    par_sent = []
    add_sent = None
    sent_last_id = -1

    for par in paragraphs:
        par = par.replace("’", "\'")
        par = par.replace("“", "\"")
        par = par.replace("”", "\"")
        sents = [] if add_sent is None else [add_sent]
        keep_sent = copy.copy(add_sent)
        add_sent = None
        for sent in doc.sentences:
            if sent.index in sents:
                continue

            if re.sub(r'\W', " ", sent.text).rstrip() in re.sub(r'\W', " ", par).rstrip():
                sents.append(sent.index)

            elif re.sub(r'\W', " ", par).rstrip() in re.sub(r'\W', " ", sent.text).rstrip():
                if re.sub(r'\W', " ", sent.text).rstrip() in re.sub(r'\W', " ", paragraphs[len(par_sent) + 1]).rstrip():
                    continue
                sents.append(sent.index)
                add_sent = copy.copy(sent.index)

            else:
                d = difflib.SequenceMatcher(None, re.sub(r'\W', " ", par).rstrip(),
                                            re.sub(r'\W', " ", sent.text).rstrip())
                match = max(d.get_matching_blocks(), key=lambda x: x[2])
                st_i, st_j, st_length = match
                if len(d.a[st_i:st_i + st_length]) > 40:
                    sents.append(sent.index)
                    add_sent = copy.copy(sent.index)
        sel_sents = sorted([s for s in sents if (s > sent_last_id or (s == keep_sent and s >= sent_last_id - 1)) \
                            and s <= sent_last_id + len(sents)])
        sent_last_id = max(sel_sents) if len(sel_sents) else sent_last_id
        par_sent.append(list(range(min(sel_sents), max(sel_sents) + 1)) if len(sel_sents) else sel_sents)

    if len([0 for s in par_sent if not len(s)]) >= 2:
        logging.warning("Too many sentences not assigned to paragraphs.")
    return par_sent

def add_mention(par, mention_id, output_df, output_list):
    sents = doc.sentences[min(paragraphs_to_sents[par]): max(paragraphs_to_sents[par]) + 1]
    segm_group_df = par_annot_df.groupby(by=[SEGMENT]).count()[[DOC]]
    segm_group_df.columns = [COUNT]
    segm_group_df[FOUND] = [0] * len(segm_group_df)
    segm_df = par_annot_df.copy()
    segm_df = segm_df.drop_duplicates().set_index([SEGMENT])

    for sent in sents:
        # iterate over sentences

        for segm in list(segm_group_df.index):
            # iterate over annotations
            segm_ = segm.replace("’", "\'")
            segm_ = segm_.replace("“", "\"")
            segm_ = segm_.replace("”", "\"")

            if re.sub(r'\W', " ", segm_.lower()) not in re.sub(r'\W', " ", sent.text.lower()):
                continue

            code_df = segm_df.loc[segm, CODE]
            event_id = doc.source_domain.split("_")[0]

            for mention in re.finditer(re.sub(r'\W', " ", segm_.lower()), re.sub(r'\W', " ", sent.text.lower())):
                # iterate over found occurrences of a annotation in a sentence
                segm_index_start = mention.start()

                token_ids = []
                for token in sent.tokens:
                    if token.sentence_begin_char >= segm_index_start and \
                            token.sentence_end_char <= segm_index_start + len(segm_):
                        token_ids.append(token.index)

                if not len(token_ids):
                    continue

                if type(code_df) == pd.Series:
                    # combine codes if one annotation can have multiple codes
                    code_json = "+".join([v + "_" + str(topic_id) for v in list(code_df.values)])
                    mention_type_list = []
                    for code_part in list(code_df.values):
                        code_part = re.sub(r'´', "", code_part)
                        aa = list(code_types_df[(code_types_df["topic_id"] == event_id) &
                                              (code_types_df[CODE] == code_part)]["type"].values)
                        mention_type_list.extend(aa)
                    mention_type_json = list(mention_type_list)[0] + "+" if len(set(mention_type_list)) == 1 \
                        else "+".join(list(mention_type_list))
                else:
                    code_json = re.sub(r'´', "", code_df)
                    mention_type_list = list(code_types_df[(code_types_df["topic_id"] == event_id) &
                                                           (code_types_df[CODE] == code_json)]["type"].values)
                    if len(mention_type_list):
                        mention_type_json = mention_type_list[0]
                    elif "Properties\\" not in code_json:
                        mention_type_json = "UNK"
                        logging.warning("Annotation code \"{}\" has no type. Marked as \"UNK\".".format(code_json))
                    else:
                        mention_type_json = "PROPERTY"

                if "Properties\\" not in code_json:
                    code_fixed = re.sub(r'[^\w\+]', "_", code_json)
                    annot_dict = {
                        # "coref_chain": re.sub(r'\W+', "_", code_json) + "_" + doc.source_domain,
                        "coref_chain": code_fixed if "+" in code_fixed else code_fixed + "_" + str(topic_id),
                        "doc_id": doc.source_domain,
                        "mention_id": doc.source_domain + "_" + "{:04d}".format(mention_id),
                        "mention_type": mention_type_json.replace("-I", "-G").replace("-Misc", "-P"),
                        "score": -1,
                        "sent_id": sent.index,
                        "tokens_number": token_ids,
                        "tokens_str": " ".join([t.word for t in sent.tokens[min(token_ids): max(token_ids) + 1]]),
                        # "tokens_str": segm_,
                        "topic_id": topic
                    }
                    output_list.append(annot_dict)

                for code in list(code_df.values) if type(code_df) == pd.Series else [code_df]:
                    segm_group_df.loc[segm, FOUND] += 1

                    code = re.sub(r'´', "", code)
                    mention_type_list = list(code_types_df[(code_types_df["topic_id"] == event_id) &
                                                           (code_types_df[CODE] == code)]["type"].values)

                    if len(mention_type_list):
                        mention_type = mention_type_list[0]
                    elif "Properties\\" not in code:
                        mention_type = "UNK"
                        logging.warning("Annotation code \"{}\" has no type. Marked as \"UNK\".".format(code))
                    else:
                        mention_type = "PROPERTY"

                    mention_id_str = doc.source_domain + "_" + "{:04d}".format(mention_id)
                    if "Properties\\" not in code:
                        output_df = output_df.append(pd.DataFrame({
                            ID: mention_id_str,
                            CODE_TYPE: mention_type.replace("-I", "-G").replace("-Misc", "-P"),
                            CODE_NAME: code,
                            CODE_MENTION: " ".join([t.word for t in sent.tokens[min(token_ids): max(token_ids) + 1]]),
                            # CODE_MENTION: segm_,
                            TARGET_CONCEPT: "",
                            EVENT_ID: event_id,
                            PUBL_ID: doc.source_domain.split("_")[-1],
                            PARAGRAPH: par,
                            SENT: sent.index,
                            START: min(token_ids),
                            END_FINAL: max(token_ids)
                        }, index=[mention_id]))
                        mention_id += 1
                    else:
                        target_df = segm_df.loc[segm, COMMENT]
                        for target_row in list(target_df.values) if type(target_df) == pd.Series else [target_df]:
                            if target_row is np.nan:
                                continue
                            for target in target_row.split("\n"):
                                mention_id_str = doc.source_domain + "_" + "{:04d}".format(mention_id)
                                output_df = output_df.append(pd.DataFrame({
                                    ID: mention_id_str,
                                    CODE_TYPE: mention_type,
                                    CODE_NAME: assoctable[code] if code in assoctable else code,
                                    CODE_MENTION: " ".join([t.word for t in sent.tokens[min(token_ids): max(token_ids) + 1]]),
                                    # CODE_MENTION: segm_,
                                    TARGET_CONCEPT: ''.join(x for x in target if x in string.printable),
                                    EVENT_ID: int(event_id),
                                    PUBL_ID: doc.source_domain.split("_")[-1],
                                    PARAGRAPH: par,
                                    SENT: sent.index,
                                    START: min(token_ids),
                                    END_FINAL: max(token_ids)
                                }, index=[mention_id]))
                                mention_id += 1
    return segm_group_df, mention_id, output_df, output_list

code_types_df = pd.read_csv(os.path.join(DATA_ROOT_FOLDER, CATEGORIES_FILE))
code_types_df[CODE] = [re.sub(r'´', "", v) for v in list(code_types_df[CODE].values)]
topics = os.listdir(DATAFOLDER)
cache = Cache()

output_df = pd.DataFrame()

for topic_index, topic_csv in enumerate(topics):
    # if "41ecbplus2" not in topic_csv:
    #     continue
    # iterate over topics
    output_list = []
    topic_id = topic_csv.split("_")[0]
    topic = topic_csv.split(".")[0]
    print("PROGRESS: Converting annotations of topic \"{}\". ".format(topic))

    raw_df_init = pd.read_csv(os.path.join(DATAFOLDER, topic_csv))
    raw_df_init.columns = [re.sub(r'\W%', "", c).replace(" ", "_") for c in list(raw_df_init.columns)]
    raw_df = raw_df_init[[DOC, CODE, SEGMENT, COMMENT, BEGINNING, END]]
    raw_df[BEGINNING] = raw_df[BEGINNING].values - 1
    raw_df[END] = raw_df[END].values - 1

    docset = None
    for cache_file in cache.list()[::-1]:
        if topic in cache_file:
            docset = cache.load(cache_file)
            break

    if docset is None:
        # read files if no cached files found
        pipeline_setup = {
            "reading": {
                "module": NewsPleaseReader.run,
                "caching": False
            },
            "preprocessing": {
                "module": Preprocessor.run,
                "caching": True}
        }
        docset = Pipeline(pipeline_setup).run(cache_file=topic, configuration={})

    mention_id = 0
    widgets_doc = [
        progressbar.FormatLabel("PROGRESS: Converting %(value)d/%(max_value)d (%(percentage)d %%) "
                                "document (in: %(elapsed)s). ")]
    bar_doc = progressbar.ProgressBar(widgets=widgets_doc, maxval=len(docset)).start()

    for doc_index, doc in enumerate(docset):
        # iterate over docs
        paragraphs = doc.fulltext.split("\n")
        paragraphs_to_sents = convert_paragraphs_to_sents(doc)

        doc_annot_df = raw_df[raw_df[DOC] == doc.source_domain]
        doc_annot_df.sort_values(by=[BEGINNING], inplace=True)

        found = 0
        for par in range(len(paragraphs_to_sents)):
            # iterate over paragraphs
            par = int(par)
            par_annot_df = doc_annot_df[doc_annot_df[BEGINNING] == par]
            if not len(paragraphs_to_sents[par]) or not len(par_annot_df):
                continue

            segm_group_df, mention_id, output_df, output_list = add_mention(par, mention_id, output_df, output_list)

            found += int(np.sum(segm_group_df[FOUND].values))
            if int(np.sum(segm_group_df[FOUND].values)) / int(np.sum(segm_group_df[COUNT].values)) < 0.5:
                if len(paragraphs_to_sents[par - 1]):
                    prev_par = par - 1
                elif par - 2 >= 0:
                    prev_par = par - 2 if len(paragraphs_to_sents[par - 2]) else par
                else:
                    prev_par = par

                if prev_par!= par:
                    logging.warning(
                        "Some phrases were not found in paragraph {}. Try sentences in {} paragraph.".format(str(par),
                                                                                                             str(prev_par)))
                    segm_group_df, mention_id, output_df, output_list = add_mention(prev_par, mention_id, output_df,
                                                                                    output_list)
            for index, row in segm_group_df.iterrows():
                if row[COUNT] > row[FOUND]:
                    logging.warning(("Some phrases \"{}\" (is: {}, found: {}) were not found in paragraph {}.".format(
                                                                    index, row[COUNT], row[FOUND], str(par))))
        logging.info("Annotated excerpts: {}, mapped to preprocessed text: {}. ".format(len(doc_annot_df), found))
        bar_doc.update(doc_index + 1)
    bar_doc.finish()
    
    if topic not in os.listdir(os.path.join(DATA_ROOT_FOLDER, "test_parsing")):
        os.mkdir(os.path.join(DATA_ROOT_FOLDER, "test_parsing", topic))
    if "annotation" not in os.listdir(os.path.join(DATA_ROOT_FOLDER, "test_parsing", topic)):
        os.mkdir(os.path.join(DATA_ROOT_FOLDER, "test_parsing", topic, "annotation"))
    if ANNOT_FOLDER not in os.listdir(os.path.join(DATA_ROOT_FOLDER, "test_parsing", topic, "annotation")):
        os.mkdir(os.path.join(DATA_ROOT_FOLDER, "test_parsing", topic, "annotation", ANNOT_FOLDER))
    with open(os.path.join(DATA_ROOT_FOLDER, "test_parsing", topic, "annotation", ANNOT_FOLDER,
                           "concept_mentions" + "_" + str(topic_id) + ".json"), "w") as file:
        json.dump(output_list, file)

now = datetime.now()
export_file_name = now.strftime("%Y-%m-%d_%H-%M") + "_NewsWCL" + str(len(topics)) + "_" + ANNOT_FOLDER
output_df.to_csv(os.path.join(DATA_ROOT_FOLDER, "export_github", export_file_name + ".csv"), index=False)

try:
    with open(os.path.join(DATA_ROOT_FOLDER, "export_github", export_file_name + '.csv'), "r", newline='') as csvinputfile:
        csvraw = csv.reader(csvinputfile, delimiter=',')
        with open(os.path.join(DATA_ROOT_FOLDER, "export_github", export_file_name + "_sep_tag" + '.csv'), 'w', newline='') \
                as csvoutputfile:
            csvoutputfile.write('sep=,\n')
            finalwriter = csv.writer(csvoutputfile, delimiter=',')
            for row in csvraw:
                finalwriter.writerow(row)

except Exception:
    logging.error(Exception)
    sys.exit(1)
