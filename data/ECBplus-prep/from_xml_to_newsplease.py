import xml.etree.ElementTree as ET
import os
import json
import string
import copy
import re
import pandas as pd
import numpy as np
from datetime import datetime
from cdcr.config import ORIGINAL_DATA_PATH, DATA_PATH

TOPIC_NUMBER_TO_CONVERT = 3

DATA_ROOT_FOLDER = os.path.join(DATA_PATH, "ECBplus-prep")
ECBPLUS = "ecbplus.xml"
ECB = "ecb.xml"

ecb_path = os.path.join(DATA_ROOT_FOLDER, "ECB+")
result_path = os.path.join(DATA_ROOT_FOLDER, "test_parsing")
path_sample = os.path.join(ORIGINAL_DATA_PATH, "_sample.json")

with open(path_sample, "r") as file:
    newsplease_format = json.load(file)


def append_text(text, word):
    space = "" if word in ".,?!)]`\"\'" or word == "'s" else " "
    space = " " if word[0].isupper() and not len(text) else space
    space = " " if word in ["-", "("] else space
    if len(text):
        space = "" if text[-1] in ["(", "``", "\"", "["] else space
        space = " " if text[-1] in [".,?!)]`\"\'"] else space
        space = "" if text[-1] in ["\""] and word.istitle() else space
        space = "" if word in ["com", "org"] else space
    if len(text) > 1:
        # space = " " if text[-2:0] == "\" " and word.istitle() else space
        space = "" if word.isupper() and text[-1] == "." and text[-2].isupper() else space
        space = " " if not len(re.sub(r'\W+', "", text[-2:])) and len(text[-2:]) == len(text[-2:].replace(" ", "")) else space
    word = "\"" if word == "``" else word
    return text + space + word


def convert_files(topic_number_to_convert=TOPIC_NUMBER_TO_CONVERT, check_with_list=True):
    doc_files = {}
    coref_dics = {}
    with open(os.path.join(DATA_ROOT_FOLDER, "test_events.json"), "r") as file:
        selected_articles = json.load(file)

    # selected_topics = sorted(list(set([a.split("_")[0] for a in selected_articles])))
    # selected_topics = ['36', "37", "38", "39", "40", "41", "42", "43", "44", "45"]
    selected_topics = [str(i) for i in range(36)]

    summary_df = pd.DataFrame()

    for topic_folder in os.listdir(ecb_path)[:topic_number_to_convert]:

        if check_with_list and topic_folder not in selected_topics:
            continue

        unknown_tag = ""
        diff_folders = {ECB: [], ECBPLUS: []}
        for topic_file in os.listdir(os.path.join(ecb_path, topic_folder)):
            if ECBPLUS in topic_file:
                diff_folders[ECBPLUS].append(topic_file)
            else:
                diff_folders[ECB].append(topic_file)

        for annot_folders in list(diff_folders.values()):

            t_number = annot_folders[0].split(".")[0].split("_")[0]
            t_name = re.search(r'[a-z]+', annot_folders[0].split(".")[0])[0]
            topic_name = t_number + t_name
            coref_dict = {}
            doc_sent_map = {}
            IS_TEXT, TEXT = "is_text", "text"

            for topic_file in annot_folders:

                tree = ET.parse(os.path.join(ecb_path, topic_folder, topic_file))

                if check_with_list and topic_file.split(".")[0] not in selected_articles:
                    continue

                root = tree.getroot()

                title, text, url, time, time2, time3 = "", "", "", "", "", ""

                token_dict, mentions, mentions_map = {}, {}, {}

                t_id = -1
                old_sent = 0
                sent_dict = {}
                for elem in root:
                    try:
                        if old_sent == int(elem.attrib["sentence"]):
                            t_id += 1
                        else:
                            old_sent = int(elem.attrib["sentence"])
                            t_id = 0

                        token_dict[elem.attrib["t_id"]] = {"text": elem.text, "sent": elem.attrib["sentence"], "id": t_id}

                        if ECB in topic_file:
                            if int(elem.attrib["sentence"]) == 0:
                                title = append_text(title, elem.text)
                            else:
                                text = append_text(text, elem.text)

                        if ECBPLUS in topic_file:
                            sent_dict[int(elem.attrib["sentence"])] = append_text(sent_dict.get(
                                                                    int(elem.attrib["sentence"]), ""), elem.text)
                    except KeyError:
                        pass

                    if elem.tag == "Markables":
                        for i, subelem in enumerate(elem):
                            tokens = [token.attrib["t_id"] for token in subelem]

                            if len(tokens):
                                mentions[subelem.attrib["m_id"]] = {"type": subelem.tag,
                                                                    "text": " ".join([token_dict[t]["text"] for t in tokens]),
                                                                    "token_numbers": [token_dict[t]["id"] for t in tokens],
                                                                    "doc_id": topic_file.split(".")[0],
                                                                    "sent_id": token_dict[tokens[0]]["sent"]}
                            else:
                                try:
                                    if subelem.attrib["instance_id"] not in coref_dict:
                                        coref_dict[subelem.attrib["instance_id"]] = {"descr": subelem.attrib["TAG_DESCRIPTOR"]}
                                    # m_id points to the target
                                except KeyError:
                                    pass
                                    # unknown_tag = subelem.tag
                                    # coref_dict[unknown_tag + subelem.attrib["m_id"]] = {"descr": subelem.attrib["TAG_DESCRIPTOR"]}

                    if elem.tag == "Relations":
                        mentions_map = {m: False for m in list(mentions)}
                        for i, subelem in enumerate(elem):
                            try:
                                if "r_id" not in coref_dict[subelem.attrib["note"]]:
                                    coref_dict[subelem.attrib["note"]].update({
                                        "r_id": subelem.attrib["r_id"],
                                        "coref_type": subelem.tag,
                                        "mentions": [mentions[m.attrib["m_id"]] for m in subelem if m.tag == "source"]
                                    })
                                else:
                                    coref_dict[subelem.attrib["note"]]["mentions"].extend([mentions[m.attrib["m_id"]] for m in subelem if m.tag == "source"])
                            except KeyError:
                                # coref_dict[unknown_tag + "".join([m.attrib["m_id"] for m in subelem if m.tag == "target"])].update({
                                #     "r_id": subelem.attrib["r_id"],
                                #     "coref_type": subelem.tag,
                                #     "mentions": [mentions[m.attrib["m_id"]] for m in subelem if m.tag == "source"]
                                # })
                                pass
                            for m in subelem:
                                mentions_map[m.attrib["m_id"]] = True

                        # for i, (m_id, used) in enumerate(mentions_map.items()):
                        #     if used:
                        #         continue
                        #
                        #     m = mentions[m_id]
                        #     if "Singleton_" + m["type"][:4] + "_" + str(m_id) + "_" + m["doc_id"] not in coref_dict:
                        #         coref_dict["Singleton_" + m["type"][:4] + "_" + str(m_id) + "_" + m["doc_id"]] = {
                        #             "r_id": str(10000 + i),
                        #             "coref_type": "Singleton",
                        #             "mentions": [m],
                        #             "descr": ""
                        #         }
                        #     else:
                        #         coref_dict["Singleton_" + m["type"][:4] + "_" + str(m_id) + "_" + m["doc_id"]].update(
                        #             {
                        #                 "r_id": str(10000 + i),
                        #                 "coref_type": "Singleton",
                        #                 "mentions": [m],
                        #                 "descr": ""
                        #             })
                    a = 1

                newsplease_custom = copy.copy(newsplease_format)

                if ECBPLUS in topic_file:

                    sent_df = pd.DataFrame(columns=[IS_TEXT, TEXT])

                    for sent_key, text in sent_dict.items():
                        sent_df = sent_df.append(pd.DataFrame({
                            IS_TEXT: 0 if len(re.sub(r'[\D]+', "", text)) / len(text) >= 0.1 or sent_key == 0 else 1,
                            TEXT: text
                        }, index=[sent_key]))

                    doc_sent_map[topic_file.split(".")[0]] = sent_df

                    small_df = sent_df[sent_df[IS_TEXT] == 0]
                    newsplease_custom["url"] = list(small_df[TEXT].values)[0] if len(small_df) > 0 else ""
                    newsplease_custom["date_publish"] = " ".join(list(small_df[TEXT].values)[1:]) if len(small_df) > 1 else ""
                    newsplease_custom["title"] = list(sent_df[sent_df[IS_TEXT] == 1][TEXT].values)[0]
                    text = " ".join(list(sent_df[sent_df[IS_TEXT] == 1][TEXT].values)[1:])

                else:
                    newsplease_custom["title"] = title
                    newsplease_custom["date_publish"] = None

                if len(text):
                    text = text if text[-1] != "," else text[:-1] + "."

                newsplease_custom["text"] = text
                newsplease_custom["source_domain"] = topic_file.split(".")[0]
                if newsplease_custom["title"][-1] not in string.punctuation:
                    newsplease_custom["title"] += "."

                doc_files[topic_file.split(".")[0]] = newsplease_custom

                if topic_name not in os.listdir(result_path):
                    os.mkdir(os.path.join(result_path, topic_name))

                with open(os.path.join(result_path, topic_name, newsplease_custom["source_domain"] + ".json"), "w") as file:
                    json.dump(newsplease_custom, file)

            coref_dics[topic_folder] = coref_dict

            entity_mentions = []
            event_mentions = []

            for chain_id, chain_vals in coref_dict.items():
                for m in chain_vals["mentions"]:
                    sent_id = 0

                    if ECBPLUS.split(".")[0] not in m['doc_id']:
                        sent_id = int(m["sent_id"])
                    else:
                        df = doc_sent_map[m["doc_id"]]
                        sent_id = np.sum([df.iloc[:list(df.index).index(int(m["sent_id"])) + 1][IS_TEXT].values])

                    token_numbers = [int(t) for t in m["token_numbers"]]
                    mention_id = m["doc_id"] + "_" + str(chain_id) + "_" + str(m["sent_id"]) + "_" + str(m["token_numbers"][0])
                    mention = {"coref_chain": chain_id,
                               "doc_id": m["doc_id"],
                               "is_continuous":  True if token_numbers == list(range(token_numbers[0], token_numbers[-1] + 1))
                                                    else False,
                               "is_singleton": len(chain_vals["mentions"]) == 1,
                               "mention_id": mention_id,
                               "mention_type": m["type"][:3],
                               "mention_full_type": m["type"],
                               "score": -1.0,
                               "sent_id": sent_id,
                               "tokens_number": token_numbers,
                               "tokens_str": m["text"],
                               "topic_id": topic_name,
                               "coref_type": chain_vals["coref_type"],
                               "decription": chain_vals["descr"]
                               }
                    if chain_id[:3] in ["ACT", "NEG"]:
                        event_mentions.append(mention)
                    else:
                        entity_mentions.append(mention)
                    summary_df = summary_df.append(pd.DataFrame({
                        "doc_id" :  m["doc_id"],
                        "coref_chain": chain_id,
                        "decription": chain_vals["descr"],
                        "short_type": chain_id[:3],
                        "full_type": m["type"],
                         "tokens_str": m["text"]
                    }, index=[mention_id]))

            annot_path = os.path.join(result_path,  topic_name, "annotation", "original")
            if topic_name not in os.listdir(os.path.join(result_path)):
                os.mkdir(os.path.join(result_path, topic_name))

            if "annotation" not in os.listdir(os.path.join(result_path, topic_name)):
                os.mkdir(os.path.join(result_path, topic_name, "annotation"))
                os.mkdir(annot_path)

            with open(os.path.join(annot_path, "entity_mentions_" + topic_name + ".json"), "w") as file:
                json.dump(entity_mentions, file)

            with open(os.path.join(annot_path, "event_mentions_" + topic_name + ".json"), "w") as file:
                json.dump(event_mentions, file)

    now = datetime.now()
    summary_df.to_csv(os.path.join(result_path, now.strftime("%Y-%m-%d_%H-%M") + "_" + "all" + ".csv"))


if __name__ == '__main__':
    try:
        topic_num = int(input("How many topics from ECBplus corpus do you want to convert? \nType the number of anything is you want "
                   "to convert all 45 topics: "))
        convert_files(topic_num)
    except (NameError, ValueError, IndexError):
        topic_num = len(os.listdir(ecb_path))
        convert_files(topic_num)

    print("\nConversion of {0} topics from xml to newsplease format and to annotations in a json file is "
             "done. \n\nFiles are saved to {1}. \nCopy the topics on which you want to execute Newsalyze to "
          "{2}.".format(str(topic_num), result_path, ORIGINAL_DATA_PATH))
