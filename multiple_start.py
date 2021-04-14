"""
Start evaluation multiple execution of the Newsalyze pipeline.
"""

import sys
import os
import re
import pandas as pd
import json
import time
from datetime import datetime
from datetime import timedelta

file_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{file_dir}/newstsc")
os.chdir(os.path.join(file_dir))

from cdcr.config import ConfigLoader
from cdcr.config import LOGGER
from cdcr.pipeline import Pipeline
from cdcr.candidates.cand_enums import *
from cdcr.config import ROOT_ABS_DIR, DATA_PATH, ORIGINAL_DATA_PATH, EVALUATION_PATH
from cdcr.util.cache import Cache
import cdcr.logger as logging
from cdcr.pipeline.modules import NewsPleaseReader, Preprocessor, CandidateExtractor
from cdcr.structures.configuration import CUSTOM_CAND_METHOD_NAME, CANDIDATES, ENTITIES
from cdcr.pipeline.modules.news_please_reader import UserInterface
file_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{file_dir}/newstsc")


CAND_ORIGIN_TYPE, CAND_COREF, CAND_CHANGE_HEAD, CAND_EXTENSION, DATASET, CAND_ANNOT_ID, ENT_METHOD, ENT_PARAM, ENT_FILE,\
ENT_WV =  "candidates_origin_type", "candidates_coref_extraction_strategy", "candidates_change_head", \
          "candidates_phrase_extension", "dataset", "candidates_annot_index", "entities_method", "entities_param_source", \
          "entities_custom_file_id", "entities_word_vectors"

ORIGIN_TYPE = "origin_type"
COREF_STRATEGY = "coref_strategy"
ENTITY_METHOD = "identification_method"
PARAMS = "parameters"
WORDVECTORS = "wordvectors"
TOPIC_NAME = "topic_name"
ANNOT_ID = "annot_dataset"
CONFIG = "config"
CHANGE_HEAD = "change_head"
PHRASE_EXTENSION = "phrase_extension"

TOPIC, NUM_WORDS, NUM_ARTICLES, NUM_MENTIONS, MODULE, SETUP, TIME = "topic", "num_words", "num_articles", "num_mentions",\
                                                                    "module", "setup", "time"

if __name__ == "__main__":
    conf = ConfigLoader.load_and_apply()
    logging.setup()
    conf.log()

    cache = Cache()

    data_folders = UserInterface.get_available_data_folders()
    config_df = pd.read_csv(os.path.join(ROOT_ABS_DIR, "cdcr/util/evaluation/experiments_all.csv"), index_col=[0])
    config_df.fillna("not_specified", inplace=True)

    with open(os.path.join(DATA_PATH, "ECBplus-prep", "test_events.json"), "r") as file:
        ecb_plus_topic_list = json.load(file)
        ecb_plus_topic_list = list(set([t.split("_")[0] + re.findall(r'[a-z]+', t)[0] for t in ecb_plus_topic_list]))

    stats_df = pd.DataFrame(columns=[TOPIC, DATASET, NUM_WORDS, NUM_ARTICLES, NUM_MENTIONS, MODULE, SETUP, TIME])
    stats_id = 0
    topic_index = 0

    for topic in data_folders:

        # if "ecbplus" in topic:
        #     continue
        #
        # if "ecb" not in topic:
        #     continue

        if len(os.listdir(os.path.join(ORIGINAL_DATA_PATH, topic))) == 1:
            continue

        if not os.path.exists(os.path.join(ORIGINAL_DATA_PATH, topic, "annotation")):
            LOGGER.info("The topic {} doesn't have an annotation folder and will be skipped. ".format(topic.upper()))
            continue

        dataset = "newswcl50"

        if len(re.findall(r'[0-9]+ecb', topic)):
            if topic not in ecb_plus_topic_list:
                continue
            dataset = "ecb"

        LOGGER.info("Executing pipeline on {} topic: {}. ".format(str(topic_index), topic.upper()))

        pipeline_setup = {
            "reading": {
                "module": NewsPleaseReader.run,
                "caching": False
            },
            "preprocessing": Preprocessor.run}
        topic_preproc = None
        for l in sorted(cache.list(), reverse=True):
            if (("ecb" in topic and any([topic == ll for ll in l.split("_")])) or
                                                       ("ecb" not in topic and topic in l)) and "preprocessing" in l:
                topic_preproc = l
                break

        if topic_preproc is None:
            docset = Pipeline(pipeline_setup).run(cache_file=topic, configuration={})
        else:
            docset = Pipeline(pipeline_setup).run(cache_file=topic_preproc, configuration={})
        num_words = len([t for doc in docset for sent in doc.sentences for t in sent.tokens])
        num_docs = len(docset)

        config_dataset_df = config_df[config_df[DATASET] == dataset]
        cand_df = config_dataset_df.groupby(by=[CAND_ORIGIN_TYPE, CAND_COREF, CAND_CHANGE_HEAD, CAND_EXTENSION,
                                                CAND_ANNOT_ID])
        config_index = 0

        existing_cand_docsets = []
        file_names_cand = []
        existing_cand_docsets_df = pd.DataFrame()
        file_id = 0
        for cached_file in cache.list()[::-1]:
            if CANDIDATES in cached_file and (("ecb" in topic and any([topic == ll for ll in cached_file.split("_")])) or
                                                       ("ecb" not in topic and topic in cached_file)):
                docset = Pipeline({"preprocessing": Preprocessor.run, "candidates": CandidateExtractor.run}).run(
                    cache_file=cached_file, configuration={})
                existing_cand_docsets.append(docset)
                file_names_cand.append(cached_file)
                existing_cand_docsets_df = existing_cand_docsets_df.append(pd.DataFrame({
                    CAND_ORIGIN_TYPE: docset.configuration.cand_extraction_config.origin_type.name.lower(),
                    CAND_COREF: docset.configuration.cand_extraction_config.coref_extraction_strategy.name.lower(),
                    CAND_CHANGE_HEAD: docset.configuration.cand_extraction_config.change_head.name.lower(),
                    CAND_EXTENSION: docset.configuration.cand_extraction_config.phrase_extension.name.lower(),
                    CAND_ANNOT_ID: docset.configuration.cand_extraction_config.annot_index
                }, index=[file_id]))
                file_id += 1

        for cand_group_name, cand_group_df in cand_df:
            origin_type, coref, change_head, extension, annot_id = cand_group_name

            current_config_df = pd.DataFrame({
                CAND_ORIGIN_TYPE: origin_type.lower(),
                CAND_COREF: coref.lower(),
                CAND_CHANGE_HEAD: change_head.lower(),
                CAND_EXTENSION: extension.lower(),
                CAND_ANNOT_ID: annot_id
            }, index=[0])

            cand_docset = None
            start_time, end_time = 0, 0
            # check is this config is already saved
            for cand_docset_index, row in existing_cand_docsets_df.iterrows():
                if all(row == current_config_df.loc[0]):
                    cand_docset = existing_cand_docsets[cand_docset_index]
                    LOGGER.info("LOADED a candidate extraction module with the following settings: ({}).".format(
                                                                ", ".join([topic] + [str(v) for v in list(cand_group_name)])))
                    break
            cand_file_for_entities = None

            if cand_docset is None:
                config_cand = {
                    "_run_config": {
                        CANDIDATES: {CUSTOM_CAND_METHOD_NAME: {
                                "annot_index": annot_id,
                                "origin_type": OriginType.from_string(origin_type),
                                "coref_extraction_strategy": CorefStrategy.from_string(coref),
                                "change_head": ChangeHead.from_string(change_head),
                                "phrase_extension": ExtentedPhrases.from_string(extension)
                        }}}
                }
                pipeline_setup_cand = {"preprocessing": Preprocessor.run, "candidates": CandidateExtractor.run}

                for cached_file in cache.list()[::-1]:
                    if "preprocessing" in cached_file and (("ecb" in topic and any([topic == ll for ll in cached_file.split("_")])) or
                                                       ("ecb" not in topic and topic in cached_file)):
                        start_time = time.time()
                        LOGGER.info("A candidate extraction module is being executed with the following settings: ({}).".format(
                            ", ".join([topic] + [str(v) for v in list(cand_group_name)])))
                        cand_docset = Pipeline(pipeline_setup_cand).run(cache_file=cached_file, configuration=config_cand)
                        end_time = time.time()
                        # cand_docset.processing_information.cand_execution_time = end_time - start_time
                        for cached_file in cache.list()[::-1]:
                            if CANDIDATES in cached_file and topic in cached_file:
                                cand_file_for_entities = cached_file
                                break

                if cand_docset is None:
                    raise ValueError("No candidate extraction was performed. Check the pipeline execution.")

            num_mentions = len([cand for cand_group in cand_docset.candidates for cand in cand_group])

            if end_time - start_time > 0:
                stats_df = stats_df.append(pd.DataFrame({
                    TOPIC: topic,
                    DATASET: dataset + "_" + os.path.basename(cand_docset.configuration.cand_extraction_config.annot_path),
                    NUM_WORDS: num_words,
                    NUM_ARTICLES: num_docs,
                    NUM_MENTIONS: num_mentions,
                    MODULE: CANDIDATES,
                    SETUP: "__".join([str(v) for v in list(cand_group_name)]),
                    TIME: end_time - start_time
                }, index=[stats_id]))
                stats_id += 1

            existing_ent_docsets = []
            existing_ent_docsets_df = pd.DataFrame()
            file_ent_id = 0
            for cached_file in cache.list()[::-1]:
                if ENTITIES in cached_file and (("ecb" in topic and any([topic == ll for ll in cached_file.split("_")])) or
                                                       ("ecb" not in topic and topic in cached_file)):
                    docset = Pipeline({"candidates": CandidateExtractor.run, "entities": EntityIdentifier.run}).run(
                        cache_file=cached_file, configuration={})
                    existing_ent_docsets.append(docset)
                    existing_ent_docsets_df = existing_ent_docsets_df.append(pd.DataFrame({
                        CAND_ORIGIN_TYPE: docset.configuration.cand_extraction_config.origin_type.name.lower(),
                        CAND_COREF: docset.configuration.cand_extraction_config.coref_extraction_strategy.name.lower(),
                        CAND_CHANGE_HEAD: docset.configuration.cand_extraction_config.change_head.name.lower(),
                        CAND_EXTENSION: docset.configuration.cand_extraction_config.phrase_extension.name.lower(),
                        CAND_ANNOT_ID: docset.configuration.cand_extraction_config.annot_index,
                        ENT_METHOD: docset.configuration.entity_method.lower(),
                        ENT_PARAM: docset.configuration.entity_identifier_config.param_source.lower(),
                        ENT_FILE: docset.configuration.entity_identifier_config.custom_files_id,
                        ENT_WV: docset.configuration.entity_identifier_config.word_vectors.lower()
                    }, index=[file_ent_id]))
                    file_ent_id += 1

            entity_global_df = cand_group_df.groupby(by=[ENT_METHOD])

            prev_method = None
            prev_break = False
            for method, entity_global_group_df in entity_global_df:

                entity_df = entity_global_group_df.groupby(by=[ENT_PARAM, ENT_FILE, ENT_WV])
                ent_config_id = 0
                for ent_group_name, entity_group_df in entity_df:
                    params, filed_id, wv = ent_group_name

                    current_config_df = pd.DataFrame({
                        CAND_ORIGIN_TYPE: origin_type.lower(),
                        CAND_COREF: coref.lower(),
                        CAND_CHANGE_HEAD: change_head.lower(),
                        CAND_EXTENSION: extension.lower(),
                        CAND_ANNOT_ID: annot_id,
                        ENT_METHOD: method.lower(),
                        ENT_PARAM: params.lower(),
                        ENT_FILE: filed_id,
                        ENT_WV: wv.lower()
                    }, index=[0])

                    to_break = False

                    for entity_index, row in existing_ent_docsets_df.iterrows():
                        if all(row == current_config_df.loc[0]):
                            LOGGER.info("Config index {0} ({1}) exists and will be skipped.".format(str(config_index),
                                ", ".join([topic] + [str(v) for v in list(cand_group_name) + [method] + list(ent_group_name)])))
                            config_index += 1
                            to_break = True
                            prev_break = True
                            break

                    if to_break:
                        continue

                    prev_break = False

                    LOGGER.info("Config index {0} ({1}) is being executed.".format(str(config_index),
                          ", ".join([topic] + [str(v) for v in list(cand_group_name) + [method] + list(ent_group_name)])))

                    config_ent = {
                        "entity_method": method,
                        "_run_config": {
                            ENTITIES: {method: {
                                "param_source": params,
                                "custom_files_id": filed_id,
                                "word_vectors": wv,
                                "load_preproc": ent_config_id != 0 or (method == "msma3" and prev_method == "msma2"),
                                "evaluation": {"evaluation_mode": True}
                            }}
                        }}
                    ent_config_id += 1

                    pipeline_setup_ent = {"candidates": CandidateExtractor.run, "entities": EntityIdentifier.run}
                    docset = None
                    start_time = time.time()
                    docset = Pipeline(pipeline_setup_ent).run(cache_file=file_names_cand[cand_docset_index]
                                if cand_file_for_entities is None else cand_file_for_entities, configuration=config_ent)
                    end_time = time.time()

                    stats_df = stats_df.append(pd.DataFrame({
                        TOPIC: topic,
                        DATASET: dataset + "_" + os.path.basename(docset.configuration.cand_extraction_config.annot_path),
                        NUM_WORDS: num_words,
                        NUM_ARTICLES: num_docs,
                        NUM_MENTIONS: num_mentions,
                        MODULE: ENTITIES,
                        SETUP: method + "_" + wv,
                        # SETUP: "_".join(list(ent_group_name)),
                        TIME: end_time - start_time
                    }, index=[stats_id]))
                    stats_id += 1
                    config_index += 1

                if not prev_break:
                    prev_method = method
        topic_index += 1
    now = datetime.now()
    basis_name = now.strftime("%Y-%m-%d_%H-%M") + "_benchmark"
    stats_df.to_csv(os.path.join(EVALUATION_PATH, basis_name + "_detailed.csv"), index=False)

    stats_groups_df = stats_df[[DATASET, NUM_WORDS, NUM_ARTICLES, NUM_MENTIONS, MODULE, SETUP, TIME]].groupby(by=[DATASET, MODULE, SETUP]).mean()
    stats_groups_df["time_hours"] = [str(timedelta(seconds=t)) for t in stats_groups_df[TIME].values]
    stats_groups_df.to_csv(os.path.join(EVALUATION_PATH, basis_name + "_grouped.csv"), index=False)
    LOGGER.info("Experimental multiple execution of the Pipeline is over. ")
