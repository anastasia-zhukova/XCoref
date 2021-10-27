import cdcr.config as config
from cdcr.util.corenlp_caller import CoreNLPCaller
from cdcr.candidates.cand_enums import *
from cdcr.structures.candidate_set import CandidateSet
from cdcr.structures.candidate_group_set import CandidateGroupSet
from cdcr.config import *
from cdcr.entities.dict_lists import LocalDictLists
from cdcr.candidates.coref_df import *
# from newsalyze.structures.params import DummyUpdater

import logging
import progressbar
import pandas as pd
import numpy as np
import copy
import re
import string


NOTIFICATION_MESSAGES = {
    "progress": "PROGRESS: Extracted coreferencial candidates from %(value)d (%(percentage)d %%) coreference groups"
                " (in: %(elapsed)s).",
    "np_progress": "PROGRESS: Extracted NP candidates from %(value)d/%(max_value)d (%(percentage)d %%) sentences (in: %(elapsed)s).",
    "mention_progress": "PROGRESS: Extracted mentions from %(value)d/%(max_value)d (%(percentage)d %%) document groups (in: %(elapsed)s).",
    "vp_progress": "PROGRESS: Extracted VP candidates from %(value)d/%(max_value)d (%(percentage)d %%) sentences (in: %(elapsed)s).",
    "annot_progress": "PROGRESS: Extracted candidates from %(value)d/%(max_value)d (%(percentage)d %%) annotations (in: %(elapsed)s) "
                      "from file {0}.",
    "no_tree": "No parent tree of a candidate extracted.",
    "corefs": "Extracting coreferences out of the combined documents...",
    "number_dismatch": "Number of original sentences does not match the number of extracted sentences on the combined "
                       "documents. The coreferences will ba taken from original documents not a combined mega-document.",
    "wrong_config": "Wrong configuration: candidate extraction strategy involved automated extracting phrases "
                    "and add_phrases are empty. "
                    "Please specify in a json config file what to extract, e.g., add_phrases = [\"NP\"]. ",
    "token_mismatch": "{0} ({1}) extracted tokens don\'t fully match {2} ({3}) annotated tokens.",
    "no_tokens": "List of tokens is zero, the annotated mention is skipped. ",
    "not_found": "Mention \"{0}\" is not found in text (article: {1}, sent: {2}), the annotated mention is skipped.",
    "no_annot": "Annotated candidates are required to be extracted but no annotation provided. "
                "Extracted candidated will be used instead.",
    "drop_dupbl": "Dropped duplicates. Candidate groups: {0} --> {1}."
}

ORIG_SENT = "orig_sent_id"
ORIG_DOC = "orig_doc"
DOC_NUM = "doc_number"
ANNOTATIONS = "annotations"
ENTITY = "entity"
EVENT = "event"
SEARCH_WINDOW = 2

DOC_ID = "doc_id"
SENT_ID = "sent_id"
HEAD_ID = "head_id"
BEGIN_ID = "begin_id"
END_ID = "end_id"
PHRASE = "phrase"
COREF = 'coref'
CAND_ID = "cand_id"
ANNOT = "_annot"
EXTRACT = "_extract"


# class Annot_Mention(DummyUpdater):
class Annot_Mention:
    def __init__(self, iterable: dict, **kwargs):
        # super().__init__(iterable, **kwargs)
        iterable.update(kwargs)
        for k,v in iterable.items():
            setattr(self, k, v)

    def __repr__(self):
        return self.tokens_str


class CandidatePhrasesExtractor:

    _logger = LOGGER

    def __init__(self):
        self.docs = None
        self.params = None
        self.doc_mask = None

        self._add_phrases = {CandidateType.NP: self.extract_NP_candidates,
                             CandidateType.VP: self.extract_VP_candidates}
        #  TODO also extract OTHER phrases, e.g., only adj or adverb phrases (which can contain significant wcl)

    def extract_phrases(self, docs):

        if docs.configuration.cand_extraction_config.origin_type != OriginType.ANNOTATED \
                and not len(docs.configuration.cand_extraction_config.add_phrases):
            raise ValueError(NOTIFICATION_MESSAGES["wrong_config"])

        self.docs = docs
        self.params = docs.configuration.cand_extraction_config

        # Check if annotations exist: load if any or return empty table
        print()

        annot_cand_set = self.read_annotations()
        coref_cand_set = self.extract_coref_candidates()

        annot_df, coref_df = (create_cand_table(cand_set, suffix, self.params.drop_duplicates)
                                for suffix, cand_set in {ANNOT: annot_cand_set,
                                                         EXTRACT: coref_cand_set}.items())

        if self.params.origin_type != OriginType.ANNOTATED or not len(annot_df):
            if not len(annot_df) and self.params.origin_type == OriginType.ANNOTATED:
                self._logger.warning(NOTIFICATION_MESSAGES["no_annot"])

            for add_phrases in self.params.add_phrases:
                if add_phrases not in self._add_phrases:
                    continue

                add_set_extended = self._add_phrases[add_phrases](coref_df)
                for cand_group in add_set_extended:
                    coref_cand_set.append(cand_group)

            extract_df = create_cand_table(coref_cand_set, EXTRACT, self.params.drop_duplicates)

            if not len(annot_df) or self.params.origin_type == OriginType.EXTRACTED:
                return CandidatePhrasesExtractor.remove_duplicates(coref_cand_set, extract_df)

            merged_df = extract_df.reset_index().merge(annot_df.reset_index(), how="left", suffixes=(EXTRACT, ANNOT),
                                                       left_on=[col + EXTRACT for col in [DOC_ID, SENT_ID, HEAD_ID]],
                                                       right_on=[col + ANNOT for col in [DOC_ID, SENT_ID, HEAD_ID]],
                                                       sort=False).set_index("index" + EXTRACT).sort_values(
                by=[DOC_ID + EXTRACT, SENT_ID + EXTRACT])

            annot_cand_dict = {cand.id: i for i, cand_group in enumerate(annot_cand_set) for cand in cand_group}
            mapped_annot = merged_df.dropna()

            mapped_cands = {}
            for cand_group in coref_cand_set:
                for cand in cand_group:
                    if cand.id in list(mapped_annot.index):
                        mapped_cands[cand.id] = cand

            not_matched_attrs = list(set(annot_cand_set[0][0].__dict__) - set(coref_cand_set[0][0].__dict__))
            annot_attrs = [attr for attr in list(coref_cand_set[0][0].__dict__) if "annot" in attr]
            for cand_id, row in mapped_annot.iterrows():
                annot_id = row["index"+ANNOT]
                annot_cand = annot_cand_set[annot_cand_dict[annot_id]][0]
                for attr in annot_attrs + not_matched_attrs:
                    setattr(mapped_cands[cand_id], attr, getattr(annot_cand, attr))

            not_mapped = annot_df.loc[list(set(list(annot_df.index)) - set(mapped_annot["index" + ANNOT].values))].sort_values(
                by=DOC_ID + ANNOT)

            add_cand = [annot_cand_set[annot_cand_dict[index]] for index in list(not_mapped.index)]
            coref_cand_set = CandidateSet(coref_strategy=coref_cand_set.coref_strategy,
                                          origin_type=OriginType.EXTRACTED_ANNOTATED,
                                      items=sorted(coref_cand_set + add_cand, reverse=True, key=lambda x: len(x)))
            return CandidatePhrasesExtractor.remove_duplicates(coref_cand_set, extract_df.append(not_mapped))
        else:
            coref_dict = {c.id: g.group_name for g in coref_cand_set for c in g}
            annot_dict = {c.id: c for g in annot_cand_set for c in g}
            # output_cand_set = copy.deepcopy(annot_cand_set)

            result_df = pd.merge(annot_df.reset_index(), coref_df.reset_index(), how='left', left_on=['doc_id_annot',
                                                                                  'sent_id_annot', "head_id_annot"],
                              right_on=['doc_id_extract', 'sent_id_extract', "head_id_extract"]).set_index("index_x")
            result_df.rename(columns={"index_y": CAND_ID}, inplace=True)
            result_df.dropna(inplace=True)
            result_df[COREF] = [coref_dict[v] for v in result_df[CAND_ID].values]

            for cand_group_key in set(result_df[COREF].values):
                df = result_df[result_df[COREF] == cand_group_key]

                if len(df) == 1:
                    continue

                cand_group = CandidateGroupSet(cand_group_key, [annot_dict[k] for k in list(df.index)])
                annot_cand_set.append(cand_group)
            annot_cand_set = CandidateSet(coref_strategy=annot_cand_set.coref_strategy,
                                          origin_type=annot_cand_set.origin_type,
                                          items=sorted(annot_cand_set, reverse=True, key=lambda x: len(x)))

            while not len(annot_cand_set[-1]):
                annot_cand_set.pop(-1)

            return CandidatePhrasesExtractor.remove_duplicates(annot_cand_set, annot_df)

    @staticmethod
    def remove_duplicates(cand_set, df):
        init_len = len(cand_set)
        df_list = list(df.index)
        cand_group_id = 0
        while cand_group_id < len(cand_set):
            cand_group = cand_set[cand_group_id]
            cand_id = 0
            while cand_id < len(cand_group):
                cand = cand_group[cand_id]
                if cand.id not in df_list:
                    cand_group.pop(cand_id)
                cand_id += 1
            if not len(cand_group):
                cand_set.pop(cand_group_id)
            cand_group_id += 1
        LOGGER.info(NOTIFICATION_MESSAGES["drop_dupbl"].format(str(init_len), str(len(cand_set))))
        return cand_set

    @staticmethod
    def find_phrase(doc, mention, search_window=0):
        # small_phrase = [t.lower() for t in mention.tokens_str.split(" ")]
        if search_window:
            sents = doc.sentences[max(mention.sent_id - search_window, 0): mention.sent_id + search_window + 1]
        else:
            sents = [doc.sentences[mention.sent_id]]

        # mention_proc = re.sub(r'\s+', " ", re.sub(r'\W', " ", mention.tokens_str.lower()))
        # mention_proc = re.sub(r'\W', " ", mention.tokens_str.lower().replace(" 's", "'s"))
        mention_proc = mention.tokens_str
        for i, sent in enumerate(sents):
            # sent_proc = re.sub(r'\s+', " ", re.sub(r'\W', " ", sent.text.lower()))
            # sent_proc = re.sub(r'\W', " ", sent.text.lower())
            sent_proc = sent.text
            for str_mention in re.finditer(mention_proc.replace("(", "<").replace(")", ">"),
                                           sent_proc.replace("(", "<").replace(")", ">")):
                segm_index_start = str_mention.start()

                token_ids = []
                for token in sent.tokens:
                    if token.sentence_begin_char >= segm_index_start and \
                            token.sentence_end_char <= segm_index_start + len(mention.tokens_str):
                        token_ids.append(token.index)

                if not len(token_ids):
                    continue

                return sent.index, sent.tokens[min(token_ids): max(token_ids) + 1]
        return None, None

    @staticmethod
    def find_phrase_tokens(doc, mention, search_window=0):
        small_phrase = [t.lower() for t in mention.tokens_str.split(" ")]
        if search_window:
            sents = doc.sentences[max(mention.sent_id - search_window, 0): mention.sent_id + search_window + 1]
        else:
            sents = [doc.sentences[mention.sent_id]]

        for i, sent in enumerate(sents):
            main_phrase = [t.word.lower() for t in sent.tokens]
            sim = []
            m = 0
            k = 0
            search = True
            while search and k < len(main_phrase):
                if small_phrase[m] == main_phrase[k]:
                    sim.append(small_phrase[m])
                    m += 1
                # elif len(small_phrase[m]) >= 4 and small_phrase[m] in main_phrase[k]:
                elif re.split(r'\W', small_phrase[m])[0] == re.split(r'\W', main_phrase[k])[0]:
                    sim.append(small_phrase[m])
                    m += 1
                # elif len(main_phrase[k]) >= 4 and main_phrase[k] in small_phrase[m]:
                #     sim.append(small_phrase[m])
                #     m += 1
                if m == len(small_phrase):
                    search = False
                else:
                    k += 1
            if len(small_phrase) - 1 <= len(sim) <= len(small_phrase):
                return sent.index, sent.tokens[k - len(small_phrase) + 1: k + 1]
        return None, None

    def read_annotations(self):

        if self.params.annot_path is None or self.params.origin_type == OriginType.EXTRACTED:
            return CandidateSet(OriginType.ANNOTATED, self.params.coref_extraction_strategy)

        doc_dict = {doc.source_domain: doc for doc in self.docs}
        annot_candidates = CandidateSet(OriginType.ANNOTATED, self.params.coref_extraction_strategy)

        for file in os.listdir(self.params.annot_path):

            with open(os.path.join(self.params.annot_path, file), "rb") as json_file:
                annotations = json.load(json_file)

            widgets = [
                progressbar.FormatLabel(NOTIFICATION_MESSAGES["annot_progress"].format(file))
            ]
            bar = progressbar.ProgressBar(widgets=widgets,
                                          maxval=len(annotations)).start()

            for m_id, mention in enumerate(annotations):
                mention = Annot_Mention(mention)
                doc = doc_dict[mention.doc_id]

                if mention.sent_id < len(doc.sentences):
                    sent_id, tokens = CandidatePhrasesExtractor.find_phrase(doc, mention)

                if mention.sent_id >= len(doc.sentences) or tokens is None:
                    sent_id, tokens = CandidatePhrasesExtractor.find_phrase(doc, mention, SEARCH_WINDOW)

                if tokens is None:
                    sent_id, tokens = CandidatePhrasesExtractor.find_phrase_tokens(doc, mention, SEARCH_WINDOW)
                    if tokens is None:
                        if len(re.sub(r'\D+', "", mention.tokens_str)) / len(re.sub(r'\W+', "", mention.tokens_str)) >= 0.1:
                            continue
                        self._logger.warning(NOTIFICATION_MESSAGES["not_found"].format(mention.tokens_str, mention.doc_id,
                                                                                       mention.sent_id))
                        continue

                if "".join([t.word for t in tokens]) == "``":
                    tokens = [doc.sentences[sent_id].tokens[t.index + 2] for t in tokens]

                # max_id = min_id + len(mention.tokens_number)
                # tokens = doc.sentences[sent_id].tokens[min_id: max_id]

                if not len(tokens):
                    self._logger.warning(NOTIFICATION_MESSAGES["no_tokens"])
                    continue

                annot_phrase = re.sub(r'\s+', " ", re.sub(r'\W', " ", mention.tokens_str.lower())).strip()
                found_phrase = re.sub(r'\s+', " ", re.sub(r'\W', " ", " ".join([t.word.lower() for t in tokens]))).strip()
                if not len(found_phrase.strip()):
                    self._logger.warning(NOTIFICATION_MESSAGES["no_tokens"])
                    continue

                if found_phrase != annot_phrase:
                    if tokens[-1].word in string.punctuation:
                            tokens = tokens[:-1]
                    else:
                        self._logger.warning(
                            NOTIFICATION_MESSAGES["token_mismatch"].format(len(tokens), found_phrase,
                                                                           len(mention.tokens_number),
                                                                           annot_phrase))

                sentence = doc.sentences[sent_id]
                dep_subtree = Candidate.find_dep_subset_static(tokens, sentence)
                head_token = Candidate.find_head_static(tokens, dep_subtree)

                cand_params = {"text": mention.tokens_str,
                                "sentence": sentence,
                                "tokens": tokens,
                                "is_representative": True,
                                "cand_id": mention.mention_id,
                                "head_token": head_token,
                                "enhancement": self.params.phrase_extension,
                                "change_head": self.params.change_head,
                                "origin_type": OriginType.ANNOTATED,
                                "annot_label": mention.coref_chain,
                                "annot_type": mention.mention_type,
                }

                if "ecb" in self.docs.topic and "ecbplus2" not in self.docs.topic:
                    cand_params.update({"annot_type_full": mention.mention_full_type,
                                     "annot_coref_type": mention.coref_type,
                                     "is_continuous": mention.is_continuous,
                                     "is_singleton": mention.is_singleton,
                                     "file_orig": file.split("_")[0]})

                # TODO add specific fields for NewsWCL and NIDENT

                if "NN" in head_token.token.pos:
                    cand_params.update({"cand_type": CandidateType.NP, "coref_subtype": "NP"})

                elif "V" in head_token.token.pos:
                    cand_params.update({"cand_type": CandidateType.VP, "coref_subtype": "VP"})

                else:
                    cand_params.update({"cand_type": CandidateType.OTHER, "coref_subtype": "OTHER"})

                cand = Candidate(**cand_params)
                cand_repr = "_".join([t.word for t in cand.tokens[:10]]) + "_" + str(doc.id) + \
                            "_" + str(sent_id) + "_" + str(cand.head_token.index)
                annot_candidates.append(CandidateGroupSet(cand_repr, [cand]))
                bar.update(m_id + 1)

            bar.finish()

        return annot_candidates

        # # TODO also return empty table if format of json files is wrong
        # return CandidateSet(OriginType.ANNOTATED,
        #                     self.params.coref_extraction_strategy)

    def extract_coref_candidates(self):

        doc_dict = {doc.id: doc for doc in self.docs}

        cand_set = CandidateSet(OriginType.EXTRACTED,
                                self.params.coref_extraction_strategy)

        if self.params.coref_extraction_strategy == CorefStrategy.NO_COREF:
            # NO coref chains needed
            return cand_set

        # collect mentions from different coref strategies
        coref_dict, coref_map_df_dict = self.collect_mentions()

        widgets = [
            progressbar.FormatLabel(NOTIFICATION_MESSAGES["progress"])
        ]
        bar = progressbar.ProgressBar(widgets=widgets,
                                      maxval=np.sum([len(coref) for coref in list(coref_dict.values())])).start()
        c_num = 0
        for doc_id, coref_chain in coref_dict.items():
            for coref in coref_chain:

                # for now skip corefs where not only pronouns are mentioned
                if all([m.mentionType == "PRONOMINAL" for m in coref.mentions]):
                    continue

                # TODO treat LISTs in future
                if any([m.mentionType == "LIST" for m in coref.mentions]) and self.params.ignore_lists:
                    continue

                if all([m.head_token.token.ner == "TITLE" or m.text in LocalDictLists.titles
                        for m in coref.mentions]):
                    continue

                cand_representative = ""
                local_cands_set = CandidateGroupSet(cand_representative)

                for m_num, m in enumerate(coref.mentions):

                    if coref_map_df_dict is None:
                        cand = Candidate.from_mention(mention=m,
                                          enhancement=self.params.phrase_extension,
                                          change_head=self.params.change_head)

                    elif doc_id not in list(coref_map_df_dict):
                        cand = Candidate.from_mention(mention=m,
                                          enhancement=self.params.phrase_extension,
                                          change_head=self.params.change_head)

                    else:
                        doc = doc_dict[coref_map_df_dict[doc_id].loc[m.sentenceIndex, ORIG_DOC]]
                        # TODO: Check if -1 is still correct: m.sentenceIndex - 1
                        sent_num = coref_map_df_dict[doc_id].loc[m.sentenceIndex, ORIG_SENT]
                        # TODO: Check if -1 is still correct: m.sentenceIndex - 1
                        tokens = [doc.sentences[sent_num].tokens[t.index] for t in m.tokens]
                        head_token = doc.sentences[sent_num].tokens[m.headIndex]

                        cand = Candidate(text=m.text,
                                         sentence=doc.sentences[sent_num], tokens=tokens,
                                         is_representative=m.is_representative, head_token=head_token,
                                         cand_type=CandidateType.COREF, coref_subtype=m.mentionType,
                                         enhancement=self.params.phrase_extension,
                                         change_head=self.params.change_head)

                    cand_representative = m.text.replace(" ", "_") + "_" + str(doc_id) + "_" +  str(coref.id) \
                                                            if m.is_representative else cand_representative
                    local_cands_set.append(cand)

                local_cands_set.group_name = cand_representative

                cand_set.append(local_cands_set)
                bar.update(c_num + 1)
                c_num += 1

        bar.finish()

        return cand_set

    def collect_mentions(self):

        if self.params.coref_extraction_strategy == CorefStrategy.ONE_DOC:
            return {doc.id: doc.corefs for doc in self.docs}, None

        # coref on combined documents CorefStrategy.MultiDoc, i.e., fewer super-documents
        coref_map_df_dict = {}
        coref_dict = {}

        # !!! put 500000 for timeout time on corenlp
        corenlp = CoreNLPCaller()
        doc_groups = len(self.docs) // self.params.max_doc_num if len(self.docs) % self.params.max_doc_num == 0 \
                                                    else len(self.docs) // self.params.max_doc_num + 1
        widgets = [
            progressbar.FormatLabel(NOTIFICATION_MESSAGES["mention_progress"])
        ]
        bar = progressbar.ProgressBar(widgets=widgets,
                                      maxval=doc_groups).start()
        for doc_group in range(doc_groups):
            mapping_df = pd.DataFrame(columns=[ORIG_SENT, ORIG_DOC])
            overall_sent_ind = 0
            text = ""

            for doc in self.docs[doc_group * self.params.max_doc_num : min(len(self.docs), (doc_group + 1) * self.params.max_doc_num)]:
                num_tokens = len(list(doc.all_tokens()))
                text += doc.fulltext + " \n"
                mapping_df = mapping_df.append(pd.DataFrame({
                    ORIG_SENT: list(range(num_tokens)),
                    ORIG_DOC: [doc.id] * num_tokens
                }, index=[sent for sent in range(overall_sent_ind, overall_sent_ind + num_tokens)]))
                overall_sent_ind += num_tokens

            output = corenlp.execute(text, annotators='tokenize, ssplit, pos, depparse, parse, ner, coref',
                                     coref="neural")

            if len(output.sentences) != overall_sent_ind:
                self._logger.warning(NOTIFICATION_MESSAGES["number_dismatch"])
                coref_dict.update({doc.id: doc.corefs for doc in self.docs[doc_group * self.params.max_doc_num : min(len(self.docs),
                                                                                     (doc_group + 1) * self.params.max_doc_num)]})
            else:
                coref_dict["doc_group_" + str(doc_group)] = output.corefs
                coref_map_df_dict["doc_group_" + str(doc_group)] = mapping_df
            bar.update(doc_group)
        bar.finish()

        return coref_dict, coref_map_df_dict

    def extract_NP_candidates(self, coref_df):
        # add NPs to candidates

        coref_np_df = copy.deepcopy(coref_df)
        cand_dict = {}

        widgets = [
            progressbar.FormatLabel(NOTIFICATION_MESSAGES["np_progress"])
        ]
        bar = progressbar.ProgressBar(widgets=widgets,
                                      maxval=np.sum([len(doc.sentences) for doc in self.docs])).start()
        sent_num = 0
        for doc_index, doc in enumerate(self.docs):
            for sent_index, tree in enumerate(doc.all_parse_trees()):

                np_subtrees = [t for t in tree.subtrees(filter=lambda t: t.label().value.startswith("NP"))]

                for j, np_cand in enumerate(np_subtrees):

                    if len(np_cand.leaves()) >= self.params.max_phrase_len:
                        continue

                    if all([l.token.pos in ["DT", "EX", "CD", "PRP", "PRP$", "RB"] for l in np_cand.leaves()]):
                        continue

                    sentence = doc.sentences[sent_index]
                    cand_tokens = [l.token for l in np_cand.leaves()]
                    dep_subtree = Candidate.find_dep_subset_static(cand_tokens, sentence)
                    cand_text_orig = "".join([t.token.before + t.token.word if i > 0 else t.token.word
                                         for i, t in enumerate(cand_tokens)])

                    and_case_index = -1
                    try:
                        if any([j.label().value == "CC" for j in np_cand]):
                            and_case_index = [dep.dependent for dep in dep_subtree if dep.dep in ["cc", "conj"]][0]
                                              # and dep.governor_gloss == head_token.word][0]
                    except IndexError:
                        pass

                    two_parts, firt_part_tokens, second_part_tokens = False, [], []
                    if and_case_index > -1:
                        if len([t for t in np_cand.subtrees(filter=lambda t: t.label().value.startswith("NP"))]) > 1:
                            continue

                        firt_part_tokens = [t for t in cand_tokens if t.index < and_case_index]
                        second_part_tokens = [t for t in cand_tokens if t.index > and_case_index]
                        first_part = "".join([t.token.before + t.token.word if i > 0 else t.token.word
                                                                for i, t in enumerate(firt_part_tokens)])

                        if any([v == first_part or v == first_part[0].lower() + first_part[1:]
                                for v in coref_np_df.loc[:, "phrase_extract"].values]):
                            first_part_head_token = Candidate.find_head_static(firt_part_tokens,
                                                           Candidate.find_dep_subset_static(firt_part_tokens, sentence))
                            two_parts = True if first_part_head_token.token.pos[:2] == "NN" else False

                    cand_token_sets = [cand_tokens] if not two_parts else [firt_part_tokens, second_part_tokens]

                    for token_set in cand_token_sets:

                        cand_text = "".join([t.token.before + t.token.word if i > 0 else t.token.word
                                             for i, t in enumerate(token_set)])
                        dep_subtree = Candidate.find_dep_subset_static(token_set, sentence)
                        head_token = Candidate.find_head_static(token_set, dep_subtree)

                        cand_df = create_cand_short_table([cand_text, doc.id, sentence.index, token_set[0].index,
                                                           token_set[-1].index, head_token.index], EXTRACT)
                        result = pd.merge(coref_np_df, cand_df, how='inner', on=[DOC_ID + EXTRACT, SENT_ID + EXTRACT,
                                                                              HEAD_ID + EXTRACT])
                        # if such an NP exists
                        if len(result):
                            continue

                        cand = Candidate(text=cand_text,
                                         sentence=sentence, tokens=token_set,
                                         is_representative=True, head_token=head_token,
                                         cand_type=CandidateType.NP, coref_subtype="NP",
                                         enhancement=ExtentedPhrases.ASIS,
                                         change_head=self.params.change_head)

                        cand.parent_tree = np_cand.parent()
                        cand_repr = "_".join([t.word for t in cand.tokens[:10]]) + "_" + str(doc.id) + \
                                    "_" + str(sent_index) + "_" + str(cand.head_token.index)
                        cand_dict[cand_repr] = CandidateGroupSet(cand_repr, [cand])
                        coref_np_df = coref_np_df.append(create_cand_short_table(cand, EXTRACT))

                bar.update(sent_num + 1)
                sent_num += 1

        bar.finish()

        return list(cand_dict.values())

    def extract_VP_candidates(self, coref_df):

        widgets = [
            progressbar.FormatLabel(NOTIFICATION_MESSAGES["vp_progress"])
        ]
        bar = progressbar.ProgressBar(widgets=widgets,
                                      maxval=np.sum([len(doc.sentences) for doc in self.docs])).start()
        sent_num = 0
        cand_index_dict = []
        cand_dict = {}

        for doc_index, doc in enumerate(self.docs):
            for sent_index, tree in enumerate(doc.all_parse_trees()):

                vp_subtrees = [t for t in tree.subtrees(filter=lambda t: t.label().value.startswith("VP"))]

                for j, vp_cand in enumerate(vp_subtrees):
                    sentence = doc.sentences[sent_index]
                    cand_tokens = [l.token for l in vp_cand.leaves()]
                    dep_subtree = Candidate.find_dep_subset_static(cand_tokens, sentence)
                    head_token = Candidate.find_head_static(cand_tokens, dep_subtree)

                    if "V" not in head_token.pos:
                        continue

                    depend_direct = [dep for dep in dep_subtree
                              if dep.governor == head_token.index and dep.dep not in ["mark", "xcomp", "advcl", "parataxis"]]
                    depend_set = [dep.token.index for dep in depend_direct]
                    depend_depend = [dep for dep in dep_subtree
                              if dep.governor in depend_set and dep.dep not in ["mark", "xcomp", "advcl", "parataxis", "acl:relcl"]]
                    depend_cand = sorted(depend_direct + depend_depend, reverse=False, key=lambda x: x.dependent)
                    cand_tokens_short = sorted([d.token for d in depend_cand] + [head_token], reverse=False,
                                               key=lambda x: x.index)

                    if len(cand_tokens_short) >= self.params.max_phrase_len:
                        continue

                    dep_subtree = Candidate.find_dep_subset_static(cand_tokens_short, sentence)
                    if not len(dep_subtree) and len(cand_tokens_short) > 1:
                        cand_tokens_short = [head_token]

                    if "{}_{}_{}".format(str(doc_index), str(sent_index), str(head_token.index)) in cand_index_dict:
                        continue

                    cand_text = " ".join([t.word for t in cand_tokens_short])

                    cand = Candidate(text=cand_text,
                                     sentence=sentence, tokens=cand_tokens_short,
                                     is_representative=True, head_token=head_token,
                                     cand_type=CandidateType.VP, coref_subtype="VP",
                                     enhancement=ExtentedPhrases.ASIS,
                                     change_head=ChangeHead.ORIG)

                    cand.parent_tree = vp_cand
                    cand_repr = "_".join([t.word for t in cand.tokens[:10]]) + "_" + str(doc.id) + \
                                "_" + str(sent_index) + "_" + str(cand.head_token.index)
                    cand_dict[cand_repr] = CandidateGroupSet(cand_repr, [cand])

                    cand_index_dict.append("{}_{}_{}".format(str(doc_index), str(sent_index), str(head_token.index)))

                bar.update(sent_num + 1)
                sent_num += 1

        bar.finish()
        return list(cand_dict.values())
