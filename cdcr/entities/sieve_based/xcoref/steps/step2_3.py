from cdcr.entities.sieve_based.step import Sieve
from cdcr.entities.const_dict_global import *
from cdcr.entities.sieve_based.xcoref.entity_xcoref import EntityXCoref

import numpy as np
import string
import re
import pandas as pd
import json
import requests
import wikipedia as wiki
import math
import copy
import time
from sklearn.metrics.pairwise import cosine_similarity as cs


MESSAGES = {
    "df": "Initializing a datatable with entities",
    "df_create": "Adding %(value)d-th (%(percentage)d %%) entity to the datatable (in: %(elapsed)s).",
    "df_res": "Updating results of {0} merge to a datatable.",
    "wiki": "Checking for missing matching wikipages.",
    "move_points": "Moving alien points in {0}.",
    "word_dict": "Extracting phrases for analysis.",
    "ne_chains": "Creating NE-chains and a comparison grid.",
    "sim_df": "Calculating similarity matrices.",
    "core": "Building cores of the clusters.",
    "merge_clusters": "Merging {0}.",
    "body": "Assembling body points of the clusters.",
    "border": "Adding border points.",
    "non-core": "Building non-core clusters.",
    "form_entities": "Forming entities.",
    "no_non_ne": "No suitable non-NE entities found. The step is skipped."
}
PRIORITY = "Priority"
CORE = "core_"


class XCorefStep3(Sieve):
    """
    A step merges person-nns, group, and person-nes entities using similarity of thier modifiers.
    """

    def __init__(self, step, docs, entity_dict, ent_preprocessor, model):

        super().__init__(XCOREF_3, step, docs.configuration.entity_identifier_config, entity_dict, ent_preprocessor,
                         model)

        self.non_ne_entities = {}
        self.phrases_non_ne_entities = {}
        self.entity_types = []

        self.ne_df = None
        self.sim_df = None
        self.sim_head_df = None
        self.sim_core_df = None

        self.head_phrase_dict = {}
        self.ne_phrase_dict = {}
        self.sim_dict = {}
        self.big_to_small_dict = {}

    def merge(self) -> dict:
        self.preprocess_entities()
        if not len(self.non_ne_entities):
            self.logger.info(MESSAGES["no_non_ne"])
            return self.entity_dict

        # ---CORE IDENTIFICATION---#
        core_clusters_init = self.form_cores()
        core_clusters_merged = self.merge_clusters(core_clusters_init, True)
        core_clusters_no_aliens = self.move_alien_points(core_clusters_merged, True)

        # ---ADD BODY POINTS---#
        clusters_body_init = self.add_body_points(core_clusters_no_aliens)
        clusters_body_merged = self.merge_clusters(clusters_body_init)
        clusters_body_no_aliens = self.move_alien_points(clusters_body_merged)

        # ---ADD BORDER POINTS---#
        clusters_w_border_points_init, points_to_add, not_merged = self.add_border_points(clusters_body_no_aliens)

        # ---ADD NON-CORE-BASED CLUSTERS---#
        clusters_w_non_core_cl_init, unmatched_entities = self.add_non_core_based_clusters(clusters_w_border_points_init,
                                                                                           not_merged, points_to_add)
        clusters_w_non_core_cl_no_aliens = self.move_alien_points(clusters_w_non_core_cl_init)

        final_clusters = sorted(clusters_w_non_core_cl_no_aliens, reverse=True, key=lambda x: len(x))
        self.create_entities(core_clusters_no_aliens, clusters_body_no_aliens, final_clusters, points_to_add,
                        unmatched_entities)
        entity_dict = {key: value for (key, value) in sorted(self.entity_dict.items(), reverse=True,
                                                                              key=lambda x: len(x[1].members))}
        self.entity_dict = self.merge_big_entities(entity_dict)

        entity_types_sim = list(self.table[self.table == 3].stack().reset_index()["level_0"].values)
        group_ne_entities = {key: entity for key, entity in self.entity_dict.items()
                          if entity.type in entity_types_sim}
        group_ne_dict = {}
        for key, entity in group_ne_entities.items():
            group_ne_dict[key] = set([k for k,v in entity.headwords_cand_tree.items() if len(v) > 1 and k[0].isupper()])

        entity_types = set(self.table[self.table == 1].stack().reset_index()["level_0"].values)
        entity_types = entity_types.union(set(self.table[self.table == 2].stack().reset_index()["level_0"].values))
        group_entities = {key: entity for key, entity in self.entity_dict.items()
                          if entity.type in entity_types}
        to_remove_from_queue = []
        for entity_key, gr_entity in group_entities.items():
            comp_vector = self.model.query(list(set([v for v in list(gr_entity.compound_dict)
                         if v[0].isupper()]).union(set(v for v in list(gr_entity.adjective_dict) if v[0].isupper()))))
            for entity_ne_key, ne_heads in group_ne_dict.items():
                head_vector = self.model.query(list(ne_heads))
                sim = cs(comp_vector, head_vector)
                if len(set(gr_entity.headwords_cand_tree).intersection(ne_heads)) or sim >= self.params.group_ne_sim:
                    try:
                        gr_entity.absorb_entity(self.entity_dict[entity_ne_key], self.step_name + XCOREF_3_SUB_1, 1.0)
                        to_remove_from_queue.append(entity_ne_key)
                        break
                    except KeyError:
                        continue
            self.update_entity_queue(gr_entity, to_remove_from_queue, self.step_name, False)

        return self.entity_dict

    def preprocess_entities(self):
        entity_types = list(self.table[self.table == 1].stack().reset_index()["level_0"].values)
        entity_types_core = list(self.table[self.table == 2].stack().reset_index()["level_0"].values)

        self.entity_types = entity_types + entity_types_core

        non_ne_entities_init = {key: entity for key, entity in self.entity_dict.items()
                           if entity.type in self.entity_types}

        for key, entity in non_ne_entities_init.items():
            if len(entity.members) > 1:
                for m in entity.members:
                    comp_ph_arr = []
                    for dep in m.dependency_subtree:
                        if dep.dep in [COMPOUND, NMOD_POSS] and dep.governor_gloss == m.head_token.word:
                            comp_ph_arr.append(dep.dependent)
                    comp_ph_arr.append(m.head_token.index)
                    comp_ph = " ".join([t.word for ind in sorted(comp_ph_arr) for t in m.tokens if ind == t.index])
                    new_entity = EntityXCoref(entity.document_set, self.ent_preprocessor, [m], None, entity.wiki_page,
                                              [comp_ph])
                    self.entity_dict[new_entity.name] = new_entity
                self.entity_dict.pop(key)

        self.non_ne_entities = {key: entity for key, entity in self.entity_dict.items()
                                                                if entity.type in self.entity_types}

        # ---WORD DICTIONARIES---#
        self.logger.info(MESSAGES["word_dict"])
        head_set = set()
        ne_words = set()
        nes_no_intersection = set()

        for key, entity in self.non_ne_entities.items():

            token_dict = {}
            for cand in entity.members:
                for token in cand.tokens:
                    token_dict[token.word] = token

            nmod_phrases = list(entity.nmod_phrases)
            headwords = [frozenset([word]) for word in list(entity.headwords_cand_tree.keys())]
            phrases = list(entity.adjective_phrases.keys()) + \
                      list(entity.compound_phrases.keys()) + headwords

            if len(set().union(*phrases)) == len(headwords) and entity.type != GROUP_TYPE:
                adapted_words = self.model.optimize_phrase(Sieve.words_to_tokens(set().union(*headwords).union(*nmod_phrases),
                                                                                 token_dict), True)
                adapted_words = self.model.optimize_phrase(Sieve.words_to_tokens(adapted_words, token_dict), True)

            elif len(set().union(*phrases).union(*nmod_phrases)) <= self.params.max_repr_words and \
                    len(set().union(*phrases)) == len(headwords):
                adapted_words = self.model.optimize_phrase(Sieve.words_to_tokens(set().union(*headwords).union(*nmod_phrases),
                                                                                 token_dict), True)

            else:
                adapted_words = self.model.optimize_phrase(Sieve.words_to_tokens(set().union(*phrases), token_dict), True)

            if self.params.max_repr_words >= len(adapted_words) > 0:
                phrase = [adapted_words]
            else:
                phrase = [self.model.optimize_phrase(Sieve.words_to_tokens([list(entity.headwords_cand_tree.keys())[0]],
                                                                           token_dict), True)]

            if entity.type == PERSON_NES_TYPE:
                nes_no_intersection.add(list(entity.headwords_cand_tree.keys())[0])
                phrase = [[list(entity.headwords_cand_tree.keys())[0]]]

            phrase_key = frozenset(set().union(*phrase))

            head_words = self.model.optimize_phrase(Sieve.words_to_tokens(list(entity.headwords_cand_tree.keys()), token_dict)
                                                    , True)
            if len(head_words) == 0:
                continue

            if phrase_key not in self.phrases_non_ne_entities:
                self.phrases_non_ne_entities[phrase_key] = []

            self.head_phrase_dict[phrase_key] = head_words[0]
            self.phrases_non_ne_entities[phrase_key].append(key)

            ne_words_local = np.array(list(phrase[0]))[np.where(np.array([
                w.replace("_", " ") in self.ent_preprocessor.phrase_ner_dict for w in phrase[0]]) > 0)].tolist()

            if len(ne_words_local) > 0:
                ne_word = head_words[0] if head_words[0] in ne_words_local else ne_words_local[-1]
                ne_words = ne_words.union({ne_word} if ne_word.istitle() or ne_word.isupper() else set())
                self.ne_phrase_dict[phrase_key] = ne_word if ne_word.istitle() or ne_word.isupper() else None
            else:
                self.ne_phrase_dict[phrase_key] = None

            head_set = head_set.union(
                self.model.optimize_phrase(Sieve.words_to_tokens(list(entity.headwords_cand_tree.keys()), token_dict), True))

        self.phrases_non_ne_entities = {key: val for key, val in sorted(self.phrases_non_ne_entities.items(),
                                                                   reverse=True, key=lambda x: len(x[1])) if
                                   len(key) > 0}

        short_phrases = list(filter(lambda x: len(x) == 1, list(self.phrases_non_ne_entities.keys())))

        head_set = set(head_set).union(set().union(*short_phrases))
        for phr in short_phrases:
            self.head_phrase_dict[phr] = list(phr)[0]

        long_phrases = list(filter(lambda x: len(x) > 1, list(self.phrases_non_ne_entities.keys())))
        for s_phr in short_phrases:
            phr_cand = []
            for l_phr in long_phrases:
                # if s_phr.issubset(l_phr):
                short_phrase_entity = self.non_ne_entities[self.phrases_non_ne_entities[s_phr][0]]
                if s_phr.issubset(l_phr) \
                        and short_phrase_entity.members[0].head_token.word.lower() \
                        in self.non_ne_entities[self.phrases_non_ne_entities[l_phr][0]].headwords_cand_tree:
                    phr_cand.append(l_phr)
                    self.big_to_small_dict[l_phr] = s_phr
                    break

        phrases = [" ".join(list(phrase)) for phrase in list(self.phrases_non_ne_entities.keys())
                   if phrase not in list(self.big_to_small_dict.values())]

        phrases_core = [phr for phr in phrases
                        if self.non_ne_entities[self.phrases_non_ne_entities[frozenset(phr.split(" "))][0]].type
                        in entity_types_core]



        # ---NE CHAINS CONSTRUCTION---#
        self.logger.info(MESSAGES["ne_chains"])
        ne_words_lc = {w.lower().replace(" ", "_") if w[-1] not in string.punctuation
                       else w.lower().replace(" ", "_")[:-1]: w for w in ne_words}

        country_ne_words_lc = {k: v for k, v in ne_words_lc.items()
                               if self.ent_preprocessor.phrase_ner_dict[v.replace("_", " ")] in [COUNTRY_NER,
                                                                                                       NATIONALITY_NER]}
        org_ne_words_lc = {k: v for k, v in ne_words_lc.items()
                           if self.ent_preprocessor.phrase_ner_dict[v.replace("_", " ")] not in [COUNTRY_NER,
                                                                                                       NATIONALITY_NER]}
        ne_df_init_country = pd.DataFrame(np.zeros((len(country_ne_words_lc), len(country_ne_words_lc))),
                                          index=list(country_ne_words_lc.values()),
                                          columns=list(country_ne_words_lc.values()))
        ne_df_init_org = pd.DataFrame(np.zeros((len(org_ne_words_lc), len(org_ne_words_lc))),
                                      index=list(org_ne_words_lc.values()),
                                      columns=list(org_ne_words_lc.values()))

        antonyms = {}
        ne_df_init_country, antonyms = self.fill_ne_df(ne_df_init_country, country_ne_words_lc, ne_words_lc, antonyms,
                                             nes_no_intersection)
        ne_clusters_country = self.form_ne_clusters(ne_df_init_country)

        ne_df_init_org, antonyms = self.fill_ne_df(ne_df_init_org, org_ne_words_lc, ne_words_lc, antonyms, nes_no_intersection, True)
        ne_clusters_org = self.form_ne_clusters(ne_df_init_org)

        # TODO make use of antonyms, e.g. "American" <> "foreign"
        for ant, vals in antonyms.items():
            if ant in vals:
                vals.remove(ant)

        def __filter_ne(ne_set):
            return len(ne_set) > 1 or len(ne_set.intersection(nes_no_intersection)) \
                   or list(ne_set)[0] in self.ent_preprocessor.ner_dict[NATIONALITY_NER] or \
                   list(ne_set)[0] in self.ent_preprocessor.ner_counted_dict[COUNTRY_NER]

        ne_clusters_org_fil = list(filter(__filter_ne, ne_clusters_org))
        ne_clusters_country_fil = list(filter(__filter_ne, ne_clusters_country))

        ne_all_org_country = set().union(*ne_clusters_org_fil).union(*ne_clusters_country_fil)
        ne_df_init = pd.concat([ne_df_init_org, ne_df_init_country], axis=1)
        self.ne_df = ne_df_init.loc[list(ne_all_org_country), list(ne_all_org_country)]
        self.ne_df.fillna(0, inplace=True)
        for cl in ne_clusters_org_fil + ne_clusters_country_fil:
            self.ne_df.loc[list(cl), list(cl)] = self.ne_df.loc[list(cl), list(cl)].replace(0, self.params.ne_word_weight)
        non_relevant_ne = ne_words - set(self.ne_df.index)
        for k, v in self.ne_phrase_dict.items():
            if v in non_relevant_ne:
                self.ne_phrase_dict[k] = None



        # ---CALCULATION OF SIMILARITY MATRICES---#
        self.logger.info(MESSAGES["sim_df"])
        change_we_coeff = any([p in self.ent_preprocessor.phrase_ner_dict for p in phrases_core])

        def __tranform_we(phrase, coef=self.params.ne_word_weight):
            # weight vectors of NE words more than non-NE words
            # phrase_mod = [p for p in phrase.split(" ") if p in self.model]
            phrase_mod = phrase.split(" ")
            if len(phrase_mod) == 0:
                # return np.zeros((1, self._model.vector_size))
                # phrase_mod = [p_1 for p in phrase.split(" ") for p_1 in p.split("-") if p_1 in self.model]
                phrase_mod = [p_1 for p in phrase.split(" ") for p_1 in p.split("-")]
            return self.model.query(phrase_mod, [coef if p in self.ent_preprocessor.phrase_ner_dict else 1 for p in phrase_mod])
            # return [np.sum([coef * self.model[p]
            #                 if p in self.ent_preprocessor.phrase_ner_dict
            #                 else self.model[p] for p in phrase_mod], axis=0)]

        # calculation of the similarity matrix and cluster center candidate matrix
        self.sim_df = pd.DataFrame(np.zeros((len(phrases), len(phrases))), index=phrases, columns=phrases)
        self.sim_core_df = pd.DataFrame(np.zeros((len(phrases_core), len(phrases_core))),
                                   index=[CORE + p for p in phrases_core], columns=phrases_core)
        for i in range(len(phrases)):
            phrase_1 = phrases[i]
            for j in range(i + 1, len(phrases)):
                phrase_2 = phrases[j]
                if change_we_coeff:
                    if self.ne_phrase_dict[frozenset(phrase_1.split(" "))] is not None \
                            and self.ne_phrase_dict[frozenset(phrase_2.split(" "))] is not None:
                        # use word vector coefs from the ne-chain based table
                        coef = self.ne_df.loc[self.ne_phrase_dict[frozenset(phrase_1.split(" "))], self.ne_phrase_dict[
                            frozenset(phrase_2.split(" "))]]
                        if not coef:
                            continue
                        sim = cs(__tranform_we(phrase_1, coef), __tranform_we(phrase_2, coef))
                    else:
                        # use default settings: default coefs for ne-containing phrases
                        sim = cs(__tranform_we(phrase_1), __tranform_we(phrase_2))
                else:
                    sim = self.model.n_similarity(phrase_1.split(" "), phrase_2.split(" "))
                if sim >= self.params.min_sim:
                    self.sim_df.loc[phrase_1, phrase_2] = sim
                    self.sim_df.loc[phrase_2, phrase_1] = sim
                    if sim >= self.params.min_core_sim and phrase_1 in phrases_core and phrase_2 in phrases_core:
                        self.sim_core_df.loc[CORE + phrase_1, phrase_2] = 1
                        self.sim_core_df.loc[CORE + phrase_2, phrase_1] = 1

        for phrase_1 in list(self.sim_df.index):
            self.sim_dict[phrase_1] = []
            for phrase_2 in list(self.sim_df.index):
                if phrase_1 == phrase_2:
                    continue
                if self.sim_df.loc[phrase_1, phrase_2] > 0:
                    self.sim_dict[phrase_1].append((phrase_2, self.sim_df.loc[phrase_1, phrase_2]))

        self.sim_head_df = pd.DataFrame(np.zeros((len(head_set), len(head_set))), index=head_set, columns=head_set)
        head_set = list(head_set)
        for i in range(len(head_set)):
            phrase_1 = head_set[i]
            self.sim_head_df.loc[phrase_1, phrase_1] = 0.5
            for j in range(i + 1, len(head_set)):
                phrase_2 = head_set[j]
                try:
                    coef = self.ne_df.loc[phrase_1, phrase_2]
                    if not coef:
                        continue
                    sim = cs(__tranform_we(phrase_1, 1.0), __tranform_we(phrase_2, 1.0))
                except KeyError:
                    # sim = self._model.n_similarity([phrase_1], [phrase_2])
                    sim = cs(__tranform_we(phrase_1), __tranform_we(phrase_2))
                self.sim_head_df.loc[phrase_1, phrase_2] = sim
                self.sim_head_df.loc[phrase_2, phrase_1] = sim

    def fill_ne_df(self, df, search_phrases, ne_words_lc, antonyms, nes_no_intersection, sub_phrase=False):
        for word_1 in list(search_phrases.keys()):
            # print(word_1)
            syns = set()
            try_request = True
            while try_request:
                request_url = 'http://api.conceptnet.io/c/en/' + word_1 + '?limit=1000'
                request_result = requests.get(request_url)
                if request_result.status_code == 200:
                    try_request = False
                else:
                    time_wait = 10.0
                    time.sleep(time_wait)
                    self.logger.warning("No results obtained for {0}. Will repeat the attempt in {1} "
                                     "seconds.".format(request_url, time_wait))

            for edge in json.loads(request_result.text)["edges"]:
                if ",/c/en/" + word_1 in edge["@id"]:
                    if "language" in edge["end"]:
                        if edge["end"]["language"] == "en":
                            if edge["rel"]["@id"] not in ["/r/SimilarTo"]:
                                syns.add(edge["end"]["label"].title())
                            if edge["rel"]["@id"] in ["/r/Antonym"]:
                                if word_1 not in antonyms:
                                    antonyms[search_phrases[word_1]] = set()
                                antonyms[search_phrases[word_1]].add(edge["start"]["label"].title())
                                antonyms[search_phrases[word_1]].add(edge["end"]["label"].title())

            for word_2 in list(search_phrases.keys()):
                if word_1 == word_2:
                    df.loc[ne_words_lc[word_1], ne_words_lc[word_2]] = self.params.ne_word_weight
                    continue

                if ne_words_lc[word_2].replace("_", " ") in syns and \
                        (word_1 not in nes_no_intersection and word_2 not in nes_no_intersection):
                    df.loc[ne_words_lc[word_1], ne_words_lc[word_2]] = self.params.ne_word_weight
                    df.loc[ne_words_lc[word_2], ne_words_lc[word_1]] = self.params.ne_word_weight

                elif len(word_2) > 7 and len(word_1) > 7:
                    if word_2[:8] in word_1:
                        df.loc[ne_words_lc[word_1], ne_words_lc[word_2]] = self.params.ne_word_weight
                        df.loc[ne_words_lc[word_2], ne_words_lc[word_1]] = self.params.ne_word_weight

                elif sub_phrase and any([ne_words_lc[word_2].replace("_", " ") in s for s in syns]):
                    df.loc[ne_words_lc[word_1], ne_words_lc[word_2]] = self.params.ne_word_weight
                    df.loc[ne_words_lc[word_2], ne_words_lc[word_1]] = self.params.ne_word_weight

                elif ne_words_lc[word_2].isupper():
                    wiki_page = None
                    if word_2 in self.ent_preprocessor.phrase_wiki_dict:
                        wiki_page = self.ent_preprocessor.phrase_wiki_dict[word_2]["page"]
                    else:
                        try:
                            wiki_page = wiki.page(word_2)
                        except (wiki.DisambiguationError, wiki.PageError, requests.exceptions.SSLError):
                            pass
                    if wiki_page is not None:
                        if len(wiki_page.original_title) > 7:
                            if wiki_page.original_title[:8] in ne_words_lc[word_1]:
                                # print(wiki_page.original_title[:8], ne_words_lc[word_1])
                                df.loc[ne_words_lc[word_1], ne_words_lc[word_2]] = self.params.ne_word_weight
                                df.loc[ne_words_lc[word_2], ne_words_lc[word_1]] = self.params.ne_word_weight
        return df, antonyms

    def form_ne_clusters(self, df):
        ne_clusters = []
        ne_all_merged = set()
        for ne in list(df.index):
            if ne in ne_all_merged:
                continue
            ne_all_merged.add(ne)
            ne_cl_old = set()
            ne_cl_new = set(df[df[ne] > 0].index)
            while len(ne_cl_old) != len(ne_cl_new):
                ne_cl_old = ne_cl_new.copy()
                for in_ne in ne_cl_old:
                    ne_cl_new = ne_cl_new.union(set(df[df[in_ne] > 0].index))
                    ne_all_merged.add(in_ne)
            ne_clusters.append(ne_cl_new)
        return ne_clusters

    def form_cores(self):
        self.logger.info(MESSAGES["core"])
        core_overlaps_df = pd.DataFrame(np.zeros((len(self.sim_core_df), len(self.sim_core_df))),
                                        index=list(self.sim_core_df.index), columns=list(self.sim_core_df.index))
        overlap_ratio_threshold = min(round(math.log(len(self.phrases_non_ne_entities), 5000), 1),
                                      self.params.overlap_ratio_max)
        overlap_ratio_threshold = max(overlap_ratio_threshold, self.params.overlap_ratio_min)

        for i in range(len(self.sim_core_df.index) - 1):
            index_1 = list(self.sim_core_df.index)[i]
            for j in range(i + 1, len(self.sim_core_df.index)):
                index_2 = list(self.sim_core_df.index)[j]
                match = np.sum(np.logical_and(self.sim_core_df.loc[index_1].values, self.sim_core_df.loc[index_2].values))
                divider = max(np.sum(self.sim_core_df.loc[index_1].values), np.sum(self.sim_core_df.loc[index_2].values))
                ratio = match / divider if divider > 0 else 0
                if ratio >= overlap_ratio_threshold:
                    if self.ne_phrase_dict[frozenset(index_1[5:].split(" "))] is not None and \
                            self.ne_phrase_dict[frozenset(index_2[5:].split(" "))] is not None:
                        if self.ne_df.loc[self.ne_phrase_dict[frozenset(index_1[5:].split(" "))],
                                     self.ne_phrase_dict[frozenset(index_2[5:].split(" "))]] > 0:
                            core_overlaps_df.loc[index_1, index_2] = ratio
                            core_overlaps_df.loc[index_2, index_1] = ratio
                    else:
                        core_overlaps_df.loc[index_1, index_2] = ratio
                        core_overlaps_df.loc[index_2, index_1] = ratio

        core_candidates = {}
        for ind in list(self.sim_core_df.index):
            winners = list(np.argwhere(core_overlaps_df[ind].values > 0).flatten().tolist())
            if len(winners) > 0:
                core_candidates[ind] = [(list(self.sim_core_df.index)[w], val)
                                        for w, val in
                                        zip(winners, core_overlaps_df[ind].values[core_overlaps_df[ind].values > 0])]

        # formation of the cluster cores
        core_candidates = {key: val for (key, val) in sorted(core_candidates.items(), reverse=True,
                                                             key=lambda x: len(x[1]))}

        core_clusters_init = []
        used_cores = []
        for core, sim_cores in core_candidates.items():
            if core in used_cores:
                continue
            used_cores.append(core)
            core_cluster, used_cores = self.check_core(core, used_cores, core_candidates)
            core_cluster.append(core)
            if len(core_cluster) > 1:
                core_clusters_init.append(set(core_cluster))
        return sorted(core_clusters_init, reverse=True, key=lambda x: len(x))

    def check_core(self, core, all_cores, core_candidates):
        sim_cores = []
        scores = np.array([v[1] for v in core_candidates[core]])
        # equavalent to argmax;  left here in case of experiments with merging top N most sim core points
        max_sim_core = scores.argsort()[-1:][::-1]
        for i, inner_core in enumerate([core_candidates[core][c][0] for c in max_sim_core]):
            if inner_core in all_cores:
                continue
            sim_cores.append(inner_core)
            all_cores.append(inner_core)
            new_sim_cores, all_cores = self.check_core(inner_core, all_cores, core_candidates)
            sim_cores.extend(new_sim_cores)
        return sim_cores, all_cores

    @staticmethod
    def remove_core_prefix(word, is_prefix=False):
        if is_prefix:
            return word[5:]
        return word

    def merge_clusters(self, cl_collection, core_phrase=False):

        loc_message = "cores" if core_phrase else "clusters"
        self.logger.info(MESSAGES["merge_clusters"].format(loc_message))

        def __no_prefix(word):
            return XCorefStep3.remove_core_prefix(word, core_phrase)

        def __max_set(data):
            """
            Calculated maxinal frequent itemsets.
            """
            def __itemset_comparison(old_set, new_set):
                if old_set.issubset(new_set):
                    return True, old_set, new_set
                if new_set.issubset(old_set):
                    return False, None, None
                else:
                    return False, None, new_set
            max_itemset_list = list()
            dict_string_to_set = dict()
            for itemset in data:
                split_itemset = set(itemset)
                dict_string_to_set[id(split_itemset)] = itemset
                if len(max_itemset_list) == 0:
                    max_itemset_list.append(split_itemset)
                    continue
                del_old, del_set, new_set = None, None, None
                for max_itemset in list(reversed(max_itemset_list)):
                    del_old, del_set, new_set = __itemset_comparison(max_itemset, split_itemset)
                    if del_old:
                        # found a bigger itemset, replace a small one
                        max_itemset_list.remove(del_set)
                        max_itemset_list.append(new_set)
                        break
                    elif new_set is None:
                        # a bigger set already exists
                        break
                if not del_old:
                    # a new itemset found
                    if new_set is not None:
                        max_itemset_list.append(new_set)
            return max_itemset_list

        if not len(cl_collection):
            return cl_collection

        # calculation of cross- and within-cluster similarity
        cl_collection = sorted(cl_collection, reverse=True, key=lambda x: len(x))
        cross_sim_arr = np.zeros((len(cl_collection), len(cl_collection)))
        for i in range(len(cl_collection)):
            for j in range(len(cl_collection)):
                norm_names_1 = [__no_prefix(v) for v in list(cl_collection[i])]
                norm_names_2 = [__no_prefix(v) for v in list(cl_collection[j])]
                df = self.sim_df.loc[norm_names_1, norm_names_2]
                heads_1 = list(set([self.head_phrase_dict[frozenset(n.split(' '))] for n in norm_names_1]))
                heads_2 = list(set([self.head_phrase_dict[frozenset(n.split(' '))] for n in norm_names_2]))
                df_heads = self.sim_head_df.loc[heads_1, heads_2]
                nes_1 = list(filter(lambda x: x is not None, list(set([self.ne_phrase_dict[frozenset(n.split(" "))]
                                                                       for n in norm_names_1]))))
                nes_2 = list(filter(lambda x: x is not None, list(set([self.ne_phrase_dict[frozenset(n.split(" "))]
                                                                       for n in norm_names_2]))))
                if len(nes_1) > 0 and len(nes_2) > 0 and np.sum(np.array(self.ne_df.loc[nes_1, nes_2].values == 0)) > 0:
                    cross_sim_arr[i][j] = 0
                else:
                    cross_sim_arr[i][j] = self.params.merge_phrases_weight * np.mean(df.values) \
                                          + (1 - self.params.merge_phrases_weight) * np.mean(df_heads.values)
        cluster_merge_mapping = {}
        for i in range(len(cross_sim_arr)):
            sim_list = np.array([v if v >= 0.2 else 0 for v in cross_sim_arr[i]])
            sim_list[i] = sim_list[i] - 0.05 * sim_list[i]
            val = list(sim_list.argsort()[-4:][::-1]) if np.sum(sim_list) > 0 else [i]
            try:
                val_ = [v for j, v in enumerate(val) if j <= val.index(i)]
            except ValueError:
                if len(val) == 4:
                    val[3] = i
                val_ = val
            cluster_merge_mapping[i] = val_
        cluster_merge_mapping = {key: val for key, val in
                                 sorted(cluster_merge_mapping.items(), reverse=False, key=lambda x: len(x[1]))}

        all_merge_maps = set()
        for outer_index, cl in cluster_merge_mapping.items():
            inters = set(cluster_merge_mapping[outer_index]).intersection(*[set(cluster_merge_mapping[inner_index])
                           for inner_index in cl if outer_index != inner_index]) if len(cl) > 1 else set(cl)
            if len(inters) > 0:
                all_merge_maps.add(frozenset(inters.union({outer_index})))

        all_merge_maps_list = list(__max_set(all_merge_maps))
        all_merge_maps_sim = []
        for merge_set in all_merge_maps_list:
            merge_list = list(merge_set)
            sim_val = []
            for v1_i, v1 in enumerate(merge_list):
                for v2_i in range(v1_i+1, len(merge_list)):
                    v2 = merge_list[v2_i]
                    sim_val.append(cross_sim_arr[v1][v2])
            all_merge_maps_sim.append(np.mean(sim_val) if len(sim_val) else cross_sim_arr[v1][v1])

        all_merge_maps_list = [k for k,v in sorted(zip(all_merge_maps_list, all_merge_maps_sim), reverse=True,
                                                   key=lambda x: x[1])]
        # all_merge_maps_sim = sorted(all_merge_maps_sim, reverse=True)
        to_merge = []
        merged_id = set()
        all_merge_maps_list_copy = copy.copy(all_merge_maps_list)
        for i in range(len(all_merge_maps_list_copy) - 1):
            if i in merged_id:
                continue
            final_cluster = all_merge_maps_list_copy[i]
            cl_1 = all_merge_maps_list_copy[i]
            for j in range(i + 1, len(all_merge_maps_list_copy)):
                if j in merged_id:
                    continue
                cl_2 = all_merge_maps_list_copy[j]
                if len(cl_1.intersection(cl_2)) > 0:
                    for inters in cl_1.intersection(cl_2):
                        if len(set(cluster_merge_mapping[inters]).intersection(cl_1)) == len(cl_1) or \
                            len(set(cluster_merge_mapping[inters]).intersection(cl_2)) == len(cl_2):
                            final_cluster = final_cluster.union(cl_2)
                            merged_id.add(j)
                        else:
                            all_merge_maps_list_copy[j] = cl_2 - cl_1
                #     # TODO check sim level
                #     # final_cluster = final_cluster.union(cl_2)
                #     all_merge_maps_list_copy[j] = cl_2 - cl_1
                #     # merged_id.add(j)
                #     merged_id.add(i)
                # if len(cl_1.intersection(cl_2)) == 1:
                #     pass
                # if len(cl_1.intersection(cl_2)) > 1:
                #     final_cluster = final_cluster.union(cl_2)
                #     merged_id.add(j)
            for j in range(i + 1, len(all_merge_maps_list_copy)):
                all_merge_maps_list_copy[j] = all_merge_maps_list_copy[j].difference(final_cluster)
            if len(final_cluster) > 1:
                to_merge.append(final_cluster)

        if all_merge_maps_list_copy[-1] not in to_merge:
            if len(all_merge_maps_list_copy[-1]) > 1:
                to_merge.append(all_merge_maps_list_copy[-1])

        to_remove = []
        cl_collection_modified = copy.deepcopy(cl_collection)
        for map in to_merge:
            map = tuple(map)
            # print("Merging ", map[1:], "into", map[0])
            for j in range(1, len(map)):
                cl_collection_modified[map[0]] = cl_collection_modified[map[0]].union(
                    cl_collection_modified[map[j]])
            to_remove.extend(list(map[1:]))
        for index in sorted(list(set(to_remove)), reverse=True, key=lambda x: x):
            del cl_collection_modified[index]
        return cl_collection_modified

    def move_alien_points(self, cl_collection, core_phrase=False):

        loc_message = "cores" if core_phrase else "clusters"
        self.logger.info(MESSAGES["move_points"].format(loc_message))

        def __no_prefix(word):
            return XCorefStep3.remove_core_prefix(word, core_phrase)

        if not len(cl_collection):
            return cl_collection

        # inner-cluster similarity
        mean_sim_arr = []
        cl_collection_modified = copy.deepcopy(cl_collection)
        phrase_coef = self.params.aliens_phrases_weight
        head_coef = 1 - self.params.aliens_phrases_weight

        for i in range(len(cl_collection)):
            norm_names = [__no_prefix(v) for v in list(cl_collection[i])]
            df = self.sim_df.loc[norm_names, norm_names]
            heads = list(set([self.head_phrase_dict[frozenset(n.split(" "))] for n in norm_names]))
            df_heads = self.sim_head_df.loc[heads, heads]
            ners = list(filter(lambda x: x is not None, list(
                set([self.ne_phrase_dict[frozenset(n.split(" "))] for n in norm_names]))))
            if len(ners) == 0:
                mean_sim_arr.append(phrase_coef * np.sum(df.values) / (len(df) * (len(df) - 1))
                                    + head_coef * np.mean(df_heads.values))
            else:
                mean_sim_arr.append(np.sum(df.values) / (len(df) * (len(df) - 1)))

        for i in range(len(cl_collection) - 1):
            if len(cl_collection[i]) < 2:
                continue
            for j in range(i + 1, len(cl_collection)):
                if len(cl_collection[j]) < 2:
                    continue
                move = []

                # similarity of a phrase from cluster 1
                for cl_i in cl_collection[i]:
                    sum_big = 0
                    sum_head_big = 0
                    sum_small = 0
                    sum_head_small = 0
                    # to its cluster members
                    for cl_i2 in cl_collection[i]:
                        if cl_i == cl_i2:
                            continue
                        sum_big += self.sim_df.loc[__no_prefix(cl_i), __no_prefix(cl_i2)]
                        sum_head_big += self.sim_head_df.loc[self.head_phrase_dict[frozenset(__no_prefix(cl_i).split(" "))],
                                                        self.head_phrase_dict[frozenset(__no_prefix(cl_i2).split(" "))]]

                    nes_1 = list(
                        filter(lambda x: x is not None, [self.ne_phrase_dict[frozenset(__no_prefix(cl_i).split(" "))]]))
                    nes_2 = list(filter(lambda x: x is not None, list(
                        set([self.ne_phrase_dict[frozenset(__no_prefix(n).split(" "))] for n in
                             list(cl_collection[j])]))))
                    if len(nes_1) > 0 and len(nes_2) > 0 \
                            and np.sum(np.array(self.ne_df.loc[nes_1, nes_2].values == 0)) > 0:
                        continue
                    # and to the other cluster
                    for cl_j in cl_collection[j]:
                        sum_small += self.sim_df.loc[__no_prefix(cl_i), __no_prefix(cl_j)]
                        sum_head_small += self.sim_head_df.loc[self.head_phrase_dict[frozenset(__no_prefix(cl_i).split(" "))],
                                                          self.head_phrase_dict[frozenset(__no_prefix(cl_j).split(" "))]]
                    if len(nes_1) == 0:
                        n_sum_big = (phrase_coef * sum_big + head_coef * sum_head_big) / (len(cl_collection[i]) - 1)
                        n_sum_small = (phrase_coef * sum_small + head_coef * sum_head_small) / len(
                            cl_collection[j])
                    else:
                        n_sum_big = sum_big / (len(cl_collection[i]) - 1)
                        n_sum_small = sum_small / len(cl_collection[j])
                    if n_sum_small - n_sum_big >= self.params.eps and n_sum_small >= mean_sim_arr[j]:
                        move.append((i, j, cl_i))
                        # print(i, j, cl_i, n_sum_big, n_sum_small)

                # similarity of a phrase from cluster 2 to its cluster members
                for cl_j in cl_collection[j]:
                    sum_big = 0
                    sum_head_big = 0
                    sum_small = 0
                    sum_head_small = 0
                    # to its cluster members
                    for cl_j2 in cl_collection[j]:
                        if cl_j == cl_j2:
                            continue
                        sum_small += self.sim_df.loc[__no_prefix(cl_j), __no_prefix(cl_j2)]
                        sum_head_small += self.sim_head_df.loc[self.head_phrase_dict[frozenset(__no_prefix(cl_j).split(" "))],
                                                          self.head_phrase_dict[
                                                              frozenset(__no_prefix(cl_j2).split(" "))]]

                    nes_1 = list(
                        filter(lambda x: x is not None, [self.ne_phrase_dict[frozenset(__no_prefix(cl_j).split(" "))]]))
                    nes_2 = list(filter(lambda x: x is not None, list(
                        set([self.ne_phrase_dict[frozenset(__no_prefix(n).split(" "))] for n in
                             list(cl_collection[i])]))))
                    if len(nes_1) > 0 and len(nes_2) > 0 \
                            and np.sum(np.array(self.ne_df.loc[nes_1, nes_2].values == 0)) > 0:
                        continue
                    # and to the other cluster
                    for cl_i in cl_collection[i]:
                        sum_big += self.sim_df.loc[__no_prefix(cl_j), __no_prefix(cl_i)]
                        sum_head_big += self.sim_head_df.loc[self.head_phrase_dict[frozenset(__no_prefix(cl_i).split(" "))],
                                                        self.head_phrase_dict[frozenset(__no_prefix(cl_j).split(" "))]]
                    if len(nes_1) == 0:
                        n_sum_big = (phrase_coef * sum_big + head_coef * sum_head_big) / len(cl_collection[i])
                        n_sum_small = (phrase_coef * sum_small + head_coef * sum_head_small) / (
                                len(cl_collection[j]) - 1)
                    else:
                        n_sum_big = sum_big / len(cl_collection[i])
                        n_sum_small = sum_small / (len(cl_collection[j]) - 1)

                    if n_sum_big - n_sum_small >= self.params.eps and n_sum_big >= mean_sim_arr[i]:
                        move.append((j, i, cl_j))
                        # print(j, i, cl_j, n_sum_small, n_sum_big)

                for (fr, to, what) in move:
                    try:
                        cl_collection_modified[fr].remove(what)
                        cl_collection_modified[to].add(what)
                    except KeyError:
                        continue
        return cl_collection_modified

    def majority_calc(self, cores):
        nouns = {}
        adjs = {}
        for core in cores:
            core_frozenset = frozenset(XCorefStep3.remove_core_prefix(core, True).split(" "))
            all_words = [s for s in re.split(r"(_| )", XCorefStep3.remove_core_prefix(core, True)) if
                         s not in [" ", "_"]]
            ent = self.non_ne_entities[self.phrases_non_ne_entities[core_frozenset][0]]
            head = ent.members[0].head_token.word
            nouns[head] = nouns.get(head, 0) + 1
            for adj in set(all_words) - set(head):
                adjs[adj] = adjs.get(adj, 0) + 1
        return nouns, adjs

    def add_body_points(self, cl_collection):
        self.logger.info(MESSAGES["body"])
        if not len(cl_collection):
            return cl_collection

        clusters_body_init = []
        for j in range(len(cl_collection)):
            u = set()
            for i, c in enumerate(cl_collection[j]):
                to_union = set([x[0] for x in list(filter(lambda x: x[1] >= self.params.min_cluster_sim,
                                                  sorted(self.sim_dict[c[5:]], reverse=True, key=lambda x: x[1])))])
                for phr in list(to_union).copy():
                    core_members = list(filter(lambda x: x is not None,
                                               [self.ne_phrase_dict[frozenset(c_[5:].split(" "))]
                                                 for c_ in cl_collection[j]]))
                    if self.ne_phrase_dict[frozenset(phr.split(" "))] is not None and len(core_members) > 0:
                        if any(self.ne_df.loc[[self.ne_phrase_dict[frozenset(phr.split(" "))]], core_members]):
                            to_union.remove(phr)
                u = u.union(to_union)
                u = u.union({c[5:]})
            clusters_body_init.append(u)

        # deciding on the cluster membership of the points in cluster intersection
        clusters_body_cleared = copy.deepcopy(clusters_body_init)
        for i in range(len(clusters_body_cleared) - 1):
            core_majority_nouns_big, core_majority_adj_big = self.majority_calc(cl_collection[i])
            for j in range(i + 1, len(clusters_body_cleared)):
                inters = clusters_body_cleared[i].intersection(clusters_body_cleared[j])
                core_majority_nouns_small, core_majority_adj_small = self.majority_calc(cl_collection[j])

                for inters_instance in inters:
                    if CORE + inters_instance in cl_collection[i]:
                        clusters_body_cleared[j].remove(inters_instance)
                        continue
                    if CORE + inters_instance in cl_collection[j]:
                        clusters_body_cleared[i].remove(inters_instance)
                        continue
                    core_set = frozenset(inters_instance.split(" "))
                    ent = self.non_ne_entities[self.phrases_non_ne_entities[core_set][0]]
                    head = ent.members[0].head_token.word
                    adjs = set([s for s in re.split(r"(_| )", inters_instance) if s not in [" ", "_"]])
                    try:
                        adjs.remove(head)
                    except KeyError:
                        try:
                            adjs.remove(head.lower())
                        except KeyError:
                            pass
                    adjs = frozenset(adjs)
                    sum_big = 0
                    sum_small = 0

                    for adj in adjs:
                        sum_big += core_majority_adj_big.get(adj, 0)
                        sum_small += core_majority_adj_small.get(adj, 0)
                    sum_big += core_majority_nouns_big.get(head, 0)
                    sum_small += core_majority_nouns_small.get(head, 0)

                    if sum_big > sum_small:
                        clusters_body_cleared[j].remove(inters_instance)
                        continue
                    elif sum_big < sum_small:
                        clusters_body_cleared[i].remove(inters_instance)
                        continue
                    else:
                        for phr in clusters_body_cleared[i]:
                            sum_big += self.model.n_similarity(phr.split(" "), inters_instance.split(" "))
                        for phr in clusters_body_cleared[j]:
                            sum_small += self.model.n_similarity(phr.split(" "), inters_instance.split(" "))
                        n_sum_big = sum_big / len(clusters_body_cleared[i])
                        n_sum_small = sum_small / len(clusters_body_cleared[j])
                        if n_sum_big >= n_sum_small:
                            clusters_body_cleared[j].remove(inters_instance)
                            continue
                        if n_sum_big < n_sum_small:
                            clusters_body_cleared[j].remove(inters_instance)
                            continue
        return clusters_body_cleared

    def add_border_points(self, cl_collection):

        self.logger.info(MESSAGES["border"])
        all_matched_phrases = set().union(*cl_collection)
        not_merged = set(self.sim_df.index) - all_matched_phrases

        phrase_cluster_dict = {ph: i for i, cl in enumerate(cl_collection) for ph in cl}
        points_to_add = {}
        for phrase in not_merged:
            sim_phrases = [x for x in sorted(self.sim_dict[phrase], reverse=True, key=lambda x: x[1])]
            if len(sim_phrases) == 0:
                points_to_add[phrase] = None
                continue
            matching_clusters = {}
            new_cl = 0
            new_score = 0
            nes_1 = list(filter(lambda x: x is not None, [self.ne_phrase_dict[frozenset(phrase.split(" "))]]))

            for good_match, score in sim_phrases:
                if good_match not in all_matched_phrases:
                    new_score += score
                    new_cl += 1
                    continue
                m1 = matching_clusters.get(phrase_cluster_dict[good_match], (0, 0))[0] + 1
                m2 = matching_clusters.get(phrase_cluster_dict[good_match], (0, 0))[1] + score
                matching_clusters[phrase_cluster_dict[good_match]] = (m1, m2)

            potential_boarders = sorted(
                list(filter(lambda x: x[1][0] >= self.params.min_border_match, matching_clusters.items())),
                reverse=True, key=lambda x: (x[1][1] / x[1][0], x[1][0]))

            if len(potential_boarders) > 0:
                nes_2 = list(filter(lambda x: x is not None,
                                    list(set(
                                        [self.ne_phrase_dict[frozenset(n.split(" "))] for n in
                                         cl_collection[potential_boarders[0][0]]]))))
                if len(nes_1) > 0 and len(nes_2) > 0 and np.sum(np.array(self.ne_df.loc[nes_1, nes_2] == 0)) > 0:
                    continue
                # print(potential_boarders[0][0], phrase, potential_boarders[0][1][1] / potential_boarders[0][1][0])
                points_to_add[phrase] = potential_boarders[0][0]
        cl_collection_modified = copy.deepcopy(cl_collection)
        for phrase, potential_boarders in points_to_add.items():
            if potential_boarders is not None:
                cl_collection_modified[potential_boarders].add(phrase)
            else:
                cl_collection_modified.append({phrase})
        return cl_collection_modified, points_to_add, not_merged

    def add_non_core_based_clusters(self, cl_collection, not_merged, points_to_add):
        self.logger.info(MESSAGES["non-core"])

        # TODO check against comp grid
        form_new_clusters = not_merged - set(points_to_add.keys())
        unmatched_entities = []
        phrases_in_new_cl = set()

        for phrase in form_new_clusters:
            if phrase in phrases_in_new_cl:
                continue
            old_entity = set()
            new_entity, phrases_in_new_cl = self.check_phrase(phrase, old_entity, form_new_clusters,
                                                              phrases_in_new_cl)
            while len(new_entity) > len(old_entity):
                old_entity = copy.deepcopy(new_entity)
                new_entity = set()
                for phr in old_entity:
                    new_entity_, phrases_in_new_cl = self.check_phrase(phr, old_entity, form_new_clusters,
                                                                       phrases_in_new_cl)
                    new_entity = new_entity.union(new_entity_)
            unmatched_entities.append(new_entity)

        cl_collection.extend(unmatched_entities)
        return cl_collection, unmatched_entities

    def check_phrase(self, phrase, old_entity, form_new_clusters, phrases_in_new_cl):
            entity = copy.deepcopy(old_entity)
            for match, score in self.sim_dict[phrase]:
                if match == phrase:
                    continue
                if match in form_new_clusters and match not in phrases_in_new_cl:
                    if score >= self.params.min_cluster_sim:
                        entity.add(match)
                        phrases_in_new_cl.add(match)
                        continue
            entity.add(phrase)
            phrases_in_new_cl.add(phrase)
            return entity, phrases_in_new_cl

    def create_entities(self, core_clusters_no_aliens, clusters_body_no_aliens, clusters_final, points_to_add,
                        unmatched_entities):
        # ---FORM ENTITIES---#
        self.logger.info(MESSAGES["form_entities"])
        phrase_entity_dict = {}

        for ph in set().union(*clusters_final):
            key_set = frozenset(ph.split(" "))
            if len(self.phrases_non_ne_entities[key_set]) > 1 or key_set in self.big_to_small_dict:
                main_ent_key = self.phrases_non_ne_entities[key_set][0]
                entity = self.entity_dict[main_ent_key]
                to_remove_from_queue = set()
                appendix = ""

                for key in self.phrases_non_ne_entities[key_set][1:]:
                    entity.absorb_entity(self.entity_dict[key], self.step_name + XCOREF_3_SUB_0, 1.0)
                    appendix = XCOREF_3_SUB_0
                to_remove_from_queue = to_remove_from_queue.union(set(self.phrases_non_ne_entities[key_set][1:]))

                if key_set in self.big_to_small_dict:
                    dep_entity_keys = self.phrases_non_ne_entities[self.big_to_small_dict[key_set]]
                    for key in dep_entity_keys:
                        entity.absorb_entity(self.entity_dict[key], self.step_name + XCOREF_3_SUB_1, 1.0)
                        appendix = XCOREF_3_SUB_1
                    to_remove_from_queue = to_remove_from_queue.union(set(dep_entity_keys))

                new_name = self.update_entity_queue(entity, list(to_remove_from_queue), self.step_name + appendix,
                                                    False)
                phrase_entity_dict[ph] = new_name
            else:
                phrase_entity_dict[ph] = self.phrases_non_ne_entities[key_set][0]

        all_core_points = set([s[5:] for s in set().union(*core_clusters_no_aliens)])
        all_body_points = set().union(*clusters_body_no_aliens) - all_core_points
        all_border_points = set(points_to_add.keys())
        all_noncore_points = set().union(*unmatched_entities)

        phrase_priority = {p: i + 1 for i, set_ in
                           enumerate([all_core_points, all_noncore_points, all_body_points, all_border_points])
                           for p in set_}

        priority_df = pd.DataFrame(phrase_priority, index=[PRIORITY]).T

        for phrase_set in clusters_final:
            if not len(phrase_set):
                continue
            priority_df_local = priority_df.loc[list(phrase_set)]
            top_prior = min(priority_df_local[PRIORITY].values)
            top_prior_list = list(priority_df_local[priority_df_local[PRIORITY] == top_prior].index)
            local_sim_df = self.sim_df.loc[top_prior_list, top_prior_list]
            main_phrase = top_prior_list[int(np.argmax([sum(local_sim_df.loc[ph].values) / (len(top_prior_list) - 1)
                                                    for ph in top_prior_list]))]
            priority_df_local.loc[main_phrase, PRIORITY] = 0
            prir_str_dict = {0: "", 1: XCOREF_3_SUB_2, 2: XCOREF_3_SUB_3, 3: XCOREF_3_SUB_4, 4: XCOREF_3_SUB_5}

            priority_list = list(reversed(list(set(priority_df_local[PRIORITY].values))))
            for i in range(len(priority_list) - 1):
                high_pr = list(priority_df_local[priority_df_local[PRIORITY] == priority_list[i + 1]].index)
                low_pr = list(priority_df_local[priority_df_local[PRIORITY] == priority_list[i]].index)
                pr_dict = {}
                for phr in low_pr:
                    mtch_phr = high_pr[int(np.argmax(self.sim_df.loc[phr, high_pr].values))]
                    if mtch_phr not in pr_dict:
                        pr_dict[mtch_phr] = []
                    pr_dict[mtch_phr].append(phr)

                for h_phr, l_phrs in pr_dict.items():
                    to_remove_from_queue = set()
                    main_entity = self.entity_dict[phrase_entity_dict[h_phr]]
                    for l_phr in l_phrs:
                        entity = self.entity_dict[phrase_entity_dict[l_phr]]
                        sim = self.sim_df.loc[h_phr, l_phr]
                        if sim == 0:
                            sim = self.model.n_similarity(h_phr.split(" "), l_phr.split(" "))
                        main_entity.absorb_entity(entity, self.step_name + prir_str_dict[priority_list[i]], sim)
                        to_remove_from_queue.add(phrase_entity_dict[l_phr])

                    if priority_list[i + 1] == 0:
                        new_name = self.update_entity_queue(main_entity, list(to_remove_from_queue),
                                                             self.step_name + prir_str_dict[priority_list[i]])
                    else:
                        new_name = self.update_entity_queue(main_entity, list(to_remove_from_queue),
                                                             self.step_name + prir_str_dict[priority_list[i]], False)
                    phrase_entity_dict[h_phr] = new_name

    def merge_big_entities(self, entity_dict):
        entity_types = set(self.table[self.table == 1].stack().reset_index()["level_0"].values)
        entity_types = entity_types.union(set(self.table[self.table == 2].stack().reset_index()["level_0"].values))

        non_ne_entities = {key: entity for key, entity in entity_dict.items()
                           if entity.type in entity_types and len(entity.members) > 1}
        from sklearn.feature_extraction.text import TfidfTransformer
        tfidf = TfidfTransformer()
        word_count_df = pd.DataFrame()

        overal_word_dict = {}
        ne_ent_dict = {}
        for ent_key, ent in non_ne_entities.items():
            final_word_dict = {}
            ne_words = []
            for word_dict in [ent.appos_dict, ent.adjective_dict, ent.compound_dict, ent.nmod_dict, ent.nummod_dict,
                              {k: len(v) for k, v in ent.headwords_cand_tree.items()}]:
                for k, v in word_dict.items():
                    if k in string.punctuation:
                        continue
                    fixed_word = k.replace("#", "").lower()
                    fixed_word = fixed_word[:-1] if fixed_word[-1] == "s" else fixed_word  # todo better lemmas
                    final_word_dict[fixed_word] = final_word_dict.get(fixed_word, 0) + v
                    if k[0].isupper():
                        ne_words.append(k)
            word_count_df = word_count_df.append(pd.DataFrame(final_word_dict, index=[ent_key]))
            overal_word_dict[ent_key] = final_word_dict
            ne_ent_dict[ent_key] = ne_words
        word_count_df.fillna(0, inplace=True)

        if not len(word_count_df):
            return entity_dict

        tfidf_vectors_df = pd.DataFrame(tfidf.fit_transform(word_count_df).toarray(), index=list(word_count_df.index),
                                        columns=list(word_count_df.columns))
        # import math
        # norm_word_count_df = pd.DataFrame()
        # for index, row in word_count_df.iterrows():
        #     norm_word_count_df = norm_word_count_df.append(row.div(math.log(len(self.entity_dict[index].members))))

        vectors_df = pd.DataFrame(columns=["d" + str(i) for i in range(self.model.vector_size)])
        for ent_key, ent in non_ne_entities.items():
            vectors_df = vectors_df.append(pd.DataFrame(self.model.query(list(overal_word_dict[ent_key]),
                                                                         list(tfidf_vectors_df.loc[ent_key, list(
                                                                             overal_word_dict[ent_key])].values)),
                                                        index=[ent_key],
                                                        columns=["d" + str(i) for i in range(self.model.vector_size)]))
        from sklearn.metrics.pairwise import cosine_similarity as cs

        sim_df = pd.DataFrame(cs(vectors_df, vectors_df), index=list(vectors_df.index), columns=list(vectors_df.index))

        sim_copy_df = sim_df.copy()
        for index in list(sim_df.index):
            ne_sel_words_1 = [w for w in ne_ent_dict[index] if w in list(self.ne_df.index)]
            for col in list(sim_df.columns):
                ne_sel_words_2 = [w for w in ne_ent_dict[col] if w in list(self.ne_df.index)]
                if not len(ne_sel_words_1) or not len(ne_sel_words_2):
                    ne_match = 0
                else:
                    ne_match_df = self.ne_df.loc[ne_sel_words_1, ne_sel_words_2]
                    ne_match = np.sum(ne_match_df.values)
                if col == index or sim_copy_df.loc[index, col] < self.params.min_feature_ratio - self.params.eps \
                        or ne_match == 0:
                    sim_copy_df.loc[index, col] = 0

        clusters = []
        added = set()

        for word in list(sim_copy_df.index):
            if word in added:
                continue
            old_cluster = {}
            if len(sim_copy_df[sim_copy_df[word] > 0]) == 0:
                continue
            new_cluster = {word}
            while len(old_cluster) != len(new_cluster):
                old_cluster = copy.deepcopy(new_cluster)
                for val in old_cluster:
                    if val in added:
                        continue
                    added.add(val)
                    new_cluster = new_cluster.union(set(sim_copy_df[sim_copy_df[val] > 0].index))
            clusters.append(list(new_cluster))

        a = 1
        for cluster in clusters:
            cluster = sorted(cluster, reverse=True, key=lambda x: len(self.entity_dict[x].members))
            main_entity = self.entity_dict[cluster[0]]
            for ent_key in cluster[1:]:
                main_entity.absorb_entity(self.entity_dict[ent_key], self.step_name, sim_copy_df.loc[cluster[0], ent_key])
            self.update_entity_queue(main_entity, cluster[1:], self.step_name, True)

        return {key: value for (key, value) in sorted(self.entity_dict.items(), reverse=True,
                                                      key=lambda x: len(x[1].members))}
