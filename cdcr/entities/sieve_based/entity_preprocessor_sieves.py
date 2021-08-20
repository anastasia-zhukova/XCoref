from cdcr.entities.dict_lists import LocalDictLists
from cdcr.entities.const_dict_global import *
from cdcr.entities.entity_preprocessor import EntityPreprocessor
import cdcr.util.wiki_dict as wiki_dict
from eventlet import Timeout

import requests
import numpy as np

import progressbar
import string
import re
import wikipedia as wiki
import pandas as pd
import copy
import spacy


TIMEOUT = 10


MESSAGES = {
    "preproccess": "PROGRESS: Creating an entity out of the %(value)d-th/%(max_value)d (%(percentage)d %%) candidate group "
                   "(in: %(elapsed)s).",
    "key": "The key \"{}\" already exists. The new entity won't be added. "
}
nlp = spacy.load('en_core_web_sm')


class EntityPreprocessorSieves(EntityPreprocessor):
    """
    A class with custom process of entity preprocessing and constructing of entity_dict created for SIEVE_BASED.
    """
    def __init__(self, docs, entity_class):

        super().__init__(docs, entity_class)

        self.phrase_wiki_dict = wiki_dict.load()

    def entity_dict_construction(self):
        candidate_set = self.docs.candidates
        entity_dict = {}

        widgets = [
            progressbar.FormatLabel(MESSAGES["preproccess"])
        ]
        bar = progressbar.ProgressBar(widgets=widgets, maxval=len(candidate_set)).start()

        # ent_preprocessesor_light = copy.deepcopy(self)
        # ent_preprocessesor_light.model = None
        all_wiki_pages_dict = {v["title"]: v["page"] for v in list(self.phrase_wiki_dict.values())}
        summary_dict = {}

        for i, cand_group in enumerate(sorted(candidate_set, reverse=True, key=lambda x: len(x))):

            root_dict, cand_per_root, cand_dict, compound_dict = {}, {}, {}, {}
            non_compound_dict = {}
            self.appos_heads_dict = {}

            for cand in cand_group:

                if not self._leave_cand(cand):
                    continue

                cand_dict[cand.id] = cand
                self.cand_dict[cand.id] = cand

                compounds = []
                appos_root = ""
                acl_head = ""
                dep_df = pd.DataFrame(map(lambda x: x.dict, cand.dependency_subtree))

                for dep in cand.dependency_subtree:

                    if dep.governor_gloss == cand.head_token.word and dep.dep in [COMPOUND, AMOD, NMOD_POSS] \
                                and dep.dependent_gloss not in LocalDictLists.titles:
                        if dep.dependent_gloss[0].isupper():
                            compounds.append(dep.dependent)

                    if dep.dep == APPOS and dep.governor_gloss == cand.head_token.word \
                            and cand.head_token.ner != PERSON_NER:
                        # and cand.head_token.ner in [NON_NER, TITLE_NER]:
                        appos_root = (dep.dependent, dep.dependent_gloss)

                    if dep.dep == ACL and dep.governor_gloss == cand.head_token.word and\
                            dep_df[dep_df[GOVERNOR_GLOSS] == dep.governor_gloss][DEP].str.contains(PUNCT).any():
                        acl_df = dep_df[(dep_df[GOVERNOR] > dep.dependent) & (dep_df[DEPENDENT] > dep.dependent)]
                        potential_head = list(set(acl_df[GOVERNOR_GLOSS].values) - set(acl_df[DEPENDENT_GLOSS].values))
                        if len(potential_head) > 0:
                            index = [t.word for t in cand.tokens].index(potential_head[0])
                            acl_head = (dep.dependent, potential_head[0]) if NN in cand.tokens[index].pos else ""

                if appos_root:
                    compounds = []
                    for dep in cand.dependency_subtree:
                        if dep.governor_gloss == appos_root[1] and dep.dep in [COMPOUND, AMOD, NMOD, NMOD_POSS]:
                            compounds.append(dep.dependent)
                    if cand.head_token.word not in self.appos_heads_dict:
                        self.appos_heads_dict[cand.head_token.word] = []
                    self.appos_heads_dict[cand.head_token.word].append(appos_root[1])

                if acl_head:
                    if cand.head_token.word not in self.appos_heads_dict:
                        self.appos_heads_dict[cand.head_token.word] = []
                    self.appos_heads_dict[cand.head_token.word].append(acl_head[1])
                    phrase = " ".join([t.word for t in cand.tokens if t.index > acl_head[0]
                                       and t.word not in string.punctuation
                                       and t.word not in LocalDictLists.stopwords])
                    if phrase not in compound_dict:
                        compound_dict[phrase] = []
                    compound_dict[phrase].append((cand.head_token.word, cand.id))

                if len(compounds):
                    if not appos_root:
                        compounds.append(cand.head_token.index)
                    else:
                        if appos_root[1] in self.phrase_ner_dict:
                            compounds.append(appos_root[0])
                        else:
                            compounds.append(cand.head_token.index)
                    compounds.sort()
                    compound_phrase = " ".join([t.word for i in range(len(compounds)) for t in cand.tokens
                                                if compounds[i] == t.index])
                    if compound_phrase not in compound_dict:
                        compound_dict[compound_phrase] = []
                    compound_dict[compound_phrase].append((cand.head_token.word, cand.id))
                else:
                    if appos_root:
                        non_compound_dict[appos_root[1]] = non_compound_dict.get(appos_root[1], []) + \
                                                           [(appos_root[1], cand.id)]
                    else:
                        non_compound_dict[cand.head_token.word] = non_compound_dict.get(cand.head_token.word, []) + \
                                                                  [(cand.head_token.word, cand.id)]

            if not len(cand_dict):
                continue

            core_mentions_dict = copy.copy(compound_dict)
            core_mentions_dict.update(non_compound_dict)
            core_mentions = list(core_mentions_dict)

            # find wikipages for compound phrases

            WIKI, CORE, PHRASE_SIZE, NO_PAGE = "wiki", "core", "size", "no_page"
            wiki_df = pd.DataFrame(columns=[WIKI, PHRASE_SIZE])
            for core in list(filter(lambda x: x[0].isupper(), core_mentions)):
                split_phrase = core.split(" ")
                # phrase_sizes = [len(split_phrase), 3, 2] if len(split_phrase) > 3 else [len(split_phrase)]
                for phrase_size in [len(split_phrase), 3, 2]:
                    phrase = " ".join(split_phrase[-(min(phrase_size, len(split_phrase))):])
                    if phrase in self.phrase_wiki_dict:
                        if self.phrase_wiki_dict[phrase]["title"] is not None:
                            if core in list(wiki_df.index):
                                wiki_df.loc[core] = [self.phrase_wiki_dict[phrase]["title"], len(core_mentions_dict[core])]
                                break
                            wiki_df = wiki_df.append(pd.DataFrame({
                                WIKI: self.phrase_wiki_dict[phrase]["title"],
                                PHRASE_SIZE: len(core_mentions_dict[core])
                            }, index=[core]))
                            break

                    try:
                        wiki_page = None
                        with Timeout(TIMEOUT, False):
                            wiki_page = wiki.page(phrase, auto_suggest=False)

                        if wiki_page is not None:
                            self.phrase_wiki_dict[wiki_page.title] = {"title": wiki_page.title, "page": wiki_page}
                            self.phrase_wiki_dict[phrase] = {"title": wiki_page.title, "page": wiki_page}
                            all_wiki_pages_dict[wiki_page.title] = wiki_page

                            if core in list(wiki_df.index):
                                wiki_df.loc[core] = [wiki_page.title, len(core_mentions_dict[core])]
                                break

                            wiki_df = wiki_df.append(pd.DataFrame({
                                WIKI: wiki_page.title,
                                PHRASE_SIZE: len(core_mentions_dict[core])
                            }, index=[core]))
                            break
                        else:
                            if core in list(wiki_df.index):
                                continue
                            wiki_df = wiki_df.append(pd.DataFrame({
                                WIKI: None,
                                PHRASE_SIZE: len(core_mentions_dict[core])
                            }, index=[core]))
                            continue

                    except wiki.PageError:
                        self.phrase_wiki_dict[phrase] = {"title": None, "page": None}
                        if core in list(wiki_df.index):
                            continue
                        wiki_df = wiki_df.append(pd.DataFrame({
                            WIKI: None,
                            PHRASE_SIZE: len(core_mentions_dict[core])
                        }, index=[core]))
                        continue

                    except requests.exceptions.SSLError:
                        if core in list(wiki_df.index):
                            continue
                        wiki_df = wiki_df.append(pd.DataFrame({
                            WIKI: None,
                            PHRASE_SIZE: len(core_mentions_dict[core])
                        }, index=[core]))
                        continue

                    except wiki.DisambiguationError:
                        if core in list(wiki_df.index):
                            continue
                        if core not in self.phrase_ner_dict:
                            wiki_df = wiki_df.append(pd.DataFrame({
                                WIKI: None,
                                PHRASE_SIZE: len(core_mentions_dict[core])
                            }, index=[core]))
                            continue
                        elif any([core == cand_dict[core_mentions_dict[w][0][1]].head_token.word for w in list(wiki_df.index)]):
                            match_list = [cand_dict[core_mentions_dict[w][0][1]].head_token.word for w in list(wiki_df.index)]
                            core_match = match_list.index(core)
                            wiki_df = wiki_df.append(pd.DataFrame({
                                WIKI: wiki_df.iloc[core_match][WIKI],
                                PHRASE_SIZE: len(core_mentions_dict[core])
                            }, index=[core]))
                            wiki_page_title = wiki_df.iloc[core_match][WIKI]
                            if wiki_page_title is not None:
                                self.phrase_wiki_dict[phrase] = {"title": wiki_page_title, "page": all_wiki_pages_dict[wiki_page_title]}
                            # all_wiki_pages_dict[wiki_page.title] = wiki_page
                            break

                        else:
                            try:
                                wiki_page = None
                                with Timeout(TIMEOUT, False):
                                    wiki_page = wiki.page(phrase, auto_suggest=True)
                                if wiki_page is not None:
                                    self.phrase_wiki_dict[wiki_page.title] = {"title": wiki_page.title, "page": wiki_page}
                                    self.phrase_wiki_dict[phrase] = {"title": wiki_page.title, "page": wiki_page}
                                    all_wiki_pages_dict[wiki_page.title] = wiki_page

                                    if core in list(wiki_df.index):
                                        wiki_df.loc[core] = [wiki_page.title, len(core_mentions_dict[core])]
                                        break

                                    wiki_df = wiki_df.append(pd.DataFrame({
                                        WIKI: wiki_page.title,
                                        PHRASE_SIZE: len(core_mentions_dict[core])
                                    }, index=[core]))
                                    break
                                else:
                                    if core in list(wiki_df.index):
                                        continue
                                    wiki_df = wiki_df.append(pd.DataFrame({
                                        WIKI: None,
                                        PHRASE_SIZE: len(core_mentions_dict[core])
                                    }, index=[core]))
                                    continue

                            except (wiki.PageError, wiki.DisambiguationError):
                                self.phrase_wiki_dict[phrase] = {"title": None, "page": None}
                                if core in list(wiki_df.index):
                                    continue
                                wiki_df = wiki_df.append(pd.DataFrame({
                                    WIKI: None,
                                    PHRASE_SIZE: len(core_mentions_dict[core])
                                }, index=[core]))
                                continue
                            except requests.exceptions.SSLError:
                                if core in list(wiki_df.index):
                                    continue
                                wiki_df = wiki_df.append(pd.DataFrame({
                                    WIKI: None,
                                    PHRASE_SIZE: len(core_mentions_dict[core])
                                }, index=[core]))
                                continue
            a =1
            # when wikipages have "Donald Trump" and "Donald" options, leave only "Donald Trump" as the best option
            for w1 in list(wiki_df.groupby(WIKI).sum().sort_values(PHRASE_SIZE, ascending=False).index):
                if w1 is None:
                    continue

                for w2 in list(wiki_df.groupby(WIKI).sum().sort_values(PHRASE_SIZE, ascending=False).index):
                    if w2 is None:
                        continue
                    if w1 == w2:
                        continue
                    if w1 == w2.split(" ")[-1] or w1 == w2.split(" ")[0]:
                        rows = list(wiki_df[wiki_df[WIKI] == w1].index)
                        wiki_df.loc[rows, WIKI] = [w2] * len(rows)

                    if w2 == w1.split(" ")[-1] or w2 == w1.split(" ")[0]:
                        rows = list(wiki_df[wiki_df[WIKI] == w2].index)
                        wiki_df.loc[rows, WIKI] = [w1] * len(rows)

            # find the best wikipage representatives for the core phrases
            wiki_none = 0
            if len(wiki_df.dropna()):
                best_wiki_page = wiki_df.groupby(WIKI).sum().sort_values(PHRASE_SIZE, ascending=False).iloc[0].name

                for core in core_mentions:
                    if core in list(wiki_df.index):
                        if wiki_df.loc[core, WIKI] == best_wiki_page:
                            continue
                    for wiki_page, _ in wiki_df.groupby(WIKI).sum().sort_values(PHRASE_SIZE, ascending=False
                                                                                ).to_dict()[PHRASE_SIZE].items():
                        if core in list(wiki_df.index) or any([core in w for w in list(wiki_df.index)]):
                            if core in list(wiki_df.index):
                                if wiki_page == wiki_df.loc[core, WIKI]:
                                    continue
                            else:
                                wiki_df = wiki_df.append(pd.DataFrame({
                                    WIKI: wiki_page,
                                    PHRASE_SIZE: len(core_mentions_dict[core])
                                }, index=[core]))
                                break
                        else:
                            wiki_df = wiki_df.append(pd.DataFrame({
                                WIKI: None,
                                PHRASE_SIZE: len(core_mentions_dict[core])
                            }, index=[core]))

                        inters = set(re.findall(r"[\w]+", wiki_page)).intersection(re.findall(r"[\w]+", core))
                        if core.split(" ")[-1] in self.appos_heads_dict:
                            inters_appos = set(re.findall(r"[\w]+", self.appos_heads_dict[core.split(" ")[-1]][0]))\
                                .intersection(re.findall(r"[\w]+", wiki_page))
                        else:
                            inters_appos = set()

                        if len(inters) < min(len(wiki_page.split(" ")), len(core.split(" "))) and not len(inters_appos):
                            if wiki_page not in summary_dict:
                                try:
                                    summ = all_wiki_pages_dict[wiki_page].summary
                                    summary_dict[wiki_page] = summ
                                except KeyError:
                                    summ = ""
                            else:
                                try:
                                    summ = summary_dict[wiki_page]
                                except KeyError:
                                    summ = ""
                            preproc_summary = nlp(summ)
                            sents = [s for s in preproc_summary.sents]
                            if not len(sents):
                                continue

                            inters_summary = set([re.sub(r'\W', "", w.lower()) for w in [t.orth_ for t in sents[0]]]).intersection(
                                set([w.lower() for w in core.split(" ")]))
                            if not len(inters_summary):
                                continue
                        elif len(inters):
                            if not any([w.istitle() for w in inters]):
                                continue
                        elif len(inters_appos):
                            if not np.sum([w.istitle() for w in inters_appos]):
                                continue
                        # if not len(inters) and not len(inters_appos):
                        #     continue

                        wiki_df.loc[core, WIKI] = wiki_page
                        # if core.istitle():
                        #     self.phrase_wiki_dict[core] = {"title": wiki_page, "page": all_wiki_pages_dict[wiki_page]}
                        break

                # determine membership of phrases without NE-based modifiers to wikipage-based cores
                for core in list(wiki_df[wiki_df.isnull().any(axis=1)].index):
                    for wiki_page, _ in wiki_df.groupby(WIKI).sum().sort_values(PHRASE_SIZE, ascending=False).to_dict()[
                                                                                                PHRASE_SIZE].items():
                        representatives = list(set(wiki_df[wiki_df[WIKI] == wiki_page].index).union({wiki_page}))
                        modifiers = [w.istitle() and w not in representatives
                                                                for w in list(re.findall(r"[\w]+", core))[:-1]]
                        if cand_dict[core_mentions_dict[core][0][1]].coref_subtype == PRONOMINAL:
                            wiki_df.loc[core, WIKI] = wiki_page
                            break

                        if all(modifiers) and len(modifiers):
                            wiki_df.loc[core, WIKI] = NO_PAGE + str(wiki_none)
                            wiki_none += 1
                            break

                        if len(core.split(" ")) == 1 and (core not in wiki_page or
                                                      not any([core.lower() in v.lower() for v in list(wiki_df.index)])):
                            wiki_df.loc[core, WIKI] = NO_PAGE + str(wiki_none)
                            wiki_none += 1
                            break
                        wiki_df.loc[core, WIKI] = wiki_page
                        # if core.istitle():
                        #     self.phrase_wiki_dict[core] = {"title": wiki_page, "page": all_wiki_pages_dict[wiki_page]}
                        break
            else:
                wiki_df[WIKI] = [NO_PAGE + str(wiki_none)] * len(wiki_df)
                for core in core_mentions:
                    if core in list(wiki_df.index):
                        continue

                    wiki_df = wiki_df.append(pd.DataFrame({
                        WIKI: NO_PAGE + str(wiki_none),
                        PHRASE_SIZE: len(core_mentions_dict[core])
                    }, index=[core]))

            # form entities from wiki_df table values
            for wiki_page in set(wiki_df[WIKI].values):
                cores_df = wiki_df[wiki_df[WIKI] == wiki_page]
                entity = self.entity_class(document_set=self.docs,
                                           ent_preprocessor=self,
                                           members=list(set([cand_dict[key] for c in list(cores_df.index)
                                          for _, key in core_mentions_dict[c]])),
                                           name=None,
                                           wikipage=all_wiki_pages_dict[wiki_page] if wiki_page in all_wiki_pages_dict
                                           else None,
                                           core_mentions=list(cores_df.index))

                if entity.name in entity_dict:
                    self.logger.warning(MESSAGES["key"].format(entity.name))
                else:
                    entity_dict.update({entity.name: entity})
                    self._update_dicts(entity)

            bar.update(i + 1)
        bar.finish()

        self._update_dicts()
        wiki_dict.update(self.phrase_wiki_dict)
        return {key: value for (key, value) in sorted(entity_dict.items(), reverse=True,
                                                                   key=lambda x: len(x[1].members))}

    def _update_dicts(self, entity=None):
        if entity is not None:
            for adj, count in entity.adjective_dict.items():
                self.labeling_dict[adj] = self.labeling_dict.get(adj, 0) + count
            for comp, count in entity.compound_dict.items():
                self.compound_dict[comp] = self.compound_dict.get(comp, 0) + count
        else:
            self.labeling_dict = {key: value for (key, value) in sorted(self.labeling_dict.items(), reverse=True,
                                                                        key=lambda x: x[1])}
            self.compound_dict = {key: value for (key, value) in sorted(self.compound_dict.items(), reverse=True,
                                                                        key=lambda x: x[1])}
