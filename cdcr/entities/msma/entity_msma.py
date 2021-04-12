from cdcr.structures.entity import Entity
from cdcr.entities.const_dict_global import *
from cdcr.entities.dict_lists import LocalDictLists


import pandas as pd
import string
import logging


class EntityMSMA(Entity):
    """
    An entity class designed for MSMA execution.
    """

    def __init__(self, document_set, ent_preprocessor, members, name, wikipage=None, core_mentions=None):

        super().__init__(document_set, members, name, ent_preprocessor, wikipage)

        self.word_dict = self._calc_word_dict()
        self.token_dict = self._build_token_dict()

        self.adjective_phrases, self.adjective_dict = {}, {}
        self.compound_phrases, self.compound_dict = {}, {}
        self.nmod_phrases, self.nmod_dict = {}, {}
        self.nummod_phrases, self.nummod_dict = {}, {}
        self.appos_phrases, self.appos_dict = [], {}
        self.acl_phrases, self.acl_dict = [], {}
        self.dobj_phrases, self.dobj_dict = {}, {}
        self.meronyms = None
        self._labeling_extraction()
        self.core_mentions = core_mentions if core_mentions is not None else self._get_core_mentions()

    def _get_core_mentions(self):
        compound_list = []
        for compound in list(self.compound_phrases):
            head = None
            compounds = []
            for word in list(compound):
                if word in self.headwords_cand_tree:
                    head = word
                else:
                    compounds.append(word)
            compounds.append(head)
            compound_list.append(" ".join(compounds))
        compound_list.extend(list(self.headwords_cand_tree))
        return compound_list

    def _calc_word_dict(self):
        word_count_dict = {}
        for cand in self.members:
            for token in cand.tokens:
                word_count_dict[token.word] = word_count_dict.get(token.word, 0) + 1
        return word_count_dict

    def _build_token_dict(self):
        token_dict = {}
        for m in self.members:
            token_dict.update({t.word: t for t in m.tokens})
        return token_dict

    def absorb_entity(self, other_entity, merge_type: str, sim_score: float):
        """
        Absorbs a provided entity in a current one and updates some of the internal parameters.

        Args:
            other_entity: A smaller entity to be merged into this one.
            merge_type: A name of the merge step at which the other entity was similar enough to be merged into this one.
            sim_score: Similarity score between two entities.

        """
        logging.info(self.name + "  <--  " + other_entity.name)
        self.merge_history[merge_type] = self.merge_history.get(merge_type, {
            NAME: self.name,
            SIZE: len(self.members),
            PHRASING_COMPLEXITY: self.phrasing_complexity,
            SIM_SCORE: 1.0,
            TYPE: self.type,
            MERGED_ENTITIES: []
        })

        self.merge_history[merge_type][MERGED_ENTITIES].append({
            NAME: other_entity.name,
            REPRESENTATIVE: other_entity.representative,
            SIZE: len(other_entity.members),
            PHRASING_COMPLEXITY: float(other_entity.phrasing_complexity),
            SIM_SCORE: float(sim_score) if type(sim_score) != dict else float(list(sim_score)[0]),
            TYPE: other_entity.type,
            MERGE_HISTORY: other_entity.merge_history})
        self.add_members(other_entity.members)

        # self._core_mentions = list(set(self._core_mentions).union(set(other.core_mentions)))
        core_mentions_low = [c.lower() for c in self.core_mentions]
        for men in other_entity.core_mentions:
            if men.lower() not in core_mentions_low:
                self.core_mentions.append(men)

        for head, cand_ids in other_entity.headwords_cand_tree.items():
            if head not in self.headwords_cand_tree:
                self.headwords_cand_tree[head] = []
            self.headwords_cand_tree[head].extend(cand_ids)

        # self._phrasing_complexity = self._wcl_metric_calc()

        for word, count in list(other_entity.word_dict.items()):
            self.word_dict[word] = self.word_dict.get(word, 0) + count

        self.token_dict.update(other_entity.token_dict)

        self.headwords_cand_tree = {key: value for (key, value) in sorted(self.headwords_cand_tree.items(), reverse=True,
                                                                          key=lambda x: len(x[1]))}

    def _labeling_extraction(self):

        def __update_dicts(word_dict, phrase_dict):
            word_dict[dep.dependent_gloss] = word_dict.get(dep.dependent_gloss, 0) + 1
            fr_set = frozenset([dep.dependent_gloss, dep.governor_gloss])
            if fr_set not in phrase_dict:
                phrase_dict[fr_set] = []
            phrase_dict[fr_set].append(cand.id)

        self.adjective_phrases, self.adjective_dict = {}, {}
        self.compound_phrases, self.compound_dict = {}, {}
        self.nmod_phrases, self.nmod_dict = {}, {}
        self.nummod_phrases, self.nummod_dict = {}, {}
        self.appos_phrases, self.appos_dict = [], {}
        self.acl_phrases, self.acl_dict = [], {}
        self.dobj_phrases, self.dobj_dict = {}, {}

        for cand in self.members:
            nmod_deps = []
            appos = None

            dep_df = pd.DataFrame(map(lambda x: x.dict, cand.dependency_subtree))

            for dep in cand.dependency_subtree:

                if dep.dep in [COMPOUND, AMOD, NUMMOD, NMOD, NMOD_POSS, APPOS, ACL, ACL_RELCL] \
                        and dep.governor_gloss == cand.head_token.word:

                    if dep.dependent_gloss.lower() in LocalDictLists.general_adj:
                        continue

                    if dep.dep == AMOD:
                        __update_dicts(self.adjective_dict, self.adjective_phrases)

                    if dep.dep == COMPOUND:
                        __update_dicts(self.compound_dict, self.compound_phrases)

                    if dep.dep == NUMMOD:
                        __update_dicts(self.nummod_dict, self.nummod_phrases)

                    if dep.dep == DOBJ:
                        __update_dicts(self.dobj_dict, self.dobj_phrases)

                    if dep.dep in [NMOD_POSS, NMOD] and dep.dependent_gloss not in LocalDictLists.stopwords \
                            and dep.dependent_gloss not in LocalDictLists.pronouns:
                        nmod_deps.append(dep.dependent_gloss)
                        self.nmod_dict[dep.dependent_gloss] = self.nmod_dict.get(dep.dependent_gloss, 0) + 1

                    if dep.dep == APPOS:
                        self.appos_dict[dep.dependent_gloss] = self.appos_dict.get(dep.dependent_gloss, 0) + 1
                        appos = dep.dependent_gloss

                    if dep.dep == ACL and\
                            dep_df[dep_df[GOVERNOR_GLOSS] == dep.governor_gloss][DEP].str.contains(PUNCT).any():
                        phrase_index = sorted([t.index for t in cand.tokens if t.index > dep.dependent])
                        phrase = [t.word for p in phrase_index for t in cand.tokens if t.index == p]
                        self.acl_phrases.append(" ".join(phrase))
                        for p in phrase:
                            if p not in LocalDictLists.stopwords and p not in string.punctuation \
                                    and p not in LocalDictLists.pronouns:
                                self.acl_dict[p] = self.acl_dict.get(p, 0) + 1

                    if dep.dep == ACL_RELCL:
                        phrase = list(dep_df[dep_df[GOVERNOR_GLOSS] == dep.dependent_gloss][DEPENDENT_GLOSS].values)\
                                 + [dep.dependent_gloss]
                        self.acl_phrases.append(" ".join(phrase))
                        for p in phrase:
                            self.acl_dict[p] = self.acl_dict.get(p, 0) + 1

                # if dep.dep in ["conj"] and dep.governor_gloss in list(self._adjectives_dict.keys()):

            nmod_deps_ = nmod_deps.copy()
            for dep in cand.dependency_subtree:
                if dep.governor_gloss in nmod_deps and dep.dependent_gloss not in LocalDictLists.stopwords \
                            and dep.dependent_gloss not in LocalDictLists.pronouns:
                    nmod_deps_.append(dep.dependent_gloss)
                    self.nmod_dict[dep.dependent_gloss] = self.nmod_dict.get(dep.dependent_gloss, 0) + 1

            if len(nmod_deps_) > 0:
                fr_set = frozenset(nmod_deps_ + [cand.head_token.word])
                if fr_set not in self.nmod_phrases:
                    self.nmod_phrases[fr_set] = []
                self.nmod_phrases[fr_set].append(cand.id)

            if appos is not None:
                df = pd.DataFrame(map(lambda x: x.dict, cand.dependency_subtree))
                try:
                    min_index = min(list(df[df[GOVERNOR_GLOSS] == appos][DEPENDENT].values) +
                                list(df[(df[DEPENDENT_GLOSS] == appos) & (df[DEP] == APPOS)][DEPENDENT].values))
                    max_index = max(list(df[(df[GOVERNOR_GLOSS] == appos)
                                            & (df[DEP].isin([NMOD, AMOD, COMPOUND]))][DEPENDENT].values) +
                                    list(df[(df[DEPENDENT_GLOSS] == appos)
                                            & (df[DEP] == APPOS)][DEPENDENT].values))
                    words = " ".join([t.word for t in cand.tokens
                                      if min_index <= t.index <= max_index])
                    self.appos_phrases.append(words)
                except ValueError:
                    continue

        self.adjective_phrases = {key: value for (key, value) in sorted(self.adjective_phrases.items(), reverse=True,
                                                                        key=lambda x: len(x[1]))}
        self.compound_phrases = {key: value for (key, value) in sorted(self.compound_phrases.items(), reverse=True,
                                                                       key=lambda x: len(x[1]))}
        self.nummod_phrases = {key: value for (key, value) in
                               sorted(self.nummod_phrases.items(), reverse=True,
                                      key=lambda x: len(x[1]))}
        self.nmod_phrases = {key: value for (key, value) in
                             sorted(self.nmod_phrases.items(), reverse=True,
                                    key=lambda x: len(x[1]))}