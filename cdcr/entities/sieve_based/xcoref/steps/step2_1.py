from cdcr.entities.sieve_based.step import Sieve
from cdcr.entities.const_dict_global import *

import re
import numpy as np


class XCorefStep1(Sieve):
    """
    A step merges entities using matching NE phrases of entity core phrases.
    """

    def __init__(self, step, docs, entity_dict, ent_preprocessor, model):

        super().__init__(XCOREF_1, step, docs.configuration.entity_identifier_config, entity_dict, ent_preprocessor,
                         model)

    def merge(self) -> dict:
        return self.iterate_over_entities()

    def find_similar_entities(self, entity_current):
        repr_current = [re.sub(r'\W+', '', w) for w in list(entity_current.core_mentions)]
        heads_current = set().union(*[list(entity_current.headwords_cand_tree.keys()),
                              list(entity_current.appos_dict.keys())])
        compounds_current = set().union(*[list(entity_current.headwords_cand_tree.keys()),
                                          list(entity_current.compound_dict.keys())])
        type_current = entity_current.type
        wiki_current = entity_current.wiki_page_name

        to_remove_from_queue = {}

        if len(self.table[entity_current.type].values[np.where(self.table[entity_current.type] > 0)]) > 0:

            for key, entity_next in list(self.entity_dict.items())[1:]:
                if entity_next.last_step == self.step_name or len(entity_current.members) < len(entity_next.members):
                    continue

                repr_next = [re.sub(r'\W+', '', w) for w in list(entity_next.core_mentions)]
                heads_next = set().union(*[list(entity_next.headwords_cand_tree.keys()),
                                              list(entity_next.appos_dict.keys())])
                compounds_next = set().union(*[list(entity_next.headwords_cand_tree.keys()),
                                                  list(entity_next.compound_dict.keys())])
                type_next = entity_next.type
                wiki_next = entity_next.wiki_page_name

                if self.table[type_current][type_next] == 0:
                    continue

                if len(repr_current) == 0 or len(repr_next) == 0:
                    continue

                intersect = frozenset(repr_current).intersection(frozenset(repr_next))
                intersect_heads = heads_current.intersection(heads_next)
                intersect_compounds = compounds_current.intersection(compounds_next)

                is_ner = np.sum([word in self.ent_preprocessor.phrase_ner_dict for word in list(intersect)])
                is_ner_heads = np.sum([word in self.ent_preprocessor.phrase_ner_dict for word in list(intersect_heads)])
                is_ner_compounds = np.sum(
                    [word in self.ent_preprocessor.phrase_ner_dict for word in list(intersect_compounds)])

                if is_ner > 0:
                    to_remove_from_queue[key] = 1.0
                    continue

                if is_ner_heads > 0 and is_ner_compounds > 0:
                    if (wiki_next is not None and wiki_next == wiki_current) or bool(wiki_next) ^ bool(wiki_current):
                        to_remove_from_queue[key] = 1.0
                        continue

                    else:
                        comp_inters_1 = heads_current.intersection(set(entity_next.compound_dict.keys()))
                        comp_inters_2 = heads_next.intersection(set(entity_current.compound_dict.keys()))
                        if len(comp_inters_1) and np.sum([entity_next.compound_dict[v] for v in comp_inters_1]) > 1:
                            to_remove_from_queue[key] = 1.0
                            continue

                        if len(comp_inters_2) and np.sum([entity_current.compound_dict[v] for v in comp_inters_2]) > 1:
                            to_remove_from_queue[key] = 1.0
                            continue

                        if is_ner_compounds > 1 or is_ner_heads > 1:
                            to_remove_from_queue[key] = 1.0
                            continue

                        if self.table[type_current][type_next] > 1 and \
                                (repr_next[0] in repr_current[0] or repr_current[0] in repr_next[0]):
                            to_remove_from_queue[key] = 1.0
                            continue
                if self.table[type_current][type_next] == 2:
                    comp_inters_2 = heads_next.intersection(set(entity_current.compound_dict.keys()))
                    if len(comp_inters_2):
                        is_head = list(comp_inters_2)[0] in self.ent_preprocessor.phrase_ner_dict
                        if is_head:
                            to_remove_from_queue[key] = 1.0
                            continue

        for key, sim in to_remove_from_queue.items():
            entity_current.absorb_entity(self.entity_dict[key], self.step_name, sim)

        return entity_current, to_remove_from_queue
