from cdcr.entities.sieve_based.step import Sieve
from cdcr.entities.const_dict_global import *

import numpy as np
import wikipedia as wiki
import re
import requests


MESSAGES = {
    "wiki": "Checking for missing matching wikipages."
}


class XCorefStep0(Sieve):
    """
    A step merges entities using matching wikipedia pages related to the entities.
    """

    def __init__(self, step, docs, entity_dict, ent_preprocessor, model):

        super().__init__(XCOREF_0, step, docs.configuration.entity_identifier_config, entity_dict, ent_preprocessor,
                         model)

    def merge(self) -> dict:

        self.logger.info(MESSAGES["wiki"])
        # wiki_dict = {}
        # for key, ent in self.entity_dict.items():
        #     if ent.wiki_page_name is None:
        #         continue
        #
        #     for m in ent.members:
        #         if m.text in wiki_dict:
        #             continue
        #         wiki_dict[m.text] = {"name": ent.wiki_page_name, "page": ent.wiki_page}

        for key, ent in self.entity_dict.items():
            if ("-ne" in ent.type and ent.wiki_page_name is None) \
                    or (MISC_TYPE == ent.type and [v for v in ent.representative.split(" ") if len(v)][-1][0].istitle()
                        and ent.wiki_page_name is None):
                new_key = re.split("and", " ".join(key.split("_")[:-3]))[0]
                if new_key in self.ent_preprocessor.phrase_wiki_dict:
                    self.entity_dict[key].wiki_page_name = self.ent_preprocessor.phrase_wiki_dict[new_key]["title"]
                    self.entity_dict[key].wiki_page = self.ent_preprocessor.phrase_wiki_dict[new_key]["page"]
                else:
                    try:
                        wiki_page = wiki.page(new_key)
                        # self.logger.info(str((key, wiki_page.original_title)))
                        self.entity_dict[key].wiki_page_name = wiki_page.original_title
                        self.entity_dict[key].wiki_page = wiki_page
                        self.ent_preprocessor.phrase_wiki_dict[wiki_page.original_title] = {"title": wiki_page.original_title, "page": wiki_page}
                        self.ent_preprocessor.phrase_wiki_dict[new_key] = {"title": wiki_page.original_title, "page": wiki_page}
                    except (wiki.DisambiguationError, wiki.PageError, wiki.WikipediaException, requests.exceptions.SSLError):
                        continue

        return self.iterate_over_entities()

    def find_similar_entities(self, entity_current):
        repr_current = entity_current.wiki_page_name
        type_current = entity_current.type

        to_remove_from_queue = {}

        if len(self.table[entity_current.type].values[np.where(self.table[entity_current.type] > 0)]) > 0:

            for key, entity_next in list(self.entity_dict.items())[1:]:
                if entity_next.last_step == self.step_name or len(entity_current.members) < len(entity_next.members):
                    continue

                repr_next = entity_next.wiki_page_name
                type_next = entity_next.type

                if repr_current is None or repr_next is None:
                    continue

                if self.table[type_current][type_next] > 0 and repr_current == repr_next:
                    to_remove_from_queue[key] = 1.0

        for key, sim in to_remove_from_queue.items():
            entity_current.absorb_entity(self.entity_dict[key], self.step_name, sim)

        return entity_current, to_remove_from_queue
