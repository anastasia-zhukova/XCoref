from cdcr.entities.identifier import Identifier
from cdcr.structures.entity_set import EntitySet
from cdcr.structures.entity import Entity
from cdcr.entities.entity_preprocessor import EntityPreprocessor
from cdcr.config import *
from cdcr.entities.const_dict_global import *
import progressbar

MESSAGES = {
    "vector_progress": "PROGRESS: Created word vectors for %(value)d/%(max_value)d (%(percentage)d %%) entities (in: %(elapsed)s).",
    "entities": "Forming final entities: "
}


class LemmaIdentifier(Identifier):

    logger = LOGGER

    def __init__(self, docs):
        super().__init__(docs)
        self.params = docs.configuration.entity_identifier_config.params

    def extract_entities(self) -> EntitySet:

        ent_preprocessor = EntityPreprocessor(self.docs, Entity)
        self.entity_dict = ent_preprocessor.entity_dict_construction()
        # self.entity_dict = {key: value for (key, value) in sorted(self.entity_dict.items(), reverse=True,
        #                                                           key=lambda x: len(x[1].members))}

        # widgets = [
        #     progressbar.FormatLabel(MESSAGES["vector_progress"])
        # ]
        # bar = progressbar.ProgressBar(widgets=widgets,
        #                               maxval=len(self.entity_dict)).start()
        lemma_dict = {}
        for ent_name, entity in self.entity_dict.items():
            for member in entity.members:
                lemma = member.head_token.lemma.lower()
                if lemma not in lemma_dict:
                    lemma_dict[lemma] = []
                lemma_dict[lemma].append(ent_name)
                break

        entity_dict_updated = {}
        for lemma, entity_keys in lemma_dict.items():
            if len(entity_keys) == 1:
                key = entity_keys[0]
                entity_dict_updated[key] = self.entity_dict[key]
                # self.logger.info("Entity: " + key)
            else:
                main_ent = self.entity_dict[entity_keys[0]]

                for ent_key in entity_keys[1:]:
                    # self.logger.info(main_ent.name + "  <--  " + self.entity_dict[ent_key].name)
                    main_ent.add_members(self.entity_dict[ent_key].members)

                main_ent.update_entity(LEMMA)
                entity_dict_updated[main_ent.name] = main_ent
                # self.logger.info("Entity: " + main_ent.name)

        entity_set = EntitySet(identification_method=self.docs.configuration.entity_method, topic=self.docs.topic)
        entity_set.extend(list(entity_dict_updated.values()))
        entity_set.sort(reverse=True, key=lambda x: len(x.members))
        return entity_set
