from cdcr.entities.identifier import Identifier
from cdcr.config import *
from cdcr.structures.entity import Entity
from cdcr.candidates.cand_enums import *
from cdcr.structures.entity_set import EntitySet
from cdcr.entities.entity_preprocessor import EntityPreprocessor

from datetime import datetime


class AsAnnotatedIdentifier(Identifier):

    logger = LOGGER

    def __init__(self, docs, entity_class=Entity):

        super().__init__(docs)

        if docs.candidates.origin_type != OriginType.ANNOTATED:
            raise ValueError("The candidates are not annotated, the golden mentions can't be assembled into entities.")

        self.config = docs.configuration.entity_identifier_config.params
        self.entity_dict = {}
        self.ent_preprocessor = EntityPreprocessor(docs, entity_class)

        # steps and entity need to be defined for a specific MSMA implementation
        self.entity_class = entity_class

    def extract_entities(self):
        cand_dict = {}
        for i, cand_group in enumerate(sorted(self.docs.candidates, reverse=True, key=lambda x: len(x))):
            for cand in cand_group:
                if cand.annot_label not in cand_dict:
                    cand_dict[cand.annot_label] = []
                cand_dict[cand.annot_label].append(cand)

        for name, cands in cand_dict.items():
            ent = Entity(self.docs, cands, name=name, ent_preprocessor=self.ent_preprocessor)
            self.entity_dict[ent.name] = ent

        self.entity_dict = {k:v for k,v in sorted(self.entity_dict.items(), reverse=True, key=lambda x: len(x[1].members))}
        entity_set = EntitySet(self.docs.configuration.entity_method, self.docs.topic, list(self.entity_dict.values()),
                               evaluation_details=None)

        json_fields = {"entities": [], "documents": [], "topic":  self.docs.topic}

        for doc in self.docs:
            doc_json = {"name": doc.source_domain, "text": [], "title": doc._title, "id": doc.id}
            for s in doc.sentences:
                s_json = []
                for t in s.tokens:
                    s_json.append({"word": t.word, "index": t.index, "ner": t.ner, "pos": t.pos, "after": t.after})
                doc_json["text"].append(s_json)
            json_fields["documents"].append(doc_json)

        for i, ent in enumerate(entity_set):
            json_fields["entities"].append({attr: val for attr, val in vars(ent).items() if
                                            attr in ["name", "type"]})
            json_fields["entities"][i]["representative"] = ent.representative
            json_fields["entities"][i]["id"] = ent.id
            json_fields["entities"][i]["size"] = len(ent.members)
            json_fields["entities"][i]["phrasing_complexity"] = float(ent.phrasing_complexity)
            json_fields["entities"][i]["mentions"] = []
            for m in ent.members:
                m_params = {}
                for attr, val in vars(m).items():
                    # if attr in ["annot_type"]:
                    #     m_params.update({attr: val})
                    if attr == "id":
                        m_params.update({attr: m.id})
                    if attr == "sentence":
                        m_params.update({attr: m.sentence.index})
                        m_params.update({"doc_id": m.document.id})
                    if attr == "tokens":
                        tokens_ids = []
                        tokens = []
                        tokens_text = ""
                        for t in m.tokens:
                            tokens.append(t.word)
                            tokens_ids.append(t.index)
                            tokens_text += t.word + t.after
                        m_params.update({attr: tokens_ids, "text": tokens_text, "tokens_text": tokens})
                    if attr == "head_token":
                        m_params.update({"head_token_index": m.head_token.index})
                        m_params.update({"head_token_word": m.head_token.word})
                json_fields["entities"][i]["mentions"].append(m_params)
            json_fields["entities"][i]["merging_history"] = ent.merge_history

        file_name = os.path.join(TMP_PATH, format(datetime.now(), "%Y-%m-%d_%H_%M_%S") + "_" + self.docs.topic +
                                 "_entity_data.json")

        with open(file_name, 'w') as outfile:
            json.dump(json_fields, outfile)
            logging.info("Exported entities into json files are saved to {0}".format(file_name))

        return entity_set
