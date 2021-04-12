from nltk.corpus import wordnet as wn
import re
from allennlp.predictors.predictor import Predictor
import logging
import gdown
import pkg_resources
from typing import *
import zipfile

from cdcr.entities.eecdcr.src.all_models.predict_model import test_model
from cdcr.entities.eecdcr.src.features.build_features import *
from cdcr.entities.eecdcr.src.features.allen_srl_reader import *
from cdcr.entities.eecdcr.src.features.create_elmo_embeddings import ElmoEmbedding
from cdcr.entities.eecdcr.src.shared.classes import Corpus, Topic, Document, Sentence, Token, EntityMention
from cdcr.entities.identifier import Identifier
from cdcr.structures.entity_set import EntitySet
from cdcr.structures.candidate import Candidate
from cdcr.entities.const_dict_global import *
from cdcr.config import *
from cdcr.candidates.cand_enums import *
from cdcr.structures.entity import Entity
from cdcr.structures.document_set import DocumentSet
from cdcr.entities.eecdcr.entity_preprocessor_eecdcr import EntityPreprocessorEECDCR


BERT_SRL = "BERT_SRL"
BERT_ONLINE_PATH = "https://s3-us-west-2.amazonaws.com/allennlp/models/bert-base-srl-2019.06.17.tar.gz"
MODELS_PATH = "https://drive.google.com/uc?export=download&confirm=9iOS&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM"

MESSAGES = {
    "srl": 'Loading SRL info...',
    "elmo": "Loading ELMO embeddings...",
    "dep": 'Augmenting predicate-arguments structures using dependency parser.',
    "r_l": 'Augmenting predicate-arguments structures using leftmost and rightmost entity mentions.',
    "allennlp": "Version of AllenNLP package is {}, which is lower than requires 0.8.5. "
                "The Semantic Role Labeling features won't be added.",
    "no_models": "Models for EECDCR ({}, {}) are not found. Please download them from "
                 "https://drive.google.com/open?id=197jYq5lioefABWP11cr4hy4Ohh1HMPGK and place into {}.",
    "entities": "Forming final entities: "
}
CAND_GROUP = "cand_group"
CLUSTER = "cluster"
ENTITIES = "entities"


class EECDCRIdentifier(Identifier):
    """
    A class is a bridge between newsalyze and EECDCR approach for entity identifier by Barhom et al "Revisiting Joint
    Modeling of  Cross-document Entity and Event Coreference Resolution" (https://arxiv.org/abs/1906.01753)
    The code is adapted from https://github.com/shanybar/event_entity_coref_ecb_plus.
    """

    logger = LOGGER

    def __init__(self, document_set: DocumentSet):
        super().__init__(document_set)

        self.config = document_set.configuration.entity_identifier_config.params
        self.mention_dict = defaultdict(Candidate)
        self.cand_dict = {}

    def extract_entities(self) -> EntitySet:

        if not os.path.exists(self.config.cd_event_model_path) or not os.path.exists(self.config.cd_entity_model_path):
            # output, _ = os.path.split(self.config.cd_event_model_path)
            # gdown.download(MODELS_PATH, output, quiet=False)
            # with zipfile.ZipFile(os.path.join(output, "trained_models_and_data.zip"), 'r') as zip_ref:
            #     zip_ref.extractall(output)

            folder, ev_model = os.path.split(self.config.cd_event_model_path)
            _, en_model = os.path.split(self.config.cd_entity_model_path)
            raise FileNotFoundError(MESSAGES["no_models"].format(ev_model, en_model, folder))

        # build data structure for eecdcr
        self._save_coref_mentions()
        corpus = self._prepare_corpus()

        # add features
        if self.config.use_srl:
            if pkg_resources.get_distribution("allennlp").version >= "0.8.5":
                self.logger.info(MESSAGES["srl"])
                srl_data = self.get_srl_data(corpus)
                match_allen_srl_structures(corpus, srl_data, self.config, is_gold=True)
            else:
                self.logger.warning(MESSAGES["allennlp"].format(pkg_resources.get_distribution("allennlp").version))

        if self.config.load_elmo:
            self.logger.info(MESSAGES["elmo"])
            elmo_embedder = ElmoEmbedding(self.config.options_file, self.config.weight_file)
            load_elmo_embeddings(corpus, elmo_embedder, set_pred_mentions=True)

        if self.config.use_dep:
            self.logger.info(MESSAGES["dep"])
            find_args_by_dependency_parsing(corpus, is_gold=True)

        if self.config.use_left_right_mentions:
            self.logger.info(MESSAGES["r_l"])
            find_left_and_right_mentions(corpus, is_gold=True)

        # run the model
        all_entity_clusters, all_event_clusters = test_model(corpus, self.config, perform_evaluation=False)

        # form entities
        ent_preprocessor = EntityPreprocessorEECDCR(self.docs, Entity)
        entity_dict, df = ent_preprocessor.entity_dict_construction(resolved_mentions=all_entity_clusters + all_event_clusters,
                                                                mention_dict=self.mention_dict)
        entity_dict_updated = {}
        self.logger.info(MESSAGES["entities"])
        for cluster in sorted(all_entity_clusters + all_event_clusters, reverse=True,
                                                                        key=lambda x: len(list(x.mentions.values()))):
            m_ids = list(cluster.mentions)
            df_cluster = df.loc[m_ids]
            df_index = df_cluster[[ENTITIES]].reset_index().groupby(by=[ENTITIES]).count()
            df_index.sort_values(by=["index"], ascending=[False], inplace=True)
            entity_keys = list(df_index.index)

            if not len(entity_keys):
                continue

            if len(entity_keys) == 1:
                key = entity_keys[0]
                entity_dict_updated[key] = entity_dict[key]
                self.logger.info("Entity: " + key)
            else:
                main_ent = entity_dict[entity_keys[0]]

                for ent_key in entity_keys[1:]:
                    self.logger.info(main_ent.name + "  <--  " + entity_dict[ent_key].name)
                    main_ent.add_members(entity_dict[ent_key].members)

                main_ent.update_entity(EECDCR)
                entity_dict_updated[main_ent.name] = main_ent
                self.logger.info("Entity: " + main_ent.name)

        entity_set = EntitySet(identification_method=self.docs.configuration.entity_method, topic=self.docs.topic)
        entity_set.extend(list(entity_dict_updated.values()))
        entity_set.sort(reverse=True, key=lambda x: len(x.members))
        return entity_set

    def _save_coref_mentions(self):
        """
        Saves resolved coreferences from corenlp into a json file required by EECDCR.

        """
        cand_list = []
        for cand_group in self.docs.candidates:
            if len(cand_group) == 1:
                continue

            for cand in cand_group:
                tokens_numbers = [t.index for t in cand.tokens]
                is_continuous = True if tokens_numbers == list(range(tokens_numbers[0], tokens_numbers[-1]+1)) else False

                cand_list.append({"doc_id": EECDCRIdentifier.get_doc_id(cand.document),
                                  "sent_id": cand.sentence.index,
                                  "tokens_numbers": tokens_numbers,
                                  "tokens_str": cand.text,
                                  "coref_chain": cand_group.group_name,
                                  "is_continuous": is_continuous,
                                  "is_singleton": False})

        with open(self.config.wd_entity_coref_file, 'w') as f:
            json.dump(cand_list, f)

    def _prepare_corpus(self) -> Corpus:
        """
        Converts our document and candidates structures into EECDCR structures.
        Returns:
            Corpus with documents, sentences, tokens, and assigned mentions.

        """
        sentence_id_to_event_candidates = defaultdict(list)
        sentence_id_to_entity_candidates = defaultdict(list)
        token_id_to_event_candidates = defaultdict(list)
        token_id_to_entity_candidates = defaultdict(list)

        for group_id, cand_group in enumerate(self.docs.candidates):
            for cand in cand_group:
                is_entity = EECDCRIdentifier.is_entity(cand)

                if is_entity:
                    eecdcr_entity_mention = self.convert_to_eecdcr_mention(cand, len(cand_group), group_id,
                                                                                       is_entity)
                    self.mention_dict[eecdcr_entity_mention.mention_id] = cand
                    self.cand_dict[eecdcr_entity_mention.mention_id] = cand_group.group_name
                    sentence_id_to_entity_candidates[EECDCRIdentifier.make_id(cand)].append(eecdcr_entity_mention)
                    for token in cand.tokens:
                        token_id_to_entity_candidates[EECDCRIdentifier.make_id(cand, token.index)].append(
                                                                                                eecdcr_entity_mention)
                else:
                    eecdcr_event_mention = self.convert_to_eecdcr_mention(cand, len(cand_group), group_id,
                                                                                      is_entity)
                    self.mention_dict[eecdcr_event_mention.mention_id] = cand
                    self.cand_dict[eecdcr_event_mention.mention_id] = cand_group.group_name
                    sentence_id_to_event_candidates[EECDCRIdentifier.make_id(cand)].append(eecdcr_event_mention)
                    for token in cand.tokens:
                        token_id_to_event_candidates[EECDCRIdentifier.make_id(cand, token.index)].append(
                                                                                                eecdcr_event_mention)

        corpus = Corpus()
        sentence_id_to_event_candidates = EECDCRIdentifier.sort_dict(sentence_id_to_event_candidates)
        sentence_id_to_entity_candidates = EECDCRIdentifier.sort_dict(sentence_id_to_entity_candidates)
        token_id_to_event_candidates = EECDCRIdentifier.sort_dict(token_id_to_event_candidates)
        token_id_to_entity_candidates = EECDCRIdentifier.sort_dict(token_id_to_entity_candidates)

        topic_id = self.docs.topic
        topic = Topic(topic_id)

        for doc in self.docs:
            doc_id = EECDCRIdentifier.get_doc_id(doc)
            eecdcr_doc = Document(doc_id)

            for sent in doc.sentences:
                sent_id = sent.index
                eecdcr_sent = Sentence(sent_id)

                full_sent_id = "{}_{}".format(doc_id, str(sent_id))
                eecdcr_sent.gold_event_mentions += sentence_id_to_event_candidates.get(full_sent_id, [])
                eecdcr_sent.gold_entity_mentions += sentence_id_to_entity_candidates.get(full_sent_id, [])

                for token in sent.tokens:
                    token_id = token.index
                    token_text = token.word
                    eecdcr_token = Token(token_id, token_text, [])

                    full_token_id = full_sent_id + "_" + str(token_id)
                    eecdcr_token.gold_event_coref_chain += token_id_to_event_candidates.get(full_token_id, [])
                    eecdcr_token.gold_entity_coref_chain += token_id_to_entity_candidates.get(full_token_id, [])
                    eecdcr_sent.add_token(eecdcr_token)

                eecdcr_doc.add_sentence(sent_id, eecdcr_sent)

            topic.add_doc(doc_id, eecdcr_doc)

        corpus.add_topic(topic_id, topic)
        return corpus

    def convert_to_eecdcr_mention(self, cand: Candidate, len_cand_group: int, group_id: int, is_entity: bool) \
            -> Mention:
        """
        Converts a candidate to a eecdcr mention.

        Args:
            cand: A candidate object.
            len_cand_group: Length of a candidate's group.
            group_id: An ID of a candidate's group.
            is_entity: A flag indicating if a candidate is an entity.

        Returns:
            A converted candidate into a mention object.
        """

        eecdcr_tokens = []
        for token in cand.tokens:
            token_id = token.index
            token_text = token.word
            eecdcr_token = Token(token_id, token_text)
            eecdcr_tokens.append(eecdcr_token)

        tokens_numbers = [t.index for t in cand.tokens]
        tokens = eecdcr_tokens
        mention_str = cand.text
        head_text = cand.head_token.word
        head_lemma = cand.head_token.lemma
        is_singleton = False if len_cand_group > 1 else True
        is_continuous = True if tokens_numbers == list(range(tokens_numbers[0], tokens_numbers[-1]+1)) else False
        mention_type = cand.annot_type if cand.annot_type is not None else cand.coref_subtype
        doc_id = EECDCRIdentifier.get_doc_id(cand.document)

        if is_entity:
            em = EntityMention(doc_id, cand.sentence.index, tokens_numbers, tokens, mention_str, head_text,
                           head_lemma, is_singleton, is_continuous, group_id, mention_type)
        else:
            if self.docs.candidates.origin_type == OriginType.ANNOTATED:
                em = EventMention(doc_id, cand.sentence.index, tokens_numbers, tokens, mention_str, head_text,
                               head_lemma, is_singleton, is_continuous, group_id)
            else:
                em = EventMention(doc_id, cand.sentence.index, [cand.head_token.index],
                                  [Token(cand.head_token.index, cand.head_token.word)], cand.head_token.word, head_text,
                               head_lemma, is_singleton, True, group_id)
        return em

    @staticmethod
    def is_entity(cand: Candidate) -> bool:
        """
        Identifies if a candidate is an entity or event.

        Args:
            cand: A candidate object

        Returns:
            A flag is a candidate is an entity.
        """
        if cand.origin_type == OriginType.ANNOTATED and hasattr(cand, "file_orig"):
            if cand.file_orig == "event":
                return False
            return True
        else:
            if cand.type == CandidateType.VP:
                return False

            if cand.head_token.ner != NON_NER:
                return True

            synsets = wn.synsets(cand.head_token.word)
            is_act = any([s.lexname() == ACT_WN for s in synsets])
            if is_act:
                return False
            return True

    @staticmethod
    def make_id(cand: Candidate, token_index: int = None) -> str:
        """
        Creates a unique ID.

        Args:
            cand: A candidate object.
            token_index: An index of candidate's head token.

        Returns:
            A string ID.
        """
        doc_id = EECDCRIdentifier.get_doc_id(cand.document)
        if token_index is not None:
            return "{}_{}_{}".format(doc_id, str(cand.sentence.index), str(token_index))
        return "{}_{}".format(doc_id, str(cand.sentence.index))

    @staticmethod
    def get_doc_id(doc: Document) -> str:
        """
        Generates a unique doc id.
        Args:
            doc: A document object.

        Returns:
            A string ID.
        """
        return "{}_{}".format(re.sub(r'\W+', "", doc.source_domain), str(doc.id))

    @staticmethod
    def sort_dict(d: dict) -> dict:
        """
        Sorts a dictionary in the reverse order of its values.
        """
        return {k:v for k,v in sorted(d.items(), reverse=False, key=lambda x: x[0])}

    @staticmethod
    def check_tag(tag: str, tag_id: int, srl_verb_obj: SRLVerb, attr: str, words: List[str]):
        """
        Checks tags from SRL and initialize SRL objects from EECDCR.

        Args:
            tag: A SRL tag.
            tag_id: A SRL tag id.
            srl_verb_obj: A SRL verb object from EECDCR.
            attr: An attribute for which we need to check in tags.
            words: A list of words from SRL tagger.

        """
        tag_attr_dict = {"ARG0": "arg0",
                         "ARG1": "arg1",
                         "TMP": "arg_tmp",
                         "LOC": "arg_loc",
                         "NEG": "arg_neg"}
        if attr in tag:
            if tag[0] == "B":
                setattr(srl_verb_obj, tag_attr_dict[attr], SRLArg(words[tag_id], [tag_id]))
            else:
                srl_arg = getattr(srl_verb_obj, tag_attr_dict[attr])
                if srl_arg is None:
                    srl_arg = SRLArg("", [])
                srl_arg.text += " " + words[tag_id]
                srl_arg.ecb_tok_ids.append(tag_id)
                setattr(srl_verb_obj, tag_attr_dict[attr], srl_arg)

    def get_srl_data(self, corpus: Corpus) -> Dict[str, Dict[str, SRLSentence]]:
        """
        Extracts labels from semantic role labeling (SRL).

        Args:
            corpus: A EECDCE document collection object.

        Returns:
            A dictionary with EECDCR SRL sentence structures.

        """

        if not os.path.exists(self.config.bert_file):
            url = BERT_ONLINE_PATH
            output, _ = os.path.split(self.config.bert_file)
            gdown.download(url, output, quiet=False)

        predictor = Predictor.from_path(self.config.bert_file)

        srl_data = defaultdict(lambda: defaultdict(SRLSentence))
        for topic in list(corpus.topics.values()):
            for doc_id, doc in topic.docs.items():

                for sent_id, sent in doc.sentences.items():
                    srl_sent = SRLSentence(doc_id, sent_id)
                    srl = predictor.predict_tokenized([t.token for t in sent.tokens])

                    for verb in srl["verbs"]:
                        srl_verb_obj = SRLVerb()
                        srl_verb_obj.verb = SRLArg(verb["verb"], [srl["words"].index(verb["verb"])])

                        for tag_id, tag in enumerate(verb["tags"]):
                            for tag_type in ["ARG0", "ARG1", "TMP", "LOC", "NEG"]:
                                EECDCRIdentifier.check_tag(tag, tag_id, srl_verb_obj, tag_type, srl["words"])
                        srl_sent.add_srl_vrb(srl_verb_obj)

                    srl_data[doc_id][sent_id] = srl_sent
        return srl_data
