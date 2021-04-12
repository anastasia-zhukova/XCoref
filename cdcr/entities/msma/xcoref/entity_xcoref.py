from cdcr.entities.msma.entity_msma import EntityMSMA



MESSAGES = {
    "no_repr":"No representative words were counted before. Run representative calculation after entity "
              "initialization.",
    "no_word_dict": "Word dictionary is empty. An entity can't be initialized."
}


class EntityXCoref(EntityMSMA):
    """
    An entity class required for MSMA 2.0 execution.
    """

    def __init__(self, document_set, ent_preprocessor, members, name, wikipage=None, core_mentions=None):

        super().__init__(document_set, ent_preprocessor, members, name, wikipage, core_mentions)

    def additional_param_update(self, **kwargs):
        self._labeling_extraction()
