from cdcr.entities.sieve_based.entity_sieves import EntitySieves


class EntityXCoref(EntitySieves):
    """
    An entity class required for SIEVE_BASED 2.0 execution.
    """

    def __init__(self, document_set, ent_preprocessor, members, name, wikipage=None, core_mentions=None):

        super().__init__(document_set, ent_preprocessor, members, name, wikipage, core_mentions)

    def additional_param_update(self, **kwargs):
        self._labeling_extraction()
