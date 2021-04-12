from cdcr.structures.one_to_n_set import OneToNSet
from typing import Iterable, Any
from cdcr.structures.entity import Entity
from pandas import DataFrame


class EntitySet(OneToNSet):
    CHILDRENS_PARENT_ATTRIBUTE = "entity_set"

    def __init__(self, identification_method: str, topic: str, items: Iterable = [], evaluation_details: DataFrame = None,
                 force: bool = False):
        """
        A set of identified entities.

        Args:
            items: Enities to be contained in the set.
            force: Whether the entities should be removed from another EntitySet if already contained in another one.
        """
        super().__init__(items, force)

        self.identification_method = identification_method
        self.topic = topic
        self.evaluation_details = evaluation_details

    def _test_and_parse(self, entity: Any, force: bool = False) -> Entity:
        """
        See ``structures.one_to_n_set.OneToNSet._test_and_parse()``.
        """
        if not issubclass(entity.__class__, Entity):
            raise TypeError("The object passed is not of " + Entity.__name__ + " class. ")
        return super()._test_and_parse(entity, force)
