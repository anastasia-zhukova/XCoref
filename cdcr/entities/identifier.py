from cdcr.structures.entity_set import EntitySet


class Identifier:
    """
    A general class for every entity identifier method.
    """

    def __init__(self, document_set):
        """

        Args:
            document_set: A collection of documents and their properties.
        """
        self.docs = document_set

    def extract_entities(self) -> EntitySet:
        """
        Identifies entities from a collection of documents by resolving candidate phrases.

        Returns:

        """
        raise NotImplementedError
