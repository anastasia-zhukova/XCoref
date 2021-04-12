"""
(Will most likely get deleted as every functionality this adds can be subtituted by @property.)

Embeddings are meant for classes embedding attributes which might reference on another.

To build references across these embeddings it might be necessary to provide callbacks (``on_embedding_finished``) so
those classes know when they can access the references to the embedded data.

When ``on_embedding_finished`` is called it means, that all embedded attributes in the caller-class has finished the
embedding (not all attributes in the callee class).
"""


class EmbeddingDependant(object):
    """
    A base class for a class which is dependant on an embedding.

    Make sure to implement ``on_embedding_finished``.
    """

    def on_embedding_finished(self) -> None:
        """
        Method to be overridden.

        Get's called when the embedding of all callers is finished.
        """
        raise NotImplementedError("on_embedding_finished is required but not implemented.")

    def embedding_finished(self) -> None:
        """
        Method that get's called when embedding is finished.

        Do not override, use ``self.on_embedding_finished()`` instead.
        """
        self.on_embedding_finished()


class EmptyEmbedding(object):
    """
    Base classes for embeddings which does not necessarily need an ``on_embedding_finished``-method to contain anything.
    """

    def on_embedding_finished(self):
        """
        Method to be overridden if wanted.

        Get's called when the embedding of all callers is finished.
        """
        pass


class PassListEmbedding(EmptyEmbedding):
    """
    Class to extend to a class implementing a list.
    """

    def embedding_finished(self):
        """
        Method that get's called when embedding is finished an passes it to each item in the list.

        Do not override, use ``self.on_embedding_finished()`` instead.
        """
        self.on_embedding_finished()

        for item in self:
            item.embedding_finished()


class PassObjectEmbedding(EmptyEmbedding):
    PASS_EMBEDDING_FINISHED = []
    """str: Attributes to be called when embedding is finished. Override it so correct attributes get called."""

    def embedding_finished(self):
        """
        Method that get's called when embedding is finished an passes it to each attribute whichs name occurs in
        ``self.PASS_EMBEDDING_FINISHED``.

        Do not override, use ``self.on_embedding_finished()`` instead.
        """
        self.on_embedding_finished()

        for attribute in self.PASS_EMBEDDING_FINISHED:
            getattr(self, attribute).embedding_finished()
