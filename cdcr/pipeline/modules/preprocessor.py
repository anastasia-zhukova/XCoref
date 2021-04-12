from cdcr.structures import DocumentSet
import progressbar


class Preprocessor:
    """
    A Preprocessor class performs preprocessing, executing different tasks depending on the given options.
    Corenlp is an obligatory preprocessing task, other libraries and parameters are optional.
    To add more preprocessing steps one needs to add them to _preprocessor_modules.
    """

    def __init__(self, module_name):
        self.module_name = module_name

    @staticmethod
    def run(document_set: DocumentSet) -> DocumentSet:
        """
        Apply preprocessing steps to the documents in the given DocSet.

        Args:
            document_set (DocumentSet): A DocSet including the documents to apply preprocessing steps.

        Returns:
            document_set (DocumentSet): A DocSet including the preprocessed documents.
        """

        # apply core nlp preprocessing to each document in the docset
        widgets = [progressbar.FormatLabel("PROGRESS: Preprocessing %(value)d/%(max_value)d (%(percentage)d %%) "
                                           "document (in: %(elapsed)s). ")]
        bar = progressbar.ProgressBar(widgets=widgets, maxval=len(document_set), redirect_stdout=True).start()

        for i, document in enumerate(document_set):
            document.apply_nlp()
            bar.update(i)
        bar.finish()
        return document_set
