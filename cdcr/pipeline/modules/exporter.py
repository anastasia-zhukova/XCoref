from cdcr.structures import DocumentSet
from cdcr.util.document_exporter import DocumentExporter


def document_export(document_set: DocumentSet):
    """
    Export a document set with ShelveExporter.

    Arguments:
        document_set: DocumentSet to export.

    Returns:
        The exported document_set.
    """
    exporter = DocumentExporter()
    exporter.export(document_set)
    return document_set
