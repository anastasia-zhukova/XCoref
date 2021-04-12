import dbm
import os
import pickle
import shelve
import datetime
from contextlib import suppress
from copy import deepcopy, copy
from typing import Tuple, List, Optional, Dict, Union, Any

from cdcr.structures import DocumentSet

DEFAULT_DIRECTORY = os.path.realpath(os.path.join(os.path.dirname(__file__),  "..", "..", "data", "exported"))
DATE_FORMAT = "%Y-%m-%d"
FILE_ENDING = ".pickle"  # must start with a .


class DocumentExporter:

    def __init__(self, directory: str = DEFAULT_DIRECTORY):
        self.directory = directory
        self._cache = {}

    def list(self) -> List[Tuple[datetime.datetime, str]]:
        """
        List all files available with their corresponding date.

        Returns:
            All available exported files and their corresponding date.
        """
        return sorted([(datetime.datetime.strptime(file.split(".", 1)[0], DATE_FORMAT), file)
                       for file in os.listdir(self.directory)
                       if not file.startswith('.')])

    def list_between(self, after: datetime.datetime = None, before: datetime.datetime = None)\
            -> List[Tuple[datetime.datetime, str]]:
        """
        List all files which correspond to a date after `after` and before `before`.

        Args:
            after: Date after which the files should correspond to.
            before: Date before which the files should correspond to.

        Returns:
            All available exported files and their corresponding date.
        """
        items_after = filter(lambda x: x[0] > after, self.list()) if after else self.list()
        return filter(lambda x: x[0] < before, items_after) if before else items_after

    def export(self, document_set, overwrite: Optional[str] = None):
        """
        Export a document_set.

        Args:
            document_set: The document set to export.
            overwrite: Key to overwrite. If None create a new, unique key.
        """
        filename = document_set.processing_information.last_module_time.strftime(DATE_FORMAT)
        key_n = 1
        key = None
        try:
            data = self.read(filename)
        except FileNotFoundError:
            data = {}
        while (key is None or key in data) and not overwrite:
            key = f"{document_set.topic} ({str(key_n).rjust(3, '0')})"
            key_n += 1
        data[key if not overwrite else overwrite] = document_set
        self.write(data, filename)

    def read(self, filename: Union[str, datetime.datetime]):
        """
        Open and read an existing exported file.

        Args:
            filename: Export-file to open.

        Returns:
            The exported data.
        """
        return pickle.load(open(self._true_filename(filename), "rb"))

    def write(self, data: Any, filename: Union[str, datetime.datetime]):
        """
        Save data in an export file.

        Args:
            data: The data to export.
            filename: File to contain the exported data after export.
        """
        pickle.dump(data, open(self._true_filename(filename), "wb"))

    def _true_filename(self, filename):
        if isinstance(filename, datetime.datetime):
            filename = filename.strftime(DATE_FORMAT)
        if not filename.endswith(FILE_ENDING):
            filename += FILE_ENDING
        return os.path.join(self.directory, filename)

    def copy(self, filename: Union[str, datetime.datetime], ignore_nonexistent: bool = True)\
            -> Dict[str, DocumentSet]:
        """
        Load the (deep) copy of an exported file and cache it.

        Args:
            filename: Filename of the exported file to open and
                return a copy from.
            ignore_nonexistent: Whether to ignore if a file does
                not exist (return empty dict) or raise an dbm.error.

        Returns:
            A copy from the exported file.
        """
        return deepcopy(self.get(filename, ignore_nonexistent))

    def get(self, filename: Union[str, datetime.datetime], ignore_nonexistent: bool = True)\
            -> Dict[str, DocumentSet]:
        """
        Load the exported file, if available from cache.
        RETURNS THE ORIGINAL OBJECT IN CACHE, so when the returned object gets modified, the cache gets
        modified as well. The exported file (on disk) remains untouched.

        Args:
            filename: Filename of the exported file to open and
                return a copy from.
            ignore_nonexistent: Whether to ignore if a file does
                not exist (return empty dict) or raise an dbm.error.

        Returns:
            The object from the exported file.
        """
        with suppress(KeyError):
            return self._cache[filename]

        try:
            data = dict(self.read(filename))
        except FileNotFoundError:
            if ignore_nonexistent:
                return {}
            raise
        self._cache[filename] = data
        return data

    @property
    def _date(self):
        """str: Today's date (for consistency across the class)."""
        return datetime.datetime.now().strftime(DATE_FORMAT)
