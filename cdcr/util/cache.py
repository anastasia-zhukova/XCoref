import os
import pickle
import shutil
from pathlib import Path
from typing import Optional, List, Union
import datetime
import logging
from cdcr.structures import DocumentSet

DEFAULT_DIRECTORY = os.path.realpath(os.path.join(os.path.dirname(__file__),  "..", "..", "data", "cache"))


class Cache:
    """Caching utility of newsalyze.

    Arguments:
        directory: Directory where the caches should be saved and loaded from.
    """

    def __init__(self, directory: Optional[str] = DEFAULT_DIRECTORY):
        self.directory = directory
        """str: Directory where the caches should be saved and loaded from."""

    def save(self, document_set: DocumentSet) -> str:
        """Save a document set in the cache.

        Arguments:
            document_set: The document set to cache.

        Returns:
            Path to the cache file.
        """

        filename = "_".join([self._time, document_set.topic, document_set.processing_information.last_module])
        directory = os.path.join(self.directory, self._date)
        path = os.path.join(directory, filename + ".pickle")
        Path(directory).mkdir(exist_ok=True)
        pickle.dump(document_set, open(path, "xb"))
        return filename

    def list(self) -> List[str]:
        """List of all available cache paths.

        Returns:
            All available cache paths.
        """
        return sorted([os.path.join(dir_[0], file)
                       for dir_ in os.walk(self.directory)
                       for file in dir_[2]
                       if not file.startswith(".")])

    def list_today(self) -> List[str]:
        """List today's created cache paths.

        Returns:
            Today's created cache paths.
        """
        dir_ = os.path.join(self.directory, self._date)
        return sorted([os.path.join(dir_[0], file)
                       for dir_ in os.walk(dir_)
                       for file in dir_[2]
                       if not file.startswith(".")])

    def load_interactive(self,
                         load_cache: Optional[bool] = None,
                         today: Optional[bool] = None,
                         cache_number: Optional[int] = None) -> Union[DocumentSet, None]:
        """Load a cache file interactively using python inputs.

        Arguments:
            load_cache: Whether to load any cache at all.
            today: Whether to only show caches from today to load.
            cache_number: The display index of the cache file.

        Returns:
            The loaded cache or none (if no cache should be loaded at all or none exists).
        """
        if load_cache is False:
            return None
        while load_cache is None:
            try:
                inp = input("Do you want to load a cached file? [Y/n] ")[0].lower()
                if inp not in ["y", "n"]:
                    print("Please answer with 'y', 'n' or nothing (for default: 'y').")
                else:
                    if inp == "y":
                        load_cache = True
                    else:
                        return None
            except IndexError:
                load_cache = True

        while today is None:
            try:
                inp = input("Do you want to only list caches which were created today? [Y/n] ")[0].lower()
                if inp not in ["y", "n"]:
                    print("Please answer with 'y', 'n' or nothing (for default: 'y').")
                else:
                    today = True if inp == "y" else False
            except IndexError:
                today = True

        available = self.list_today() if today else self.list()

        if cache_number:
            return self.load(available[cache_number])

        if not len(available):
            print("No cached files found. Continuing without loading from cache.")
            return None

        for i, file in enumerate(available):
            print(f"[{i}]: {file}")

        while True:
            try:
                inp = input(f"The cache file with which number do you want to load? [0-{len(available)-1}] ")
                inp = int(inp) if len(inp) else -1
                return self.load(available[inp])
            except (IndexError, ValueError) as e:
                logging.warning(e)
                print(f"Please answer with a number between 0 and {len(available)-1} or nothing "
                      f"(for latest: {len(available)-1}).")

    @staticmethod
    def load(path: str) -> Union[DocumentSet, None]:
        """Load a document set from cache.

        Args:
            path: Location of the cache or a topic name.

        Returns:
            The cached document set.
        """
        try:
            return pickle.load(open(path, "rb"))
        except FileNotFoundError:
            return None

    def purge(self) -> None:
        """Remove all available caches."""
        shutil.rmtree(self.directory)

    @property
    def _date(self):
        """str: Today's date (for consistency across the class)."""
        return datetime.datetime.now().strftime("%Y-%m-%d")

    @property
    def _time(self):
        """str: Current time (for consistency across the class)."""
        return datetime.datetime.now().strftime("%H-%M-%S")
