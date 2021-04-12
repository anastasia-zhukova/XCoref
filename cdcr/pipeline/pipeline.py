# warnings.simplefilter("ignore", UserWarning, append=True)
# warnings.simplefilter("ignore", FutureWarning, append=True)
# warnings.simplefilter("ignore", Per, append=True)

from contextlib import suppress
from copy import copy
from datetime import datetime
from typing import Union, Callable, Dict
import logging
from cdcr.logger import LOGGER
from cdcr.pipeline.modules.exporter import document_export
from cdcr.structures import DocumentSet
from cdcr.util.cache import Cache

import setup
import os

from cdcr.pipeline.modules import NewsPleaseReader
from cdcr.pipeline.modules import Preprocessor
from cdcr.pipeline.modules import CandidateExtractor
from cdcr.pipeline.modules import EntityIdentifier
from cdcr.structures.configuration import Configuration
import cdcr.util.wiki_dict as wiki_dict




class Pipeline:
    """
    Executor of the pipeline. It schedules tasks for execution, and call each module.
    """

    # names of modules must be in one word to be correctly processed
    PIPELINE_MODULES = {
        "reading": {
            "module": NewsPleaseReader.run,
            "caching": False
        },
        "preprocessing": Preprocessor.run,
        "candidates": CandidateExtractor.run,
        "entities": EntityIdentifier.run,
        "export": {
            "module": document_export,
            "caching": False
        }
    }

    def __init__(self, pipeline_modules: Dict[str, Union[Callable, Dict[str, Union[bool, Callable]]]] = None):
        self.pipeline_modules = copy(self.PIPELINE_MODULES)
        if pipeline_modules:
            self.pipeline_modules = pipeline_modules

    def run(self, cache_file: str = None, configuration=None):
        """
        Run the newsalyze pipeline.

        Args:
            cache_file: Cache file to load.
                When False does not load any and start from beginning.
                Otherwise ask interactively using `input`.
            configuration: Configuration to assign to document_set.configuration.

        Returns:
            A parsed set of documents containing extensive information
            about entities which appear in one or multiple of the documents.
        """
        if any(setup.check_errors().values()):
            raise SystemExit(f"It seems like the system was not set up "
                             f"correctly. Run `python3 setup.py`. (Errors: {setup.check_errors()})")

        cache = Cache()

        if cache_file is None:
            document_set = cache.load_interactive()
        elif type(cache_file) is str:
            document_set = cache.load(cache_file)
        else:
            document_set = cache_file

        # initialize word2vec model
        # keep_word2vec.init()
        # init wiki sictionary
        wiki_dict.init()

        start_module_index = 0
        if document_set:
            try:
                module_names = list(self.pipeline_modules)
                start_module_index = module_names.index(document_set.processing_information.step_for_restoring)+1
                document_set.processing_information.module_names = module_names
                if configuration is not None:
                    if len(configuration):
                        document_set.configuration = Configuration(document_set, configuration)
                else:
                    document_set.configuration = Configuration(document_set, configuration)
            except ValueError:
                LOGGER.error(
                    "Unknown save location. The module after which the "
                    "selected cache was saved is not in use anymore. "
                    "Thus the restoring point is unknown.")
                raise

        # Execute the pipeline (only partially when loaded from cache
        # as start_module_index has been adjusted).
        for name, module in list(self.pipeline_modules.items())[start_module_index:]:
            LOGGER.info(f"Running module {name.upper()}.")
            caching = True
            if not callable(module):
                with suppress(KeyError):
                    caching = module["caching"]
                module = module["module"]
            if document_set is None: # and cache_file is not None:
                # reading module to create a document_set from a provided path
                document_set = module(document_set, cache_file, configuration, list(self.pipeline_modules))
            else:
                document_set = module(document_set)
            document_set.processing_information.last_module = name
            document_set.processing_information.last_module_time = datetime.now()
            document_set.processing_information.step_for_restoring = name
            if caching:
                cache.save(document_set)
            logging.info("\n")

        LOGGER.info("Pipeline execution is over.")
        return document_set
