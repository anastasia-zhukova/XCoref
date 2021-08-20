"""
Start Newsalyze.
"""

import sys
import os

file_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{file_dir}/newstsc")
os.chdir(os.path.join(file_dir))

from cdcr.pipeline.modules import NewsPleaseReader, Preprocessor, CandidateExtractor, EntityIdentifier
from cdcr.pipeline.modules.exporter import document_export

from cdcr.config import ConfigLoader
import cdcr.logger as logging
from cdcr.pipeline import Pipeline

if __name__ == "__main__":
    conf = ConfigLoader.load_and_apply()
    logging.setup()
    conf.log()

    Pipeline({
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
    }).run()
