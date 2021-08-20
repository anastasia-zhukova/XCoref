"""
Check and (if necessary) install the correct versions of the word vector model and also of corenlp.
Check versions of LIWC and python and hint possible problems to the user.
This script is to be automatically executed before execution of the pipeline.
"""


import sys
import os
import requests
import zipfile
import io
import nltk
import gdown
import gzip
import shutil
import spacy
from cdcr.config import GLOVE_MAGN_PATH, LOGGER
CORE_NLP = 'http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip'
GLOVE_MAGN = "http://magnitude.plasticity.ai/glove/medium/glove.840B.300d.magnitude"

def check_errors():
    old_cwd = os.getcwd()
    _cwd_to_here()

    errors = {
        "python": sys.version_info < (3, 6) or sys.version_info >= (3, 8),
        "corenlp": not os.path.isdir('./resources/corenlp/stanford-corenlp-full-2018-10-05'),
        "evaluation": not os.path.isdir('./resources/evaluation_results'),
        "wordvectors": not os.path.isfile(GLOVE_MAGN_PATH)
    }

    os.chdir(old_cwd)
    return errors


def init():
    LOGGER.info('Running... (this takes a while, do not abort)')
    old_cwd = os.getcwd()
    _cwd_to_here()

    errors = check_errors()
    if errors["python"]:
        raise SystemExit('Sorry, this code needs Python 3.6. Please refer to the readme to install.')

    if errors["corenlp"]:
        LOGGER.info(f"Downloading {CORE_NLP}...")
        r = requests.get(CORE_NLP)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall('./resources/corenlp/')

    if errors["wordvectors"]:
        LOGGER.info(f"Downloading {GLOVE_MAGN}...")
        gdown.download(GLOVE_MAGN, GLOVE_MAGN_PATH, quiet=False)

    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    spacy.cli.download('en_core_web_sm')
    os.chdir(old_cwd)
    LOGGER.info('Setup completed')


def _cwd_to_here():
    file_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(file_dir)


if __name__ == '__main__':
    init()
