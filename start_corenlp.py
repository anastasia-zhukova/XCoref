"""
Start the corenlp server.
"""

import subprocess
import os

def init():
    filedir = os.path.dirname(os.path.realpath(__file__))
    command = 'java -mx6g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 300000 -preload "tokenize,ssplit,pos,lemma,ner,parse,depparse,coref"'
    wd = os.path.join(filedir, "resources/corenlp/stanford-corenlp-full-2018-10-05")
    os.chdir(wd)
    subprocess.run(command, shell=True)


if __name__ == '__main__':
    init()
