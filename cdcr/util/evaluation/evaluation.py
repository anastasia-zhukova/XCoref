from cdcr.util.cache import Cache
from cdcr.util.evaluation.calculate_scores import EvalScorer
from cdcr.config import EVALUATION_PATH

from datetime import datetime
import os, re
import logging
import progressbar

EVAL_MODULES = ["entities"]

cache = Cache()
document_set_list = []
widgets = [progressbar.FormatLabel("PROGRESS: Reading details of %(value)d-th/%(max_value)d (%(percentage)d %%) cached files (in: %(elapsed)s).")]
bar = progressbar.ProgressBar(widgets=widgets, maxval=len(cache.list())).start()
docset_dict = {}

for i, dir in list(enumerate(cache.list())):
    for module in EVAL_MODULES:
        if module in dir:
        # if module in dir and "ecb" in dir:
        # if module in dir and "ecb" not in dir:
            try:
                docset = Cache.load(dir)
                document_set_list.append(docset)
            except (MemoryError, FileNotFoundError):
                document_set_list.append(Cache.load(dir))

    bar.update(i + 1)
bar.finish()
results = EvalScorer(document_set_list, EVAL_MODULES).run_evaluation()

now = datetime.now()
for mod, res in enumerate(results):
    for i, res_df in enumerate(res):
        export_file_name = now.strftime("%Y-%m-%d_%H-%M") + "_" + EVAL_MODULES[mod] + "_" + str(i)
        res_df.to_csv(os.path.join(EVALUATION_PATH, export_file_name + ".csv"), index=True)

logging.info("Results saved to {}".format(EVALUATION_PATH))
