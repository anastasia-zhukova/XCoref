import cdcr.config as config
from cdcr.util.cache import Cache
from cdcr.config import *

from flask import Flask
from flask import render_template
import os
import logging
import json

from cdcr.structures import DocumentSet

app = Flask(__name__, root_path=ROOT_ABS_DIR)


class VisualizationRunner:
    """
    The module saves to json file a prepared for the visualization dataset, and runs a server with a visualization.
    """

    logger = LOGGER

    def __init__(self, module_name):
        self.module_name = module_name
        self._cache = Cache()

    @classmethod
    def run(cls, document_set: DocumentSet):
        """
        Save the visualization results as json files.

        Args:
            document_set (DocumentSet): A DocSet including the visualization results.

        Returns:
            document_set (DocumentSet): An unchanged DocSet.
        """
        config = document_set.configuration.visualization_config

        cls.logger.info("Saving visualizations into {}.".format(config.data_file_path))
        with open(config.data_file_path, 'w') as save_file:
            json.dump(document_set.visualizations, save_file)
        cls.logger.info("Successfully saved.")

        cls.start_server(config)
        return document_set

    @classmethod
    def start_server(cls, config):
        port = 5000
        app.static_folder = config.static_folder
        app.template_folder = os.path.join(ROOT_ABS_DIR, config.templates_folder)
        cls.logger.info("Open http://localhost:{0}/ for visualization.".format(port))
        app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)


@app.route("/")
def index():
    return render_template("index.html")
