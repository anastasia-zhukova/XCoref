import json
import os


class Params:
    """
    A general class to represent parameters.
    """
    def __init__(self, iterable: dict, **kwargs):
        """

        Args:
            iterable: a dictionary with parameters and values.
            **kwargs: additional parameters.
        """
        iterable.update(kwargs)
        for k, v in iterable.items():
            setattr(self, k, v)

    def update(self, iterable: dict):
        """
        Update an object with more attributes.

        Args:
            iterable: A dictionary with new parameters.

        Returns:

        """
        for k,v in iterable.items():
            setattr(self, k, v)


class ParamsHelperClass:
    """
    A general helper class for parameters required to execute an module or an approach inside a module. A subclass
    needs to implement generating default parameters, saving them into a config json file, and load from a confing
    json file.

    Each config file needs to be saved into ./resources/user_run_config/<topic_name>/<module_name>/<config_folder>/,
    where
        - topic_name is a name of a dataset on which you run the config now
        - module_name is a name of the module for which you create a config subclass
        - config_folder contains all the config files and follow the naming convention <method_name>_<datetime>_<other>
    """

    def get_default_params(self, def_folder_path: str = None):
        """
        Returns default parameters.

        Args:
            def_folder_path: A path from which some default confid value can be read.

        Returns:
            Itself with generated parameters.

        """
        return self

    def save_config(self, config_path: str):
        """

        Args:
            config_path:

        Returns:

        """
        raise NotImplementedError

    def read_config(self, config_path: str):
        """
        Reads saved config from a given folder and updates the class attributes.

        Args:
            config_path: A folder with saved config.
        """
        raise NotImplementedError
