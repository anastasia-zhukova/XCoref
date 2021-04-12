"""
ConfigLoader module which changes class variables in order to
change the default configuration for those classes.

Assuming having a class (e.g. from a pip module) looking
like this:

.. code-block:: python

    class Test:
        DEFAULT_TIMEOUT = 50

        def connect(timeout=None):
            if not timeout:
                timeout = Test.DEFAULT_TIMEOUT
            some_more_code_here()

The variable ``Test.DEFAULT_TIMEOUT`` can be changed in a
config with:

.. code-block:: ini

    [Test]

    DEFAULT_TIMEOUT = 60

For submodules these can be used as following (assuming
the ``Test``-class is in a submodule called ``sub.module``):

.. code-block:: ini

    # First possibility:
    [sub]
    module.Test.DEFAULT_TIMEOUT = 60
    # Second possibility:
    [sub.module]
    Test.DEFAULT_TIMEOUT = 60
    # Third possibility:
    [sub.module.Test]
    DEFAULT_TIMEOUT = 60

To force a type on a variable use:

.. code-block:: ini

    [module.and.class]
    INT = 123
    INT__type__ = int
    FLOAT = 1.23
    FLOAT__type__ = float
    COMPLEX = 1.23+3j
    COMPLEX__type__ = complex
    STRING = somecontent
    STRING__type__ = str
    BOOL = True
    BOOL__type__ = bool
    NONE__type__ = NoneType

Note: Only int, float, complex, str, bool and none are supported.
    If none is set it first looks up the old type and if
    it is not parseable to that type or that type is None it takes
    the best guess (trying to parse in the following order):
    ``bool -> int -> float -> complex -> str``
"""
# from __future__ import annotations
import ast
import os, logging, json
from configparser import ConfigParser, NoOptionError
from importlib import import_module
from contextlib import suppress
from typing import Union, Optional

from cdcr.logger import LOGGER

# wordvectors
WORD2VEC_WE = "word2vec"
GLOVE_WE = "glove"
FASTTEXT_WE = "fasttext"
ELMO_WE = "elmo"

ROOT_DIR = os.path.relpath(os.path.join(os.path.dirname(os.path.relpath(__file__)), ".."))
# ROOT_DIR = os.path.relpath(os.getcwd())
ROOT_ABS_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
# ROOT_ABS_DIR = os.getcwd()
CORENLP_SERVER = 'http://localhost:9000'
DATA_PATH = os.path.join(ROOT_DIR, "data")
ORIGINAL_DATA_PATH = os.path.join(DATA_PATH, "original")
SAVED_DATA_PATH = os.path.join(DATA_PATH, "saved")
LINK_LIST_PATH = os.path.join(ROOT_DIR, "data", "links")
COMP_TABLES_PATH = os.path.join(ROOT_DIR, "resources", "comparison_tables_2")
WORDVECTOR_MODELS_PATH = os.path.join(ROOT_DIR, "resources", "word_vector_models")
WORD2VEC_MODEL = os.path.join("GoogleNews-vectors-negative300", "GoogleNews-vectors-negative300.bin")
WORD2VEC_MODEL_PATH = os.path.join(WORDVECTOR_MODELS_PATH, WORD2VEC_MODEL)

# WORD2VEC_MAGN_PATH = os.path.join(WORDVECTOR_MODELS_PATH, "word2vec-GoogleNews-magnitude", "GoogleNews-vectors-negative300.magnitude")
WORD2VEC_MAGN_PATH = "D:\\GoogleNews-vectors-negative300.magnitude"
FASTTEXT_MAGN_PATH = os.path.join(WORDVECTOR_MODELS_PATH, "fasttext-CommonCrawl-magnitude", "crawl-300d-2M.magnitude")
# FASTTEXT_MAGN_PATH = "D:\\crawl-300d-2M.magnitude"
ELMO_MAGN_PATH = os.path.join(WORDVECTOR_MODELS_PATH, "elmo-1mlnwords-magnitude", "elmo_2x2048_256_2048cnn_1xhighway_weights.magnitude")
# GLOVE_MAGN_PATH = "D:\\GloVe\\glove.840B.300d.magnitude"
GLOVE_MAGN_PATH = os.path.join(WORDVECTOR_MODELS_PATH, "GloVe-CommonCrawl-magnitude", "glove.840B.300d.magnitude")

RESOURCES_PATH = os.path.join(ROOT_DIR, "resources")
RESOURCES_ABS_PATH = os.path.join(ROOT_ABS_DIR, "resources")
TMP_PATH = os.path.join(RESOURCES_PATH, "tmp")
USER_CONFIG_SETTINGS = os.path.join(RESOURCES_PATH, "user_run_config")
ANNOTATION_PATH = os.path.join(RESOURCES_PATH, "annotations")
EVALUATION_PATH = os.path.join(RESOURCES_PATH, "evaluation_results")
NEWALYZE_PATH = os.path.relpath(os.path.join(os.path.dirname(os.path.relpath(__file__))))
EECDCR_PATH = os.path.join(NEWALYZE_PATH, "entities", "eecdcr")
DEFAULT_RUN_CONFIG = "default_2"  # default or default_2, see configs in resources/user_run_config

REQUIRED_FIELDS = "text,description,title"

COMP_TABLES_NAME = "comparison_tables"
CONFIG_PARAMS_ENTITY_IDENTIFICATION_NAME = "config_params_entity_identification"


def bool_from_str(value: str):
    try:
        value = ast.literal_eval(value.capitalize())
        if value is True or value is False:
            return value
    except SyntaxError:
        raise ValueError("The value is neither True nor False.")
    raise ValueError("The value is neither True nor False.")


class ConfigLoader:
    DEFAULT_FILE = '../conf.ini'
    """
    str: Path of the config file (relative to this file or absolute) when no other path is passed to the constructor.
    """
    MODULE_SEPARATOR = "."
    """str: Pythons module seperator character."""

    TYPE_ENDING = "__type__"
    """
    str: The ending to set a variable type in the config.

    Example:

    Assuming ``TYPE_ENDING = "__type__"`` a type is declared by using the variable name and adding ``TYPE_ENDING``:

    .. code-block:: cfg

        variable = 123
        variable__type__ = str
    """

    PARSE_ORDER = [bool_from_str, int, float, complex, str]
    """list of type: Order in which a value is tried to be parsed if no type is set with the variable."""

    NONE_TYPE = "NoneType"
    """
    str: Type to set in the config in order to declare a variable as ``None``.

    Example:

    Assuming ``TYPE_ENDING = "NoneType"`` (and ``TYPE_ENDING = "__type__"``) a variable must be set in the config
    as following:

    .. code-block:: cfg

        variable__type__ = NoneType

    The variable itself must not be set.
    """

    TYPE_FROM_STRING = {
        "int": int,
        "float": float,
        "complex": complex,
        "str": str,
        "bool": bool,
    }
    """(dict of str: type): Types to get from a string representation of the type."""

    PARSE_TYPE = {
            int: int,
            float: float,
            complex: complex,
            str: str,
            bool: bool_from_str,
        }
    """(dict of type: Callable[str, Any]): Function to apply in order to try and parse a string to this type."""

    FORCE_DISABLED_LOGS = False
    """boolean: Whether to not log (under any circumstances)."""

    def __init__(self, path: Optional[str] = None):
        """
        Create a config object for parsing and applying a config file.

        Args:
            path: Absolute path to the config file to use.
                Uses ``ConfigLoader.DEFAULT_FILE`` (relative to this file or absolute) if ``None``.
        """

        self._logger = LOGGER
        """logging.Logger: Logger for this module."""

        if not path:
            path = ConfigLoader.DEFAULT_FILE
            if not os.path.isabs(path):
                path = os.path.join(os.path.dirname(__file__), ConfigLoader.DEFAULT_FILE)
                path = os.path.abspath(path)

        self.path = path
        """str: Absolute path to the configuration file."""

        if not ConfigLoader.FORCE_DISABLED_LOGS:
            self._logger.info(f"Setting config path to {path}.")

        self._parser = ConfigParser()
        """configparser.ConfigParser: Parser used for parsing the config file."""

        # Preserve case of variables
        self._parser.optionxform = str

        self._changed = set()
        """set: Variables of classes that have been changed."""

        self._skipped = set()
        """
        set: Variables of classes that have been skipped (because the variable was not found
        or the declaration in the config is faulty.
        """

    def read(self) -> None:
        """Read the configuration file."""
        self._parser.read(self.path)

    def apply(self) -> None:
        """Apply the configurations settings to its classes."""
        for section in self._parser.sections():
            for option in self._parser.options(section):
                # Check if the variable should be none, otherwise skip if it is a type-variable.
                set_none = False
                if option.endswith(ConfigLoader.TYPE_ENDING):
                    if self._is_none_type(section, option):
                        option = option[:-len(ConfigLoader.TYPE_ENDING)]
                        set_none = True
                    else:
                        continue

                # Get the module from section name
                module = None
                class_ = ""
                try:
                    module = import_module(section)
                except ModuleNotFoundError:
                    try:
                        module, class_ = section.rsplit(ConfigLoader.MODULE_SEPARATOR, 1)
                        module = import_module(module)
                    except (ModuleNotFoundError, ValueError):
                        self._mark_skipped(section, option)

                full_option = option
                if len(class_):
                    full_option = class_ + ConfigLoader.MODULE_SEPARATOR + option
                sub_modules = full_option.split(ConfigLoader.MODULE_SEPARATOR)
                variable = sub_modules.pop()

                try:
                    # Get any submodules from the option names
                    while len(sub_modules):
                        module = getattr(module, sub_modules.pop(0))
                    if set_none:
                        # Set the variable to none (as wished by the name of the variable)
                        setattr(module, variable, None)
                    else:
                        # Set the variable
                        old = getattr(module, variable)
                        value = self.__get_value(section, option, type(old))
                        setattr(module, variable, value)
                    self._mark_applied(section, option)
                except (AttributeError, TypeError, NameError):
                    self._mark_skipped(section, option)

    def log(self) -> None:
        """Log the changed and skipped variables (if ``ConfigLoader.FORCE_DISABLED_LOGS is not False``)."""
        if ConfigLoader.FORCE_DISABLED_LOGS:
            return
        if len(self._changed):
            changed = ", ".join(self._changed)
            self._logger.info(f"Changing variables of following modules: {changed}.")
        if len(self._skipped):
            skipped = ", ".join(self._skipped)
            self._logger.warning(f"Not changing variables because of erroneous settings: {skipped}.")

    def _mark_applied(self, section: str, option: str) -> None:
        """
        Mark a variable as applied (and remove the marker from skipped, if existent).

        Args:
            section: Section in which the variable (option) is set.
            option: Option (variable) name.
        """
        var = ConfigLoader.MODULE_SEPARATOR.join([section, option])
        with suppress(KeyError):
            self._skipped.remove(var)
        self._changed.add(var)

    def _mark_skipped(self, section: str, option: str) -> None:
        """
        Mark a variable as skipped.

        Args:
            section: Section in which the variable (option) is set.
            option: Option (variable) name.
        """
        var = ConfigLoader.MODULE_SEPARATOR.join([section, option])
        # Not removing the applied-marker, as the applied change is still in use.
        self._skipped.add(var)

    def __get_value(self, section: str, option: str, old_type: type) -> Union[bool, int, float, complex, str, None]:
        """
        Get the value of an option (variable) with the best fitting type:

        1. Type set in the config (if existent and fitting).
        2. Type of the old value (if fitting).
        3. Best fitting type (trying in the order of ``ConfigLoader.PARSER_ORDER``).

        Args:
            section: Section in which the variable (option) is set.
            option: Option (variable) name.
            old_type: Type of the old variables value.

        Returns:
            The value with the correct type (one out of ``ConfigLoader.PARSER_ORDER`` or ``NoneType``).
        """
        value = self._parser.get(section, option)
        with suppress(NoOptionError, ValueError):
            # Check if VARIABLE__type__ is set
            type_string = self._parser.get(section, option + self.TYPE_ENDING)
            type_ = self.TYPE_FROM_STRING[type_string]
            return self.PARSE_TYPE[type_](value)

        with suppress(ValueError, KeyError):
            # Check if it is the same type as the old variable
            return self.PARSE_TYPE[old_type](value)

        for type_ in self.PARSE_ORDER:
            with suppress(ValueError):
                # Take a best guess. Last (str) always works.
                return type_(value)

    def _is_none_type(self, section: str, option: str) -> bool:
        """
        Check if a variable shall be None.

        Args:
            section: Section in which the variable (option) is set.
            option: Option (variable) name.

        Returns:
            Whether the variable shall be None.
        """
        return self._parser.get(section, option) == ConfigLoader.NONE_TYPE

    @staticmethod
    def load_and_apply(path: Optional[str] = None, logging: bool = False): # -> ConfigLoader
        """
        Load and apply a config file.

        Args:
            path: Absolute path to the config file to use.
                Uses ``Config.DEFAULT_FILE`` (relative to this file or absolute) if ``None``.
            logging: Whether to log the skipped and applied variables.
                Can also be later logged manually by calling the ``log()`` method.

        Returns:
            ConfigLoader: Config object assosicated to the config file at the given path or default path.
        """
        config = ConfigLoader(path)
        config.read()
        config.apply()
        if logging:
            config.log()
        return config
