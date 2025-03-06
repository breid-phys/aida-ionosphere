import configparser
from pathlib import Path
import traceback

from .logger import AIDAlogger


logger = AIDAlogger(__name__)


def api_config(filename):
    """Read a AIDA input file (in .ini format)

    This function reads a (well-formatted) AIDA input file and adds all the
    parameters to a dictionary which is carried around and used by the model

    Parameters
    ==========
    filename : string
        Full path for the AIDA input file (usually AIDA.inp)


    Returns
    =======
    inputs : dictionary
        Dictionary containing the AIDA input paramters


    .. todo:: None

    |"""
    # logger = logging.getLogger('main')

    if not isinstance(filename, Path):
        filename = Path(filename)

    # Check filename exists
    if not filename.exists():
        raise ValueError(f"ERROR: file {filename.expanduser()} not found")

    config = configparser.ConfigParser(delimiters=(";", "="), strict=True)

    try:
        config.read(filename.expanduser())
    except Exception:
        logger.error(traceback.format_exc())

    config_struct = {}
    for section in config.sections():
        config_struct[section] = {}

        for option in config.options(section):
            value = config.get(section, option)

            if value.lower() == "none":
                value = None

            config_struct[section][option] = value

    return config_struct
