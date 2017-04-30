import argparse
from collections import namedtuple
import itertools
import os
import re

import jinja2
import ruamel.yaml

FIRST_CAP_RE = re.compile('(.)([A-Z][a-z]+)')
ALL_CAP_RE = re.compile('([a-z0-9])([A-Z])')


def camel_to_snake_case(name):
    """ Convert CamelCase op names to snake case """
    s1 = FIRST_CAP_RE.sub(r'\1_\2', name)
    return ALL_CAP_RE.sub(r'\1_\2', s1).lower()


def header_guard(library, header_name):
    """ Derive a C++ header guard from the library and header name """
    guard_str = header_name.replace('.', '_')
    return ''.join([library, '_', guard_str]).upper()


def strip_and_split(s, sep):
    """ Split s on sep and strip the resulting elements """
    return (c.strip() for c in s.split(sep))


def parse_args(args):
    """ Parse and return arguments """
    default_conf_path = os.path.join(
        os.path.dirname(__file__), 'conf', 'default.yml')

    parser = argparse.ArgumentParser("Tensorflow Custom Operator Generator")
    parser.add_argument('config', help="Configuration File", default=default_conf_path)
    return parser.parse_args(args)


def load_config(config_file):
    """ Load the configuration file """
    with open(config_file, "r") as f:
        return ruamel.yaml.load(f, ruamel.yaml.RoundTripLoader)


def jinja_env_factory(template_path):
    """ Creates a jinja environment, loading templates from template_path """
    jinja_loader = jinja2.FileSystemLoader(template_path)
    jinja_env = jinja2.Environment(loader=jinja_loader,
                                   trim_blocks=False, lstrip_blocks=False)

    class LastJoiner(object):
        """
        Opposite of jinja2.utils.Joiner.
        Won't emit separator on the last iteration.
        """

        def __init__(self, sep=u', '):
            self.sep = sep

        def __call__(self, loop_last):
            if loop_last:
                return u''

            return self.sep

    jinja_env.globals['last_joiner'] = LastJoiner

    # Create a filter for formatting a list
    jinja_env.filters['format_list'] = lambda l, p: [p % s for s in l]

    return jinja_env

# Create types for Inputs/Outputs
InOut = namedtuple("InOut", ["name", "type",
                             "tf_type", "np_type", "shape"])
# Create types for Typed Attributes
Attr = namedtuple("Attr", ["original", "name", "types",
                           "tf_types", "np_types", "default"])


def parse_inout(s, shape):
    """ Parse .Input() and .Output() directives """
    var, type_ = tuple(c.strip() for c in s.split(":"))

    if "*" in type_:
        raise ValueError("Failed to parse '{}'. "
                         "List lengths are not yet supported".format(s))

    from tensorflow.python.framework.dtypes import (
        _STRING_TO_TF,
        _TYPE_TO_STRING,
        _TF_TO_NP)

    TF_TYPES = _TYPE_TO_STRING.values()
    tf_type = "tensorflow::" + type_ if type_ in TF_TYPES else type_
    np_type = ("np." + _TF_TO_NP[_STRING_TO_TF[type_]].__name__
               if type_ in _STRING_TO_TF else type_)

    # Set a default shape for variable if None exists
    shape = (1024, ) if shape is None else shape

    return InOut(var, type_, tf_type, np_type, shape)


def parse_attr_type(s):
    """
    Parse type attribute directives. For example
    "FT: {float, double} = DT_FLOAT64"
    """

    # Separate s into "FT" and "{float, double} = DT_FLOAT64"
    var, types = tuple(strip_and_split(s, ":"))

    # Separate types into "{float, double}", "DT_FLOAT64"
    split = types.split("=")
    default = split[1].strip() if len(split) > 1 else None
    types = split[0].strip()

    # Handle the multiple types case
    if types.startswith("{") and types.endswith("}"):
        types = tuple(c.strip() for c in types[1:-1].split(","))
    else:
        types = (types,)

    from tensorflow.python.framework.dtypes import (
        _STRING_TO_TF,
        _TYPE_TO_STRING,
        _TF_TO_NP)

    TF_TYPES = _TYPE_TO_STRING.values()
    tf_types = tuple("tensorflow::" + t if t in TF_TYPES else t for t in types)
    np_types = ["np." + _TF_TO_NP[_STRING_TO_TF[t]].__name__
                if t in _STRING_TO_TF
                else "np." + t for t in types]

    return Attr(s, var, types, tf_types, np_types, default)
