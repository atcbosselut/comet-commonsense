import json
import logging
import pandas as pd
import comet.config as cfg

from distutils.dir_util import mkpath


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

logger = logging.getLogger(__name__)


def get_atomic_categories(eval_mode=True):
    """
    Return the names of ATOMIC categories
    """
    generate_config_files("atomic", eval_mode=eval_mode)
    config_file = "config/atomic/config_0.json"
    config = cfg.read_config(cfg.load_config(config_file))
    opt, _ = cfg.get_parameters(config)
    return opt.data.categories


def load_atomic_data(in_file, categories):
    """
    Load ATOMIC data from the CSV file
    :param in_file: CSV file
    :param categories: list of ATOMIC categories
    :return: list of tuples: (e1 and catgory, e2)
    """
    df = pd.read_csv(in_file, index_col=0)
    df.iloc[:, :len(categories)] = df.iloc[:, :len(categories)].apply(lambda col: col.apply(json.loads))
    df = df.groupby("event").agg({cat: "sum" for cat in categories})

    examples = {row.name.lower().replace('___', '<blank>'): {
        cat: [e2.lower() for e2 in set(row[cat])] for cat in categories if len(row[cat]) > 0}
        for _, row in df.iterrows()}

    return examples


def generate_config_files(type_, name="base", eval_mode=False):
    """
    Generate a configuration file for ATOMIC (copied from the original code).
    :return:
    """
    key = "0"
    with open("config/default.json".format(type_), "r") as f:
        base_config = json.load(f)
    with open("config/{}/default.json".format(type_), "r") as f:
        base_config_2 = json.load(f)
    if eval_mode:
        with open("config/{}/eval_changes.json".format(type_), "r") as f:
            changes_by_machine = json.load(f)
    else:
        with open("config/{}/changes.json".format(type_), "r") as f:
            changes_by_machine = json.load(f)

    base_config.update(base_config_2)

    if name in changes_by_machine:
        changes = changes_by_machine[name]
    else:
        changes = changes_by_machine["base"]

    replace_params(base_config, changes[key])

    mkpath("config/{}".format(type_))

    with open("config/{}/config_{}.json".format(type_, key), "w") as f:
        json.dump(base_config, f, indent=4)


def replace_params(base_config, changes):
    for param, value in changes.items():
        if isinstance(value, dict) and param in base_config:
            replace_params(base_config[param], changes[param])
        else:
            base_config[param] = value


