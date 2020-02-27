import json
import logging
import pandas as pd


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

logger = logging.getLogger(__name__)

CATEGORIES = ["oReact", "oEffect", "oWant", "xAttr", "xEffect", "xIntent", "xNeed", "xReact", "xWant"]


def get_atomic_categories():
    """
    Return the names of ATOMIC categories
    """
    return CATEGORIES


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

