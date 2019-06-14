import os
import sys

sys.path.append(os.getcwd())

import torch
import src.data.conceptnet as cdata
import src.data.data as data

from utils.utils import DD
import utils.utils as utils
import random
from src.data.utils import TextEncoder
from tqdm import tqdm

opt = DD()
opt.dataset = "conceptnet"
opt.exp = "generation"

opt.data = DD()

# Use relation embeddings rather than
# splitting relations into its component words
# Set to "language" for using component words
# Set to "relation" to use unlearned relation embeddings
opt.data.rel = "language"

# Use 100k training set
opt.data.trainsize = 100

# Use both dev sets (v1 an v2)
opt.data.devversion = "12"

# Maximum token length of e1
opt.data.maxe1 = 10

# Maximum token length of e2
opt.data.maxe2 = 15

relations = [
    'AtLocation', 'CapableOf', 'Causes', 'CausesDesire', 'CreatedBy',
    'DefinedAs', 'DesireOf', 'Desires', 'HasA', 'HasFirstSubevent',
    'HasLastSubevent', 'HasPainCharacter', 'HasPainIntensity',
    'HasPrerequisite', 'HasProperty', 'HasSubevent', 'InheritsFrom',
    'InstanceOf', 'IsA', 'LocatedNear', 'LocationOfAction', 'MadeOf',
    'MotivatedByGoal', 'NotCapableOf', 'NotDesires', 'NotHasA',
    'NotHasProperty', 'NotIsA', 'NotMadeOf', 'PartOf', 'ReceivesAction',
    'RelatedTo', 'SymbolOf', 'UsedFor'
]

special = [data.start_token, data.end_token]
special += ["<{}>".format(relation) for relation in relations]

encoder_path = "model/encoder_bpe_40000.json"
bpe_path = "model/vocab_40000.bpe"

text_encoder = TextEncoder(encoder_path, bpe_path)

for special_token in special:
    text_encoder.decoder[len(text_encoder.encoder)] = special_token
    text_encoder.encoder[special_token] = len(text_encoder.encoder)

data_loader = cdata.GenerationDataLoader(opt)
data_loader.load_data("data/conceptnet/")

data_loader.make_tensors(text_encoder, special, test=False)

opt.data.maxr = data_loader.max_r

save_path = "data/conceptnet/processed/generation"
save_name = os.path.join(save_path, "{}.pickle".format(
    utils.make_name_string(opt.data)))

utils.mkpath(save_path)

print("Data Loader will be saved to {}".format(save_name))

torch.save(data_loader, save_name)
