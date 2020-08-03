import h5py
import json
import jsonlines
import logging
import math
import os
import numpy
import re
import sys
from collections import OrderedDict, Counter
from fuel.datasets import H5PYDataset
#from fuel.utils import find_in_data_path
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
from matplotlib import pyplot
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from numpy import save
from numpy import asarray


def normalizeText(text):
    text = text.lower()
    text = re.sub(r'<br />', r' ', text).strip()
    text = re.sub(r'^https?:\/\/.*[\r\n]*', ' L ', text, flags=re.MULTILINE)
    text = re.sub(r'[\~\*\+\^`_#\[\]|]', r' ', text).strip()
    text = re.sub(r'[0-9]+', r' N ', text).strip()
    text = re.sub(r'([/\'\-\.?!\(\)",:;])', r' \1 ', text).strip()
    return text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

conf_file = sys.argv[1] if len(sys.argv) > 1 else None
with open(conf_file) as f:
    locals().update(json.load(f))

with open('list.txt', 'r') as f:
    files = f.read().splitlines()

## Load data and define vocab ##
logger.info('Reading json and jpeg files...')
movies = []

for i, file in enumerate(files):

    with open(file) as f:
        data = json.load(f)
        data['imdb_id'] = file.split('/')[-1].split('.')[0]
        # if 'plot' in data and 'plot outline' in data:
        #    data['plot'].append(data['plot outline'])
        im_file = file.replace('json', 'jpeg')
        if all([k in data for k in ('genres', 'plot')] + [os.path.isfile(im_file)]):
            plot_id = numpy.array([len(p) for p in data['plot']]).argmax()
            data['txt'] = normalizeText(data['plot'][plot_id])
            
            if len(data['plot']) > 0:
                data['img'] = im_file
                movies.append(data)
    logger.info('{0:05d} out of {1:05d}: {2:02.2f}%'.format(
        i, len(files), float(i) / len(files) * 100))

logger.info('done reading files.')


# Define train, dev and test subsets
counts = OrderedDict(
    Counter([g for m in movies for g in m['genres']]).most_common())
target_names = list(counts.keys())[:n_classes]


le = MultiLabelBinarizer()
Y = le.fit_transform([m['genres'] for m in movies])
labels = numpy.nonzero(le.transform([[t] for t in target_names]))[1]


B = numpy.copy(Y)
rng = numpy.random.RandomState(rng_seed)
train_idx, dev_idx, test_idx = [], [], []
for l in labels[::-1]:
    t = B[:, l].nonzero()[0]
    t = rng.permutation(t)
    n_test = int(math.ceil(len(t) * test_size))
    n_dev = int(math.ceil(len(t) * dev_size))
    n_train = len(t) - n_test - n_dev
    test_idx.extend(t[:n_test])
    dev_idx.extend(t[n_test:n_test + n_dev])
    train_idx.extend(t[n_test + n_dev:])
    B[t, :] = 0


indices = numpy.concatenate([train_idx, dev_idx, test_idx])
nsamples = len(indices)
nsamples_train, nsamples_dev, nsamples_test = len(
    train_idx), len(dev_idx), len(test_idx)

train_data = [{'label': [genre for genre in movies[idx]['genres'] if genre in target_names], 'img': movies[idx]['img'], 'text': movies[idx]['txt']} for idx in train_idx]
dev_data = [{'label': [genre for genre in movies[idx]['genres'] if genre in target_names], 'img': movies[idx]['img'], 'text': movies[idx]['txt']} for idx in dev_idx]
test_data = [{'label': [genre for genre in movies[idx]['genres'] if genre in target_names], 'img': movies[idx]['img'], 'text': movies[idx]['txt']} for idx in test_idx]

with jsonlines.open('/001/usuarios/isaac.bribiesca/mmimdb/train.jsonl', 'w') as writer:
    writer.write_all(train_data)

with jsonlines.open('/001/usuarios/isaac.bribiesca/mmimdb/dev.jsonl', 'w') as writer:
    writer.write_all(dev_data)

with jsonlines.open('/001/usuarios/isaac.bribiesca/mmimdb/test.jsonl', 'w') as writer:
    writer.write_all(test_data)