"""
Compile the list of examples from the Who's Waldo dataset such that:
- the caption has a verb phrase
- AND there are no fewer than 2 identity clusters
- AND there are no fewer than 2 people detections
"""

import sys
sys.path.append('.')
import json
import os
import nltk.data
from flair.models import SequenceTagger
from flair.data import Sentence


root_dir = '.'
meta_dir = os.path.join(root_dir, 'dataset_meta')
whos_waldo_dir = os.path.join(root_dir, 'whos_waldo')

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
tagger = SequenceTagger.load('pos-fast')
chunker = SequenceTagger.load('chunk-fast')

def read_json(json_path):
    f = json.load(open(json_path, 'r', encoding='utf-8'))
    return f

interactive_imgs = []

for i in range(271747):
    folder = '%06d' % i
    folder_path = os.path.join(whos_waldo_dir, folder)
    corefs = read_json(os.path.join(folder_path, 'coreferences.json'))
    detections = read_json(os.path.join(folder_path, 'detections.json'))
    with open(os.path.join(folder_path, 'caption.txt'), 'r') as f:
        caption = f.readlines()[0].strip('\n')
    if len(detections) < 2: continue
    if len(corefs) < 2: continue
    has_verb = False

    sentences = sent_detector.tokenize(caption)
    sentences = [Sentence(s, use_tokenizer=True) for s in sentences]
    tagger.predict(sentences)
    chunker.predict(sentences)

    for s in sentences:
        VERB_spans = list(filter(lambda sp: sp.tag[:2] == 'VB', s.get_spans('pos')))
        if len(VERB_spans) > 0:
            has_verb = True
            break
    if not has_verb: continue
    interactive_imgs.append(folder)

with open(os.path.join(meta_dir, 'interactive_img_ids.txt'), 'w') as fp:
    for item in interactive_imgs:
        fp.write("%s\n" % item)