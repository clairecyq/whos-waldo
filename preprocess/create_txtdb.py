"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Process annotations into LMDB
"""
import argparse
import json
import os, io, shutil, sys

sys.path.append('.')
from os.path import exists
from cytoolz import curry
from pytorch_pretrained_bert import BertTokenizer
from data.data import open_lmdb

annotation_dir = './storage/annotations'
txt_db_dir = './storage/txt_db'


def tokenize(tokenizer, text):
    ids = []
    for word in text.strip().split():
        ws = tokenizer.tokenize(word)
        if not ws: continue # some special char
        ids.extend(tokenizer.convert_tokens_to_ids(ws))
    return ids


def name_token_to_lower(caption):
    return caption.replace('[NAME]', '[name]')


def replace_name_token_cased(tokens):
    """
    :param tokens: tokens output by the cased BertTokenizer
    :return: tokens with the sequence 164('['), 1271('name'), 166(']') replaced by 104 ('[NAME]')
    """
    while 1271 in tokens:
        i = tokens.index(1271)
        if i - 1 >= 0 and i + 1 < len(tokens) and tokens[i - 1] == 164 and tokens[i + 1] == 166:
            tokens[i - 1] = 104
            del tokens[i + 1]
            del tokens[i]
        else:
            tokens[i] = 105
    for i in range(len(tokens)):
        if tokens[i] == 105: tokens[i] = 1271
    return tokens


def process_people(opt, db, tokenizer):
    annotation_json_path = os.path.join(annotation_dir, opt.ann, opt.split + '.json')
    examples = json.load(io.open(annotation_json_path, 'r', encoding='utf-8'))
    print(f'loaded {len(examples)} examples for {opt.split}...')

    id2len = {}
    id2category = {}
    for example in examples:
        id_ = example['id']
        caption_lower = name_token_to_lower(example['caption'])
        tokens = tokenize(tokenizer, caption_lower)
        input_ids = replace_name_token_cased(tokens)  # all the [name] tokens map to 104
        name_pos = [i for i in range(len(input_ids)) if input_ids[i] == 104]
        assert len(name_pos) == len(example['corefs'])
        iden2token_pos = {}
        for i in range(len(name_pos)):
            iden = example['corefs'][i][1]
            if not iden in iden2token_pos.keys():
                iden2token_pos[iden] = [name_pos[i]]
            else:
                iden2token_pos[iden].append(name_pos[i])

        gt_rev = {v: k for k, v in example['gt'].items()}
        example['gt_rev'] = gt_rev
        example['iden2token_pos'] = iden2token_pos
        example['input_ids'] = input_ids
        db[id_] = example
        id2len[id_] = len(input_ids)
        id2category[id_] = example['category']
    print(f'database length for {opt.split} is {len(examples)}')
    return id2len, id2category


def main(opts):
    output_dir = os.path.join(txt_db_dir, opts.output, opts.split)
    if exists(output_dir):
        shutil.rmtree(output_dir)
        print("Removed existing DB at " + output_dir)

    os.makedirs(output_dir)
    meta = vars(opts)
    meta['tokenizer'] = opts.tokenizer

    tokenizer = BertTokenizer(opts.vocab, do_lower_case=False)
    meta['UNK'] = tokenizer.convert_tokens_to_ids(['[UNK]'])[0]
    meta['CLS'] = tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
    meta['SEP'] = tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
    meta['MASK'] = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
    meta['NAME'] = tokenizer.convert_tokens_to_ids(['[NAME]'])[0]
    meta['v_range'] = (tokenizer.convert_tokens_to_ids('!')[0],
                       len(tokenizer.vocab))
    with open(f'{output_dir}/meta.json', 'w') as f:
        json.dump(vars(opts), f, indent=4)

    open_db = curry(open_lmdb, output_dir, readonly=False)
    with open_db() as db:
        id2len, id2category = process_people(opts, db, tokenizer)
    with open(f'{output_dir}/id2len.json', 'w') as f:
        json.dump(id2len, f)
    with open(f'{output_dir}/id2category.json', 'w') as f:
        json.dump(id2category, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required=True,
                        help='which split')
    parser.add_argument('--ann', required=True,
                        help='name of directory containing annotations')
    parser.add_argument('--output', required=True,
                        help='output dir of DB')
    parser.add_argument('--vocab', default='./dataset_meta/vocab-cased-with-name.txt',
                        help='vocabulary for tokenizer')
    parser.add_argument('--tokenizer', default='bert-base-cased',
                        help='which BERT tokenizer to used')

    args = parser.parse_args()
    main(args)
