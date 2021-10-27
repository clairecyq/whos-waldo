"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

convert image npz to LMDB
"""
import argparse
import glob
import io
import json
import multiprocessing as mp
import os, shutil
from os.path import basename, exists

from cytoolz import curry
import numpy as np
from tqdm import tqdm
import lmdb

import msgpack
import msgpack_numpy

msgpack_numpy.patch()

img_db_dir = './storage/img_db'
split_dir = './dataset_meta/splits'
npz_dir = '/whos-waldo-features'


def read_json(json_path):
    f = json.load(io.open(json_path, 'r', encoding='utf-8'))
    return f


def _compute_nbb(img_dump, conf_th, max_bb, min_bb):
    num_bb = max(min_bb, (img_dump['conf'] > conf_th).sum())
    num_bb = min(max_bb, num_bb)
    return int(num_bb)


@curry
def load_npz(conf_th, max_bb, min_bb, num_bb, fname, keep_all=False):
    try:
        img_dump = np.load(fname, allow_pickle=True)
        if keep_all:
            nbb = img_dump['norm_bb'].shape[0]
        else:
            nbb = _compute_nbb(img_dump, conf_th, max_bb, min_bb, num_bb)
        dump = {}
        for key, arr in img_dump.items():
            if arr.dtype == np.float32:
                arr = arr.astype(np.float16)
            if arr.ndim == 2:
                dump[key] = arr[:nbb, :]
            elif arr.ndim == 1:
                dump[key] = arr[:nbb]
            else:
                raise ValueError('wrong ndim')
    except Exception as e:
        # corrupted file
        print(f'corrupted file {fname}', e)
        dump = {}
        nbb = 0

    name = basename(fname)
    return name, dump, nbb


def dumps_npz(dump, compress=False):
    with io.BytesIO() as writer:
        if compress:
            np.savez_compressed(writer, **dump, allow_pickle=True)
        else:
            np.savez(writer, **dump, allow_pickle=True)
        return writer.getvalue()


def dumps_msgpack(dump):
    return msgpack.dumps(dump, use_bin_type=True)


def main(opts):
    output_dir = os.path.join(img_db_dir, opts.output, opts.split)
    if exists(output_dir):
        shutil.rmtree(output_dir)
        print("Removed existing DB at " + output_dir)
    os.makedirs(output_dir)

    db_name = 'all'

    env = lmdb.open(f'{output_dir}/{db_name}', map_size=1024 ** 4)
    txn = env.begin(write=True)

    # need to do a search based on split
    split_file = os.path.join(split_dir, opts.split + '.txt')
    with open(split_file, 'r') as f:
        split_lst = [id.rstrip('\n') for id in f.readlines()]
    print(f'There are in total {len(split_lst)} examples in {opts.split}')

    files = []
    for id in split_lst:
        folder = os.path.join(npz_dir, id)
        npz_files = [f for f in glob.glob(os.path.join(folder, '*.*')) if f.endswith('.npz')]
        try:
            assert len(npz_files) == 1
        except:
            print(f'{id} does not have exactly 1 features npz file: {npz_files}')
            continue
        npz_fn = npz_files[0]
        files.append(npz_fn)

    print("Number of files: " + str(len(files)))

    load = load_npz(opts.conf_th, opts.max_bb, opts.min_bb, opts.num_bb,
                    keep_all=True)

    name2nbb = {}
    with mp.Pool(opts.nproc) as pool, tqdm(total=len(files)) as pbar:
        for i, (fname, features, nbb) in enumerate(
                pool.imap(load, files, chunksize=128)):
            if not features:
                continue  # corrupted feature
            if opts.compress:
                dump = dumps_npz(features, compress=True)
            else:
                dump = dumps_msgpack(features)
            id = files[i].split('/')[-2]
            txn.put(key=id.encode('utf-8'), value=dump)
            if i % 1000 == 0:
                txn.commit()
                txn = env.begin(write=True)
            name2nbb[id] = nbb
            pbar.update(1)
        txn.put(key=b'__keys__',
                value=json.dumps(list(name2nbb.keys())).encode('utf-8'))
        txn.commit()
        env.close()
    with open(f'{output_dir}/name2nbb.json', 'w') as f:
        json.dump(name2nbb, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True,
                        help='output dir of DB')
    parser.add_argument("--split", required=True, type=str,
                        help="train/val/test split")
    parser.add_argument('--compress', action='store_true',
                        help='compress the tensors')
    parser.add_argument('--conf_th', type=float, default=0.2,
                        help='threshold for dynamic bounding boxes '
                             '(-1 for fixed)')
    parser.add_argument('--max_bb', type=int, default=100,
                        help='max number of bounding boxes')
    parser.add_argument('--min_bb', type=int, default=1,
                        help='min number of bounding boxes')
    parser.add_argument('--num_bb', type=int, default=100,
                        help='number of bounding boxes (fixed)')
    parser.add_argument('--nproc', type=int, default=8,
                        help='number of cores used')
    args = parser.parse_args()
    main(args)
