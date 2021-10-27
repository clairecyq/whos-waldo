"""
Generate metadata about the examples in each split.
- Splits train, val, test: uses ground truth generated using ground_truth_correspondences.py
- val_turkers, test_turkers: uses a subset of the ground truth from val, test that are verified by turkers
"""
import sys
sys.path.append('.')
import os
import json
import io
import argparse

root_dir = '.'
meta_dir = os.path.join(root_dir, 'dataset_meta')
whos_waldo_dir = os.path.join(root_dir, 'whos_waldo')
annotations_dir = os.path.join(root_dir, 'storage', 'annotations')
split_dir = os.path.join(meta_dir, 'splits')

def read_json(json_path):
    f = json.load(io.open(json_path, 'r', encoding='utf-8'))
    return f


def read_txt(f):
    return [n.rstrip('\n') for n in f.readlines()]


def start_idx(n):
    """
    :param n: name item, e.g.,
    :return: the starting position of the name
    e.g., start_idx([[13, 16], 1]) --> 13
    """
    return n[0][0]


def flatten_sort_corefs(corefs):
    """
    :param corefs: unflattened coreferences grouped by identities
    :return: flattened coreferences sorted by position in the caption, with the identity cluster index
    e.g., flatten_sort_corefs([ [ [1, 5], [21, 25] ], [ [13, 17] ] ])
                        --> [[[1, 5], 0], [[13, 17], 1], [[21, 25], 0]]
    """
    res = []
    for cluster_idx in range(len(corefs)):
        for name in corefs[cluster_idx]:
            res.append([name, cluster_idx])
    res.sort(key=start_idx)
    return res


with open(os.path.join(meta_dir, 'interactive_img_ids.txt'), 'r') as fp:
    interactive_imgs = read_txt(fp)


def generate_annotation(ex, filtered_gt_indices=None):
    folder = os.path.join(whos_waldo_dir, ex)

    with open(os.path.join(folder, 'caption.txt'), 'r') as f:
        caption = read_txt(f)[0]
    detections = read_json(os.path.join(folder, 'detections.json'))
    corefs = read_json(os.path.join(folder, 'coreferences.json'))
    ground_truth = read_json(os.path.join(folder, 'ground_truth.json'))

    if filtered_gt_indices:
        filtered_gt = {}
        for cluster_idx in filtered_gt_indices:
            filtered_gt[str(cluster_idx)] = ground_truth[str(cluster_idx)]
        ground_truth = filtered_gt

    category = 'other'
    if len(corefs) == 1 and len(detections) == 1:
        category = 'one-to-one'
    elif ex in interactive_imgs:
        category = 'interactive'

    flattened_corefs = flatten_sort_corefs(corefs)

    d = {'id': ex, 'caption': caption,
        'corefs': flattened_corefs,
        'gt': ground_truth, 'category': category}
    return d


def main(output):
    if not os.path.exists(os.path.join(annotations_dir, output)):
        os.mkdir(os.path.join(annotations_dir, output))
    for split in ['test', 'val', 'train']:
        split_file = os.path.join(split_dir, f'{split}.txt')
        with open(split_file, 'r') as fp:
            split_lst = read_txt(fp)
        print('split length for ' + split + ': ', len(split_lst))
        lst = []
        for ex in split_lst:
            d = generate_annotation(ex)
            lst.append(d)
        print(f'{split} of length {len(lst)}')

        with open(os.path.join(annotations_dir, output, split + '.json'), 'w') as fp:
            json.dump(lst, fp, ensure_ascii=False)

    for split in ['val_turkers', 'test_turkers']:
        split_file = os.path.join(split_dir, f'{split}.json')
        split_json = read_json(split_file)
        print('split length for ' + split + ': ', len(split_json))
        lst = []
        for ex, filtered_gt_indices in split_json.items():
            d = generate_annotation(ex, filtered_gt_indices)
            lst.append(d)
        print(f'{split} of length {len(lst)}')

        with open(os.path.join(annotations_dir, output, split + '.json'), 'w') as fp:
            json.dump(lst, fp, ensure_ascii=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True,
                        help='output dir name of annotations')
    args = parser.parse_args()
    main(args.output)

