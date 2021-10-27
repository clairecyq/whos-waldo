import random
import torch
import numpy as np
from toolz.sandbox import unzip
from torch.nn.utils.rnn import pad_sequence
from .sampler import TokenBucketSampler
from .data import (DetectFeatTxtTokDataset, DetectFeatLmdb, TxtTokLmdb,
                   pad_tensors, get_gather_index, get_ids_and_lens)


def _has_overlap(la, lb):
    if len(la) < len(lb):
        la, lb = lb, la
    s = set(la)
    return any(b in s for b in lb)


def sample_negative(sample_pool, ground_truths, num_sample):
    """ random and retry """
    outputs = ground_truths[:1]
    while _has_overlap(outputs, ground_truths):
        outputs = random.sample(sample_pool, num_sample)
    return outputs


class TokenBucketSamplerForItm(TokenBucketSampler):
    def __init__(self, dset, *args, **kwargs):
        super().__init__(dset.lens, *args, **kwargs)
        self.dset = dset

    def __iter__(self):
        it = super().__iter__()
        self.dset.new_epoch()
        self._lens = self.dset.lens
        return it


class WhosWaldoDataset(DetectFeatTxtTokDataset):
    def __init__(self, txt_db, img_db, category, neg_sample_p=0.5):
        assert isinstance(txt_db, TxtTokLmdb)
        assert isinstance(img_db, DetectFeatLmdb)

        self.txt_db = txt_db
        self.img_db = img_db
        self.category = category

        self.txt_lens, self.ids = get_ids_and_lens(txt_db)
        id2category = self.txt_db.id2category

        filtered_lens = []
        filtered_ids = []
        for i, (id_, tl) in enumerate(zip(self.ids, self.txt_lens)):
            if not id2category[id_] in self.category:
                continue
            assert id_ in self.img_db.name2nbb.keys()
            assert self.img_db.name2nbb[id_] > 0
            if self.category == ['one-to-one']:
                assert self.img_db.name2nbb[id_] == 1
            elif self.category == ['interactive']:
                assert self.img_db.name2nbb[id_] > 1, id_

            filtered_lens.append(tl)
            filtered_ids.append(id_)

        self.txt_lens = filtered_lens
        self.ids = filtered_ids
        self.neg_sample_p = neg_sample_p
        self.new_epoch()


    def new_epoch(self):
        """ should be called every epoch for more randomness"""
        self.labels = np.random.choice(
            [0, 1], size=len(self.ids),
            p=[self.neg_sample_p, 1 - self.neg_sample_p])

        self.lens = []
        self.train_imgs = []
        self.train_neg_imgs = []
        for i, (id_, tl) in enumerate(zip(self.ids, self.txt_lens)):
            if self.labels[i] == 0:
                img_neg_id = sample_negative(self.ids, [id_], 1)[0]
            else:
                img_neg_id = id_

            self.train_imgs.append(id_)
            self.train_neg_imgs.append(img_neg_id)
            self.lens.append(tl + self.img_db.name2nbb[img_neg_id])

    def __getitem__(self, i):
        example = super().__getitem__(i)
        # labels and negative images should be sampled every epoch
        ground_truth_label = self.labels[i]
        img_neg_id = self.train_neg_imgs[i]
        img_feat, img_pos_feat, num_bb = self._get_img_feat(img_neg_id)  # uses images from the neg example

        # text input
        input_ids = example['input_ids']  # use text from original example
        input_ids = self.txt_db.combine_inputs(input_ids)

        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)
        target = torch.Tensor(1).long()
        target.data.fill_(ground_truth_label)

        return input_ids, img_feat, img_pos_feat, attn_masks, target, example['id'], img_neg_id, example['iden2token_pos'], example['gt']


def _compute_ot_scatter(txt_lens, max_txt_len, joint_len):
    ot_scatter = torch.arange(0, joint_len, dtype=torch.long
                              ).unsqueeze(0).repeat(len(txt_lens), 1)
    for i, tl in enumerate(txt_lens):
        max_ind = max_txt_len + (joint_len-tl)
        ot_scatter.data[i, tl:] = torch.arange(max_txt_len, max_ind,
                                               dtype=torch.long).data
    return ot_scatter


def _compute_pad(lens, max_len):
    pad = torch.zeros(len(lens), max_len, dtype=torch.uint8)
    for i, l in enumerate(lens):
        pad.data[i, l:].fill_(1)
    return pad


def whos_waldo_ot_collate(inputs):
    (input_ids, img_feats, img_pos_feats, attn_masks, targets, id, img_neg_id, iden2token_pos, gt
     ) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    targets = torch.cat(targets, dim=0)
    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    # OT inputs
    max_tl = max(txt_lens)
    max_nbb = max(num_bbs)
    ot_scatter = _compute_ot_scatter(txt_lens, max_tl, attn_masks.size(1))
    txt_pad = _compute_pad(txt_lens, max_tl)
    img_pad = _compute_pad(num_bbs, max_nbb)
    ot_inputs = {'ot_scatter': ot_scatter,
                 'scatter_max': ot_scatter.max().item(),
                 'txt_pad': txt_pad,
                 'img_pad': img_pad}

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'targets': targets,
             'ot_inputs': ot_inputs,
             'id': id,
             'img_neg_id': img_neg_id,
             'iden2token_pos': iden2token_pos,
             'gt': gt,
             'num_bbs': num_bbs}
    return batch



