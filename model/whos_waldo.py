from collections import defaultdict
import json
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
from .ot import optimal_transport_dist
from .uniter_model import (UniterPreTrainedModel, UniterModel)

nullid_file = '/storage/nullid.npz'
with np.load(nullid_file) as null_load:
    NULLID = null_load['nullid']

with open('/dataset_meta/blurry_bbs.json', 'r', encoding='utf-8') as file:
    blurry_bbs = json.load(file)


def _compute_pad(lens, max_len):
    pad = torch.zeros(len(lens), max_len, dtype=torch.uint8)
    for i, l in enumerate(lens):
        pad.data[i, l:].fill_(1)
    return pad


class WhosWaldo(UniterPreTrainedModel):
    def __init__(self, config, img_dim):
        super().__init__(config, img_dim)
        self.uniter = UniterModel(config, img_dim)

    def forward(self, batch, task, null_id=False):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attention_mask = batch['attn_masks']
        gather_index = batch['gather_index']
        targets = batch['targets']
        ot_inputs = batch['ot_inputs']
        ids = batch['id']
        iden2token_pos = batch['iden2token_pos']
        gt = batch['gt']
        num_bbs = batch['num_bbs']

        if task == 'matching':
            return self.forward_matching(input_ids, position_ids, img_feat, img_pos_feat,
                                         attention_mask, gather_index, targets,
                                         ot_inputs, iden2token_pos)
        elif task == 'gt':
            return self.forward_gt(input_ids, position_ids, img_feat, img_pos_feat,
                                   attention_mask, gather_index,
                                   ot_inputs, ids, iden2token_pos, gt, num_bbs, null_id)
        else:
            raise NotImplementedError('Undefined task for WhosWaldo model')

    def forward_matching(self, input_ids, position_ids, img_feat, img_pos_feat,
                         attention_mask, gather_index, targets, ot_inputs, iden2token_pos):
        """
        for 1-1 pairs
        """
        _, _, sigmoid_sim = self.forward_ot(
            input_ids, position_ids, img_feat, img_pos_feat, attention_mask,
            gather_index, ot_inputs, iden2token_pos, use_null_id=False
        )
        sigmoid_sim = sigmoid_sim.reshape(sigmoid_sim.shape[0])
        matching_loss = F.binary_cross_entropy(sigmoid_sim, targets.float(), reduction='none')
        matching_scores = np.sum(np.where(sigmoid_sim.cpu().detach().numpy() < 0.5, 0, 1) == targets.cpu().detach().numpy())
        return matching_loss, matching_scores

    def forward_ot(self, input_ids, position_ids, img_feat, img_pos_feat,
                   attention_mask, gather_index, ot_inputs, iden2token_pos, use_null_id=False):
        """
        compute similarity matrices
        """
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=False)

        ot_scatter = ot_inputs['ot_scatter']

        b = sequence_output.size(0)
        tl = input_ids.size(1)
        il = img_feat.size(1)
        max_l = max(ot_inputs['scatter_max'] + 1, tl + il)

        ot_scatter = ot_scatter.unsqueeze(-1).expand_as(sequence_output)
        ctx_emb = torch.zeros(b, max_l, self.config.hidden_size,
                              dtype=sequence_output.dtype,
                              device=sequence_output.device
                              ).scatter_(dim=1, index=ot_scatter,
                                         src=sequence_output)
        txt_emb = ctx_emb[:, :tl, :]
        img_emb = ctx_emb[:, tl:tl + il, :]

        img_pad = ot_inputs['img_pad']

        # trim txt_emb & txt_pad to only include [NAME] relevant tokens
        batch_size, max_text_len, emb_size = txt_emb.shape
        filtered_matrices = []
        txt_lens = []

        for ex in range(batch_size):
            iden2token_pos_ex = iden2token_pos[ex]
            if use_null_id:
                txt_lens.append(len(iden2token_pos_ex)+1)
            else:
                txt_lens.append(len(iden2token_pos_ex))  # number of identities

            mat_ex = txt_emb[ex]
            filtered_rows = []

            for identity_num, positions in iden2token_pos_ex.items():
                identity_embeddings = []
                for pos in positions:
                    identity_embeddings.append(mat_ex[pos+1])  # +1 as the [CLS] token is the first token
                arr = torch.stack(identity_embeddings, dim=0)
                mean_embedding = torch.mean(arr, axis=0)
                filtered_rows.append(mean_embedding)
            if use_null_id:
                filtered_rows.append(torch.tensor(data=NULLID, requires_grad=True).half().cuda())

            filtered_rows = torch.stack(filtered_rows, dim=0)
            filtered_matrices.append(filtered_rows)

        filtered_matrices = pad_sequence(filtered_matrices, batch_first=True)
        txt_emb = filtered_matrices
        max_tl = max(txt_lens)

        txt_pad = _compute_pad(txt_lens, max_tl).cuda()

        # NOTE: run in fp32 for stability
        T, _, C = optimal_transport_dist(txt_emb.float(), img_emb.float(),
                                         txt_pad, img_pad)

        sim = 1-C
        sigmoid_sim = torch.sigmoid(sim)
        T = torch.transpose(T, 1, 2)
        return T, sim, sigmoid_sim

    def forward_gt(self, input_ids, position_ids, img_feat, img_pos_feat,
                   attention_mask, gather_index, ot_inputs,
                   ids, iden2token_pos, gt, num_bbs, use_null_id=False):

        T, sim, sigmoid_sim = self.forward_ot(input_ids, position_ids, img_feat, img_pos_feat,
                                          attention_mask, gather_index, ot_inputs, iden2token_pos, use_null_id)

        gt_id_targets = []
        gt_id_results = []
        gt_face_targets = []
        gt_face_results = []

        null_id_cnt = 0
        null_id_correct = 0
        null_id_pairs_cnt = 0
        null_id_pairs_correct = 0
        null_id_ce_loss = 0

        for batch_idx in range(len(ids)):
            id = ids[batch_idx]
            gt_ex = gt[batch_idx]
            gt_rev_ex = {v: k for k, v in gt_ex.items()}
            box_cnt = num_bbs[batch_idx]
            iden2token_pos_ex = iden2token_pos[batch_idx]

            sigmoid_idx = sigmoid_sim[batch_idx]
            sim_idx = sim[batch_idx]

            # each row (identity)
            for identity_idx in gt_ex.keys():
                id_row = sim_idx[int(identity_idx)][:box_cnt]
                id_row_sm = F.softmax(id_row)
                gt_id_results.append(id_row_sm)
                gt_id_targets.append(gt_ex[identity_idx])

            # each column (person detection)
            num_ids = len(iden2token_pos_ex)
            sim_idx_T = torch.transpose(sim_idx, 0, 1)
            for face_idx in gt_rev_ex.keys():
                face_col = sim_idx_T[int(face_idx)][:num_ids]
                face_col_sm = F.softmax(face_col)
                gt_face_results.append(face_col_sm)
                gt_face_targets.append(int(gt_rev_ex[face_idx]))

            # null identity
            if use_null_id and num_ids > box_cnt and id in blurry_bbs.keys():
                null_id_cnt += 1
                blur_list = blurry_bbs[id]

                gt_boxes = [int(i) for i in gt_rev_ex.keys()]  # boxes for which we have gt
                null_id_row = sigmoid_idx[num_ids][:box_cnt]
                null_id_res = []
                nullid_targets = []

                for i in range(box_cnt):
                    if i in gt_boxes:  # has ground truth, not a null person!
                        nullid_targets.append(0.0)
                        null_id_res.append(null_id_row[i])
                        null_id_pairs_cnt += 1
                    elif i in blur_list: # does not have ground truth and is blurry, consider as null person
                        nullid_targets.append(1.0)
                        null_id_res.append(null_id_row[i])
                        null_id_pairs_cnt += 1

                null_id_res = torch.tensor(null_id_res)
                nullid_targets = torch.tensor(nullid_targets)
                example_loss = F.binary_cross_entropy(null_id_res, nullid_targets, reduction='mean')
                # average score for example
                example_scores = np.mean(np.where(null_id_res.numpy() < 0.5, 0, 1) == nullid_targets.numpy())
                # total score for this example
                example_total_scores = np.sum(np.where(null_id_res.numpy() < 0.5, 0, 1) == nullid_targets.numpy())

                null_id_ce_loss += example_loss
                null_id_correct += example_scores
                null_id_pairs_correct += example_total_scores

        # no example in the entire batch has ground truth
        if len(gt_id_results) == 0 or len(gt_face_results) == 0:
            return None, None, null_id_cnt, T, sim, sigmoid_sim

        gt_id_results = pad_sequence(gt_id_results, batch_first=True, padding_value=0.0)
        gt_id_targets = torch.tensor(gt_id_targets).cuda()
        gt_id_loss = F.cross_entropy(gt_id_results, gt_id_targets, reduction='none')
        gt_id_scores = (gt_id_results.max(dim=-1)[1] == gt_id_targets).sum().item()

        gt_face_results = pad_sequence(gt_face_results, batch_first=True, padding_value=0.0)
        gt_face_targets = torch.tensor(gt_face_targets).cuda()
        gt_face_loss = F.cross_entropy(gt_face_results, gt_face_targets, reduction='none')
        gt_face_scores = (gt_face_results.max(dim=-1)[1] == gt_face_targets).sum().item()

        gt_losses = {
            'gt_row_loss': gt_id_loss,
            'gt_col_loss': gt_face_loss,
            'gt_null_id_loss': None
        }
        gt_scores = {
            'gt_row_scores': gt_id_scores,
            'gt_col_scores': gt_face_scores,
            'gt_null_id_scores': None,
            'gt_id_res': gt_id_results.max(dim=-1)[1] == gt_id_targets
        }

        if null_id_cnt != 0:
            gt_losses['gt_null_id_loss'] = null_id_ce_loss / null_id_cnt
            gt_scores['gt_null_id_scores'] = float(null_id_correct) / float(null_id_cnt)
            gt_scores['gt_null_id_gt_pairs_correct'] = null_id_pairs_correct
            gt_scores['gt_null_id_gt_pairs_total'] = null_id_pairs_cnt
        return gt_losses, gt_scores, null_id_cnt, T, sim, sigmoid_sim
