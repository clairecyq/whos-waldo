import argparse
import json, os

import torch
from torch.utils.data import DataLoader
from apex import amp
from horovod import torch as hvd

from utils.visualize import print_visualization
from data import (PrefetchLoader, DetectFeatLmdb,
                  TxtTokLmdb, WhosWaldoDataset, whos_waldo_ot_collate)
from model import UniterConfig
from model import WhosWaldo
from utils.logger import LOGGER
from utils.misc import Struct
from utils.const import IMG_DIM
import warnings
warnings.filterwarnings("ignore")

with open('/dataset_meta/blurry_bbs.json', 'r', encoding='utf-8') as file:
    blurry_bbs = json.load(file)


@torch.no_grad()
def evaluate_gt(model, eval_loader, null_id, result_txt, html_correct, html_incorrect, output_dir, vis_num):
    print("start running evaluation in evaluate_gt")
    html_correct = open(html_correct, 'w')
    html_incorrect = open(html_incorrect, 'w')

    model.eval()
    total_scores = 0
    correct_scores = 0
    random_guess = 0
    per_example_result = {}  # {img_id: fraction of links correct}
    cnt = 0
    vis_cnt = 0

    for i, batch in enumerate(eval_loader):
        id = batch['id'][0]
        gt = batch['gt'][0]
        assert len(gt) > 0
        targets = batch['targets']
        assert targets[0] != 0
        boxes = batch['img_pos_feat'][0]
        num_faces = boxes.shape[0]

        gt_losses, gt_scores, null_id_cnt, _, sim, _ = model(batch, task='gt', null_id=null_id)
        assert gt_losses is not None

        total_scores += len(gt)
        correct_scores += gt_scores['gt_row_scores']
        random_guess += len(gt) / num_faces
        cnt += 1
        per_example_result[id] = gt_scores['gt_row_scores'] / len(gt)

        num_idens = len([x for y in batch['iden2token_pos'][0].values() for x in y])
        sim = sim[0].detach().cpu().numpy()

        if vis_cnt < vis_num:
            if gt_scores['gt_row_scores'] < len(gt):  # not all correct
                print_visualization(id, num_idens, boxes, sim, null_id, gt, html_out=html_incorrect, output_dir=output_dir)
            elif gt_scores['gt_row_scores'] == len(gt):  # all correct
                print_visualization(id, num_idens, boxes, sim, null_id, gt, html_out=html_correct, output_dir=output_dir)
            else:
                raise ValueError('Scores cannot be higher than number of gt pairs')
            vis_cnt += 1

    html_correct.close()
    html_incorrect.close()
    with open(result_txt, 'w') as fp:
        print(f'images evaluated {cnt}', file=fp)
        print(f'Possible scores {total_scores}', file=fp)
        print(f'Scores achieved {correct_scores}', file=fp)
        print(f'Correct scores% {100 * correct_scores / total_scores}%', file=fp)
        print(f'Scores from random guess {100 * random_guess / total_scores}&', file=fp)
        print(per_example_result, file=fp)

    model.train()
    print(f'Correct scores% {100 * correct_scores / total_scores}%')


def main(opts):
    model_dir = f'/storage/finetune/{opts.model_dir}'
    hvd.init()
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
        device, n_gpu, hvd.rank(), opts.fp16))
    train_opts = Struct(json.load(open(f'{model_dir}/log/hps.json')))
    opts.conf_th = train_opts.conf_th
    opts.max_bb = train_opts.max_bb
    opts.min_bb = train_opts.min_bb
    opts.num_bb = train_opts.num_bb
    opts.null_id = train_opts.null_id

    # load DBs and image dirs
    eval_img_db = DetectFeatLmdb(f'/storage/img_db/{opts.img_db}/{opts.split}',
                                 opts.conf_th, opts.max_bb,
                                 opts.min_bb, opts.num_bb,
                                 opts.compressed_db)
    eval_txt_db = TxtTokLmdb(f'/storage/txt_db/{opts.txt_db}/{opts.split}', train_opts.max_txt_len)
    categories = ['interactive']
    if not opts.interactive_only:
        categories.append('other')
    eval_gt_dataset = WhosWaldoDataset(eval_txt_db, eval_img_db, categories, 0)

    print(f'{len(eval_gt_dataset)} examples in dataset')

    # Prepare model
    if opts.use_pretrained:
        ckpt_file = '/storage/pretrain/uniter-base-pretrained.pt'
    else:
        ckpt_file = f'{model_dir}/ckpt/model_step_{opts.ckpt}.pt'
    checkpoint = torch.load(ckpt_file)
    model_config = UniterConfig.from_json_file(f'{model_dir}/log/model.json')
    model = WhosWaldo(model_config, img_dim=IMG_DIM)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model = amp.initialize(model, enabled=opts.fp16, opt_level='O2')

    eval_gt_dataloader = DataLoader(eval_gt_dataset, batch_size=1,
                                    num_workers=opts.n_workers,
                                    pin_memory=opts.pin_mem,
                                    collate_fn=whos_waldo_ot_collate)
    eval_gt_dataloader = PrefetchLoader(eval_gt_dataloader)

    output_dir = f'{model_dir}/eval_outputs'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    html_correct = f'{output_dir}/{opts.eval_output_name}-correct.html'
    html_incorrect = f'{output_dir}/{opts.eval_output_name}-incorrect.html'
    result_txt = f'{output_dir}/{opts.eval_output_name}.txt'
    evaluate_gt(model, eval_gt_dataloader, opts.null_id, result_txt, html_correct, html_incorrect, output_dir, opts.vis_num)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt-db",
                        type=str, required=True,
                        help="The input train corpus.")
    parser.add_argument("--img-db",
                        type=str, required=True,
                        help="The input train images.")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="The directory storing the finetuned models")
    parser.add_argument("--ckpt", type=int, default=0,
                        help="specify the checkpoint to run inference, must be specified if --use-pretrained=False")
    parser.add_argument("--use-pretrained",
                        type=bool, default=False,
                        help="Whether to use pretrained UNITER model")
    parser.add_argument("--vis-num",
                        type=int, default=0,
                        help="How many examples to visualize.")
    parser.add_argument("--split",
                        type=str, required=True,
                        help="The split to evaluate on.")
    parser.add_argument("--null-id",
                        action='store_true',
                        help="whether to use null identity")
    parser.add_argument("--eval-output-name",
                        type=str, required=True,
                        help="file name of eval outputs")
    parser.add_argument("--interactive-only",
                        action='store_true',
                        help="if true, evaluate on only the 'interactive' category, otherwise include 'other'")
    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')
    parser.add_argument("--batch_size", type=int,
                        help="batch size for evaluation")
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true',
                        help="pin memory")
    parser.add_argument('--fp16', default=True,
                        help="fp16 inference")
    args = parser.parse_args()

    main(args)
