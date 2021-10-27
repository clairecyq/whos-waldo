import argparse
from collections import defaultdict
import json
import os, shutil
from os.path import exists, join
from time import time

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from apex import amp
from horovod import torch as hvd

from tqdm import tqdm

from data import (TokenBucketSamplerForItm,
                  MetaLoader, PrefetchLoader,
                  TxtTokLmdb, ImageLmdbGroup,
                  WhosWaldoDataset, whos_waldo_ot_collate)

from model.whos_waldo import WhosWaldo
from optim import get_lr_sched
from optim.misc import build_optimizer

from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.distributed import (all_reduce_and_rescale_tensors, all_gather_list,
                               broadcast_tensors)
from utils.save import ModelSaver, save_training_meta
from utils.misc import NoOp, parse_with_config, set_dropout, set_random_seed
from utils.const import IMG_DIM, BUCKET_SIZE
import warnings
warnings.filterwarnings("ignore")


def build_dataloader_itm(dataset, collate_fn, is_train, opts):
    if is_train:
        batch_size = opts.train_batch_size
    else:
        batch_size = opts.val_batch_size
    sampler = TokenBucketSamplerForItm(
        dataset, bucket_size=BUCKET_SIZE,
        batch_size=batch_size, droplast=is_train)
    loader = DataLoader(dataset, batch_sampler=sampler,
                        num_workers=opts.n_workers, pin_memory=opts.pin_mem,
                        collate_fn=collate_fn)
    return loader


def build_whos_waldo_dataset(txt_db, img_db, category, neg_prob):
    dataset = WhosWaldoDataset(txt_db, img_db, category, neg_prob)
    collate_fn = whos_waldo_ot_collate
    return dataset, collate_fn


def create_dataloaders(datasets, is_train, opts, all_img_dbs=None):
    if all_img_dbs is None:
        all_img_dbs = ImageLmdbGroup(opts.conf_th, opts.max_bb, opts.min_bb,
                                     opts.num_bb, opts.compressed_db)
    dataloaders = {}
    for dset in datasets:
        assert len(dset['tasks']) == len(dset['mix_ratio'])
        img_db = all_img_dbs[dset['img']]

        for i, t in enumerate(dset['tasks']):
            task = f'{t}_{dset["name"]}'
            LOGGER.info(f"Loading {task} text dataset "
                        f"{dset['db']}, {dset['img']}")
            txt_db = TxtTokLmdb(dset['db'], opts.max_txt_len)

            if task.startswith('matching'):
                dataset = build_whos_waldo_dataset(txt_db, img_db, ['one-to-one'], opts.itm_neg_prob)
            elif task.startswith('gt'):
                dataset = build_whos_waldo_dataset(txt_db, img_db, ['interactive', 'other'], 0)
            else:
                raise ValueError(f'Undefined task {task}')

            LOGGER.info(f"{len(dataset[0])*hvd.size()} samples loaded")
            loader = build_dataloader_itm(*dataset, is_train, opts)
            if is_train:
                ratio = dset['mix_ratio'][i]
                dataloaders[task] = (loader, ratio)
            else:
                dataloaders[task] = PrefetchLoader(loader)
    return dataloaders, all_img_dbs


def main(opts):
    hvd.init()
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    rank = hvd.rank()
    opts.rank = rank
    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(device, n_gpu, hvd.rank(), opts.fp16))

    if opts.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, "
                         "should be >= 1".format(
                            opts.gradient_accumulation_steps))

    set_random_seed(opts.seed)

    if rank == 0:
        save_training_meta(opts)
        TB_LOGGER.create(join(opts.output_dir, 'log'))
        pbar = tqdm(total=opts.num_train_steps)
        model_saver = ModelSaver(join(args.output_dir, 'ckpt'))
        add_log_to_file(join(opts.output_dir, 'log', 'log.txt'))
    else:
        LOGGER.disabled = True
        pbar = NoOp()
        model_saver = NoOp()

    all_dbs = [db['db'] for db in opts.train_datasets + opts.val_datasets]

    tokenizer = json.load(open(f'{all_dbs[0]}/meta.json'))['tokenizer']
    assert all(tokenizer == json.load(open(f'{db}/meta.json'))['tokenizer']
               for db in all_dbs)

    # build data loaders
    train_dataloaders, all_img_dbs = create_dataloaders(
        opts.train_datasets, is_train=True, opts=opts)
    val_dataloaders, _ = create_dataloaders(
        opts.val_datasets, is_train=False, opts=opts, all_img_dbs=all_img_dbs)
    meta_loader = MetaLoader(train_dataloaders,
                             accum_steps=opts.gradient_accumulation_steps,
                             distributed=n_gpu > 1)
    meta_loader = PrefetchLoader(meta_loader)

    # Prepare model
    if opts.checkpoint:
        checkpoint = torch.load(opts.checkpoint)
    else:
        checkpoint = {}
    model = WhosWaldo.from_pretrained(opts.model_config, checkpoint, img_dim=IMG_DIM)
    model.to(device)
    model.train()
    # make sure every process has same model parameters in the beginning
    broadcast_tensors([p.data for p in model.parameters()], 0)
    set_dropout(model, opts.dropout)

    # Prepare optimizer
    optimizer = build_optimizer(model, opts)
    task2scaler = {t: i for i, t in enumerate(train_dataloaders.keys())}
    model, optimizer = amp.initialize(model, optimizer,
                                      num_losses=len(task2scaler),
                                      enabled=opts.fp16, opt_level='O2')

    global_step = 0
    LOGGER.info(f"***** Running training with {n_gpu} GPUs *****")
    LOGGER.info("  Batch size = %d", opts.train_batch_size)
    LOGGER.info("  Accumulate steps = %d", opts.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", opts.num_train_steps)

    # to compute training statistics
    task2loss = {task: RunningMeter(f'loss/task_{task}')
                 for task in train_dataloaders.keys()}

    for task in train_dataloaders.keys():
        if task.startswith('matching'):
            task2loss[f'{task}_matching_acc'] = RunningMeter(f'accuracy/task_{task}_matching')

        elif task.startswith('gt'):
            task2loss[f'{task}_gt_acc'] = RunningMeter(f'accuracy/task_{task}_gt')
            task2loss[f'{task}_gt_nullid_acc'] = RunningMeter(f'accuracy/task_{task}_gt_nullid')
            task2loss[f'{task}_xe'] = RunningMeter(f'loss/task_{task}_xe')
            task2loss[f'{task}_xe_row'] = RunningMeter(f'loss/task_{task}_xe_row')
            task2loss[f'{task}_xe_col'] = RunningMeter(f'loss/task_{task}_xe_col')
            task2loss[f'{task}_xe_null'] = RunningMeter(f'loss/task_{task}_xe_null')

    n_examples = defaultdict(int)
    n_in_units = defaultdict(int)
    n_loss_units = defaultdict(int)

    start = time()
    # quick hack for amp delay_unscale bug
    optimizer.zero_grad()
    optimizer.step()
    matching_ex = 0
    matching_scores = 0
    gt_ex = 0
    gt_nullid_ex = 0
    gt_tot_scores = 0
    gt_nullid_tot_scores = 0
    for step, (name, batch) in enumerate(meta_loader):
        # forward pass
        n_examples[name] += batch['input_ids'].size(0)
        n_in_units[name] += (batch['attn_masks'] == 1).sum().item()
        task = name.split('_')[0]
        loss = model(batch, task=task, null_id=opts.null_id)
        targets = batch['targets']

        if task.startswith('matching'):
            matching_loss, scores = loss
            n_loss_units[name] += matching_loss.size(0)
            loss = matching_loss.mean()

            matching_ex += len(targets)
            matching_scores += scores
            if global_step % opts.valid_steps == 0:
                val_acc = matching_scores / matching_ex
                task2loss[f'{name}_matching_acc'](val_acc)
                matching_scores = 0
                matching_ex = 0

        elif task.startswith('gt'):
            gt_losses, gt_scores, null_id_cnt, T, sim, sigmoid_sim = loss
            if gt_losses is not None and gt_losses['gt_row_loss'] is not None:
                n_gt_ex = gt_losses['gt_row_loss'].shape[0]
                gt_ex += n_gt_ex
                gt_nullid_ex += null_id_cnt

                n_loss_units[name] += gt_losses['gt_row_loss'].size(0)
                gt_loss = 0
                row_loss = gt_losses['gt_row_loss'].sum() / n_gt_ex
                col_loss = gt_losses['gt_col_loss'].sum() / n_gt_ex
                gt_loss += row_loss
                gt_loss += col_loss

                null_loss = None
                if gt_losses['gt_null_id_loss'] is not None:
                    null_loss = gt_losses['gt_null_id_loss'] / null_id_cnt
                    gt_loss += null_loss
                    gt_nullid_tot_scores += gt_scores['gt_null_id_scores']

                gt_tot_scores += gt_scores['gt_row_scores']
                if step % opts.valid_steps == 0:
                    acc = gt_tot_scores / gt_ex
                    task2loss[f'{name}_gt_acc'](acc)
                    if gt_nullid_ex > 0:
                        null_id_acc = gt_nullid_tot_scores / gt_nullid_ex
                        task2loss[f'{name}_gt_nullid_acc'](null_id_acc)
                    gt_ex = 0
                    gt_nullid_ex = 0
                    gt_tot_scores = 0
                    gt_nullid_tot_scores = 0

                task2loss[f'{name}_xe'](gt_loss.item())
                task2loss[f'{name}_xe_row'](row_loss.item())
                task2loss[f'{name}_xe_col'](col_loss.item())
                if gt_losses['gt_null_id_loss'] is not None:
                    task2loss[f'{name}_xe_null'](null_loss)

            else: # No gt in tha batch
                continue
            loss = gt_loss
        else:
            raise NotImplementedError("Undefined task")

        # backward pass
        delay_unscale = (step+1) % opts.gradient_accumulation_steps != 0
        with amp.scale_loss(loss, optimizer, delay_unscale=delay_unscale,
                            loss_id=task2scaler[name]) as scaled_loss:
            scaled_loss.backward()
            if not delay_unscale:
                # gather gradients from every processes
                # do this before unscaling to make sure every process uses
                # the same gradient scale
                grads = [p.grad.data for p in model.parameters()
                         if p.requires_grad and p.grad is not None]
                all_reduce_and_rescale_tensors(grads, float(1))

        task2loss[name](loss.item())

        # optimizer update and logging
        if (step + 1) % opts.gradient_accumulation_steps == 0:
            global_step += 1

            # learning rate scheduling
            lr_this_step = get_lr_sched(global_step, opts)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            TB_LOGGER.add_scalar('lr', lr_this_step, global_step)

            # log loss
            # NOTE: not gathered across GPUs for efficiency
            TB_LOGGER.log_scaler_dict({l.name: l.val
                                       for l in task2loss.values()
                                       if l.val is not None})
            TB_LOGGER.step()

            # update model params
            if opts.grad_norm != -1:
                grad_norm = clip_grad_norm_(amp.master_params(optimizer),
                                            opts.grad_norm)
                TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
            optimizer.step()
            optimizer.zero_grad()
            pbar.update(1)

            if global_step % 100 == 0:
                # monitor training throughput
                LOGGER.info(f'==============Step {global_step}===============')
                for t in train_dataloaders.keys():
                    assert all(tt == t for tt in all_gather_list(t))
                    tot_ex = sum(all_gather_list(n_examples[t]))
                    ex_per_sec = int(tot_ex / (time()-start))
                    tot_in = sum(all_gather_list(n_in_units[t]))
                    in_per_sec = int(tot_in / (time()-start))
                    tot_l = sum(all_gather_list(n_loss_units[t]))
                    l_per_sec = int(tot_l / (time()-start))
                    LOGGER.info(f'{t}: {tot_ex} examples trained at '
                                f'{ex_per_sec} ex/s')
                    TB_LOGGER.add_scalar(f'perf/{t}_ex_per_s', ex_per_sec,
                                         global_step)
                    TB_LOGGER.add_scalar(f'perf/{t}_in_per_s', in_per_sec,
                                         global_step)
                    TB_LOGGER.add_scalar(f'perf/{t}_loss_per_s', l_per_sec,
                                         global_step)
                LOGGER.info(f'===============================================')

            if global_step % opts.valid_steps == 0:
                LOGGER.info(f'Step {global_step}: start validation')
                validate(model, val_dataloaders, opts.null_id)
                model_saver.save(model, global_step)
        if global_step >= opts.num_train_steps:
            LOGGER.info('Actually finished all the training steps')
            LOGGER.info(global_step)
            LOGGER.info(opts.num_train_steps)
            break
    if global_step % opts.valid_steps != 0:
        LOGGER.info(f'Step {global_step}: start validation')
        validate(model, val_dataloaders, opts.null_id)
        model_saver.save(model, global_step)


def validate(model, val_dataloaders, null_id):
    model.eval()
    for task, loader in val_dataloaders.items():
        LOGGER.info(f"validate on {task} task")
        if task.startswith('matching'):
            val_log = validate_matching(model, loader)
        elif task.startswith('gt'):
            val_log = validate_gt(model, loader, null_id)
        else:
            raise NotImplementedError(f'Undefined task {task}')
        val_log = {f'{task}_{k}': v for k, v in val_log.items()}
        TB_LOGGER.log_scaler_dict(
            {f'valid_{task}/{k}': v for k, v in val_log.items()})
    model.train()

@torch.no_grad()
def validate_matching(model, val_loader):
    LOGGER.info("start running Matching validation...")
    val_loss = 0
    tot_score = 0
    n_ex = 0
    st = time()
    for i, batch in enumerate(val_loader):
        loss, matching_scores = model(batch, task='matching')
        val_loss += loss.sum().item()
        tot_score += matching_scores
        n_ex += loss.shape[0]
    val_loss = sum(all_gather_list(val_loss))
    tot_score = sum(all_gather_list(tot_score))
    n_ex = sum(all_gather_list(n_ex))
    if n_ex < 1:
        return {'valid/loss': 0.0,
               'valid/acc': 0.0,
               'valid/ex_per_s': 0}
    tot_time = time() - st
    val_loss /= n_ex
    val_acc = tot_score / n_ex
    val_log = {'valid/loss': val_loss,
               'valid/acc': val_acc,
               'valid/ex_per_s': n_ex / tot_time}

    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"score: {val_acc * 100:.2f}")
    return val_log

@torch.no_grad()
def validate_gt(model, val_loader, null_id):
    LOGGER.info("start running GT validation...")
    val_loss = 0
    val_row_loss = 0
    val_col_loss = 0
    val_null_loss = 0
    val_null_score = 0
    tot_score = 0
    n_ex = 0
    n_null_id_ex = 0
    st = time()
    for i, batch in enumerate(val_loader):
        gt_losses, gt_scores, null_id_cnt, _, _, _ = model(batch, task='gt', null_id=null_id)
        if gt_losses is None: continue
        n_gt_ex = gt_losses['gt_row_loss'].shape[0]

        row_loss = gt_losses['gt_row_loss'].sum()
        col_loss = gt_losses['gt_col_loss'].sum()
        val_loss += row_loss + col_loss
        val_row_loss += row_loss
        val_col_loss += col_loss
        if gt_losses['gt_null_id_loss'] is not None:
            val_loss += gt_losses['gt_null_id_loss']
            val_null_loss += gt_losses['gt_null_id_loss']
            val_null_score += gt_scores['gt_null_id_scores']
        else:
            assert null_id_cnt == 0
        tot_score += gt_scores['gt_row_scores']
        n_ex += n_gt_ex
        n_null_id_ex += null_id_cnt

    val_loss = sum(all_gather_list(val_loss))
    tot_score = sum(all_gather_list(tot_score))
    n_ex = sum(all_gather_list(n_ex))
    n_null_id_ex = sum(all_gather_list(n_null_id_ex))

    tot_time = time() - st
    val_loss /= n_ex
    val_row_loss /= n_ex
    val_col_loss /= n_ex
    val_acc = tot_score / n_ex

    val_log = {'valid/gt_loss': val_loss,
               'valid/gt_row_loss': val_row_loss,
               'valid/gt_col_loss': val_col_loss,
               'valid/acc': val_acc,
               'valid/ex_per_s': n_ex / tot_time
    }

    if n_null_id_ex != 0:
        val_null_loss /= n_null_id_ex
        val_nullid_acc = val_null_score / n_null_id_ex
        val_log['valid/gt_null_id_loss'] = val_null_loss
        val_log['valid/nullid_acc'] = val_nullid_acc

    LOGGER.info(f"validation finished in {int(tot_time)} seconds, " f"score: {val_acc * 100:.2f}")
    return val_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')
    parser.add_argument("--model_config", type=str,
                        help="path to model structure config json")
    parser.add_argument("--checkpoint", default=None, type=str,
                        help="path to model checkpoint (*.pt)")
    parser.add_argument(
        "--output_dir", default=None, type=str,
        help="The output directory where the model checkpoints will be "
             "written.")
    parser.add_argument('--null_id', default=False, type=bool,
                        help='whether to use null identity')
    parser.add_argument('--mrm_prob', default=0.15, type=float,
                        help='probability to mask in MRM training')
    parser.add_argument('--itm_neg_prob', default=0.5, type=float,
                        help='probability to make negative examples'
                             'in ITM training')
    parser.add_argument('--max_txt_len', type=int, default=60,
                        help='max number of tokens in text (BERT BPE)')
    parser.add_argument('--conf_th', type=float, default=0.2,
                        help='threshold for dynamic bounding boxes '
                             '(-1 for fixed)')
    parser.add_argument('--max_bb', type=int, default=100,
                        help='max number of bounding boxes')
    parser.add_argument('--min_bb', type=int, default=10,
                        help='min number of bounding boxes')
    parser.add_argument('--num_bb', type=int, default=36,
                        help='static number of bounding boxes')

    # training parameters
    parser.add_argument("--train_batch_size", default=4096, type=int,
                        help="Total batch size for training. "
                             "(batch by tokens)")
    parser.add_argument("--val_batch_size", default=4096, type=int,
                        help="Total batch size for validation. "
                             "(batch by tokens)")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
                        help="Number of updates steps to accumualte before "
                             "performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--valid_steps", default=1000, type=int,
                        help="Run validation every X steps")
    parser.add_argument("--num_train_steps", default=100000, type=int,
                        help="Total number of training updates to perform.")
    parser.add_argument("--optim", default='adamw',
                        choices=['adam', 'adamax', 'adamw'],
                        help="optimizer")
    parser.add_argument("--betas", default=[0.9, 0.98], nargs='+',
                        help="beta for adam optimizer")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="tune dropout regularization")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="weight decay (L2) regularization")
    parser.add_argument("--grad_norm", default=2.0, type=float,
                        help="gradient clipping (-1 for no clipping)")
    parser.add_argument("--warmup_steps", default=10000, type=int,
                        help="Number of training steps to perform linear "
                             "learning rate warmup for.")

    # device parameters
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true', help="pin memory")

    # can use config files
    parser.add_argument('--config', required=True, help='JSON config files')

    args = parse_with_config(parser)

    if exists(args.output_dir) and os.listdir(args.output_dir):
        shutil.rmtree(args.output_dir)
        print("************ Removed dir: "+args.output_dir)

    # options safe guard
    if args.conf_th == -1:
        assert args.max_bb + args.max_txt_len + 2 <= 512
    else:
        assert args.num_bb + args.max_txt_len + 2 <= 512

    main(args)
