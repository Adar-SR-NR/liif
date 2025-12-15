""" Train for generating LIIF, from image to implicit representation.

    Config:
        train_dataset:
          dataset: $spec; wrapper: $spec; batch_size:
        val_dataset:
          dataset: $spec; wrapper: $spec; batch_size:
        (data_norm):
            inp: {sub: []; div: []}
            gt: {sub: []; div: []}
        (eval_type):
        (eval_bsize):

        model: $spec
        optimizer: $spec
        epoch_max:
        (multi_step_lr):
            milestones: []; gamma: 0.5
        (resume): *.pth

        (epoch_val): ; (epoch_save):
"""

import argparse
import os

import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

import datasets
import models
import utils
from test import eval_psnr


def make_data_loader(spec, tag='', rank=0):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    
    if rank == 0:
        log('{} dataset: size={}'.format(tag, len(dataset)))
        for k, v in dataset[0].items():
            if hasattr(v, 'shape'):
                log('  {}: shape={}'.format(k, tuple(v.shape)))
            else:
                log('  {}: value={}'.format(k, v))

    sampler = DistributedSampler(dataset, shuffle=(tag == 'train'))

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=False, num_workers=8, pin_memory=True, sampler=sampler)
    return loader


def make_data_loaders(rank):
    train_loader = make_data_loader(config.get('train_dataset'), tag='train', rank=rank)
    val_loader = make_data_loader(config.get('val_dataset'), tag='val', rank=rank)
    return train_loader, val_loader


def prepare_training(local_rank):
    if config.get('resume') is not None:
        sv_file = torch.load(config['resume'], map_location='cpu') # 建議加 map_location
        model = models.make(sv_file['model'], load_sd=True).cuda(local_rank)
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1
        # ... (LR scheduler 部分不變)
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        for _ in range(epoch_start - 1):
            lr_scheduler.step()
    else:
        model = models.make(config['model']).cuda(local_rank)
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        # ... (LR scheduler 部分不變)
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])

    # 僅 rank 0 打印模型參數
    if local_rank == 0:
        log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model, optimizer, local_rank):
    model.train()
    loss_fn = nn.L1Loss()
    train_loss = utils.Averager()

    data_norm = config['data_norm']
    t = data_norm['inp']
    # [修改] 指定 device 為 local_rank
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda(local_rank)
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda(local_rank)
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda(local_rank)
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda(local_rank)

    # [修改] 僅 rank 0 顯示進度條
    iterator = tqdm(train_loader, leave=False, desc='train') if local_rank == 0 else train_loader
    for batch in iterator:
        for k, v in batch.items():
            batch[k] = v.cuda(local_rank) # [修改] 指定 GPU

        inp = (batch['inp'] - inp_sub) / inp_div
        pred = model(inp, batch['coord'], batch['cell'],
                     scale=batch['scale_h'], scale2=batch['scale_w'])
        
        gt = (batch['gt'] - gt_sub) / gt_div
        loss = loss_fn(pred, gt)
        train_loss.add(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = None; loss = None

    return train_loss.item()


def main(config_, save_path):
    global config, log, writer
    
    # [新增] 初始化 DDP 環境
    # torchrun 會自動設定 LOCAL_RANK 等環境變數
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    
    config = config_
    
    # [修改] 僅 Rank 0 負責寫檔和 Log
    if local_rank == 0:
        log, writer = utils.set_save_path(save_path)
        with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
            yaml.dump(config, f, sort_keys=False)
    else:
        # 其他進程使用空的 log 函數避免報錯
        log = lambda *args, **kwargs: None
        class DummyWriter:
            def add_scalar(self, *args): pass
            def add_scalars(self, *args): pass
            def flush(self): pass
        writer = DummyWriter()

    train_loader, val_loader = make_data_loaders(rank=local_rank)
    
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, lr_scheduler = prepare_training(local_rank)

    # [修改] 使用 DDP 封裝模型
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_v = -1e18

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        # [新增] 每個 epoch 需設定 sampler 的 epoch 以打亂數據
        train_loader.sampler.set_epoch(epoch)
        
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        train_loss = train(train_loader, model, optimizer, local_rank)
        if lr_scheduler is not None:
            lr_scheduler.step()

        log_info.append('train: loss={:.4f}'.format(train_loss))
        writer.add_scalars('loss', {'train': train_loss}, epoch)

        # 取得原始模型 (DDP 包了一層 module)
        model_ = model.module
        
        # [修改] 僅 Rank 0 負責儲存與驗證
        if local_rank == 0:
            model_spec = config['model']
            model_spec['sd'] = model_.state_dict()
            optimizer_spec = config['optimizer']
            optimizer_spec['sd'] = optimizer.state_dict()
            sv_file = {
                'model': model_spec,
                'optimizer': optimizer_spec,
                'epoch': epoch
            }

            torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

            if (epoch_save is not None) and (epoch % epoch_save == 0):
                torch.save(sv_file,
                    os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

            if (epoch_val is not None) and (epoch % epoch_val == 0):
                # 驗證通常只在單卡或 Rank 0 做即可，避免重複計算
                val_res = eval_psnr(val_loader, model_,
                    data_norm=config['data_norm'],
                    eval_type=config.get('eval_type'),
                    eval_bsize=config.get('eval_bsize'))

                log_info.append('val: psnr={:.4f}'.format(val_res))
                writer.add_scalars('psnr', {'val': val_res}, epoch)
                if val_res > max_val_v:
                    max_val_v = val_res
                    torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

            t = timer.t()
            prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
            t_epoch = utils.time_text(t - t_epoch_start)
            t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
            log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

            log(', '.join(log_info))
            writer.flush()
        
        # 確保所有 GPU 同步進入下一個 epoch
        dist.barrier()
    if local_rank == 0:
        writer.close() # 關閉 writer
        
    dist.destroy_process_group() # [新增] 銷毀進程組，釋放資源


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    # [修改] 移除 --gpu 參數，改由環境變數或 torchrun 控制
    args = parser.parse_args()

    # DDP 不需要手動設置 CUDA_VISIBLE_DEVICES，通常由啟動腳本控制

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        # 僅 print 一次
        if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
            print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path)
