import torch
import argparse
import random
import numpy as np
import os
import time

from configs.defaults import get_cfg_defaults
from data.dataset import load_dataset
from utils.logger import setup_logger
from models.model import CAT
from utils.preprocess import frames_preprocess
from utils.loss import compute_cls_loss, compute_seq_loss


def train():
    model = CAT(num_class=cfg.DATASET.NUM_CLASS,
                num_clip=cfg.DATASET.NUM_CLIP,
                dim_embedding=cfg.MODEL.DIM_EMBEDDING,
                pretrain=cfg.MODEL.PRETRAIN,
                dropout=cfg.TRAIN.DROPOUT,
                use_TE=cfg.MODEL.TRANSFORMER,
                use_SeqAlign=cfg.MODEL.ALIGNMENT,
                freeze_backbone=cfg.TRAIN.FREEZE_BACKBONE).to(device)

    for name, param in model.named_parameters():
        print(name, param.nelement())
    logger.info('Model have {} paramerters in total'.format(sum(x.numel() for x in model.parameters())))

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.TRAIN.MAX_EPOCH, eta_min=cfg.TRAIN.LR * 0.01)

    # Load checkpoint
    start_epoch = 0
    if args.load_path and os.path.isfile(args.load_path):
        checkpoint = torch.load(args.load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        logger.info('-> Loaded checkpoint %s (epoch: %d)' % (args.load_path, start_epoch))

    # Mulitple gpu
    if torch.cuda.device_count() > 1 and torch.cuda.is_available():
        logger.info('Let us use %d GPUs' % torch.cuda.device_count())
        model = torch.nn.DataParallel(model)

    model.train()



    # Create checkpoint dir
    if cfg.TRAIN.SAVE_PATH:
        checkpoint_dir = os.path.join(cfg.TRAIN.SAVE_PATH, 'save_models')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)


    # Start training
    start_time = time.time()
    for epoch in range(start_epoch, cfg.TRAIN.MAX_EPOCH):
        loss_per_epoch = 0
        num_true_pred = 0
        for iter, sample in enumerate(train_loader):

            frames1 = frames_preprocess(sample['clips1'][0]).to(device, non_blocking=True)
            frames2 = frames_preprocess(sample['clips2'][0]).to(device, non_blocking=True)
            labels1 = sample['labels1'].to(device, non_blocking=True)
            labels2 = sample['labels2'].to(device, non_blocking=True)

            pred1, seq_features1 = model(frames1)
            pred2, seq_features2 = model(frames2)

            loss_cls = compute_cls_loss(pred1, labels1) + compute_cls_loss(pred2, labels2)
            loss_seq = compute_seq_loss(seq_features1, seq_features2)
            loss = loss_cls + cfg.MODEL.SEQ_LOSS_COEF * loss_seq

            if (iter + 1) % 10 == 0:
                logger.info( 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, cfg.TRAIN.MAX_EPOCH, iter + 1, len(train_loader), loss.item()))

            loss_per_epoch += loss.item()
            num_true_pred += torch.sum(torch.argmax(pred1, dim=-1) == labels1) + torch.sum(torch.argmax(pred2, dim=-1) == labels2)

            # Update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Log training statistics
        loss_per_epoch /= (iter + 1)
        accuracy = num_true_pred / (cfg.DATASET.NUM_SAMPLE * 2)
        logger.info('Epoch [{}/{}], LR: {:.6f}, Accuracy: {:.4f}, Loss: {:.4f}'
                    .format(epoch + 1, cfg.TRAIN.MAX_EPOCH, optimizer.param_groups[0]['lr'], accuracy, loss_per_epoch))


        # Save model every X epochs
        if (epoch + 1) % cfg.MODEL.SAVE_EPOCHS == 0:
            save_dict = {'epoch': epoch + 1,  # after training one epoch, the start_epoch should be epoch+1
                         'optimizer_state_dict': optimizer.state_dict(),
                         'loss': loss.item(),
                         }
            try:  # with nn.DataParallel() the net is added as a submodule of DataParallel
                save_dict['model_state_dict'] = model.module.state_dict()
            except:
                save_dict['model_state_dict'] = model.state_dict()

            save_name = 'epoch_' + str(epoch + 1) + '.tar'
            torch.save(save_dict, os.path.join(checkpoint_dir, save_name))
            logger.info('Save ' + os.path.join(checkpoint_dir, save_name) + ' done!')

        # Learning rate decay
        scheduler.step()

    end_time = time.time()
    duration = end_time - start_time

    hour = duration // 3600
    min = (duration % 3600) // 60
    sec = duration % 60

    logger.info('Training cost %dh%dm%ds' % (hour, min, sec))



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='configs/train_resnet_config.yml', help='config file path')
    parser.add_argument('--save_path', default=None, help='path to save models and log')
    parser.add_argument('--load_path', default=None, help='path to load the model')
    parser.add_argument('--log_name', default='train_log', help='log name')


    args = parser.parse_args()
    return args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":

    args = parse_args()
    cfg = get_cfg_defaults()
    if args.config:
        cfg.merge_from_file(args.config)


    setup_seed(cfg.TRAIN.SEED)
    use_cuda = cfg.TRAIN.USE_CUDA and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    logger_path = os.path.join(cfg.TRAIN.SAVE_PATH, 'logs')
    logger = setup_logger('Sequence Verification', logger_path, args.log_name, 0)
    logger.info('Running with config:\n{}\n'.format(cfg))

    train_loader = load_dataset(cfg)
    train()