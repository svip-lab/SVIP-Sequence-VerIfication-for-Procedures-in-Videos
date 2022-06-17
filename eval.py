import torch
import torch.nn.functional as F
import argparse
import random
import numpy as np
import os
import time

from tqdm import tqdm
from sklearn.metrics import auc, roc_curve

from configs.defaults import get_cfg_defaults
from data.dataset import load_dataset
from utils.logger import setup_logger
from models.model import CAT
from utils.preprocess import frames_preprocess
from train import setup_seed



def eval_one_model(model, dist='NormL2'):
    if torch.cuda.device_count() > 1 and torch.cuda.is_available():
        # logger.info("Let's use %d GPUs" % torch.cuda.device_count())
        model = torch.nn.DataParallel(model)

    # auc metric
    model.eval()

    with torch.no_grad():
        for iter, sample in enumerate(tqdm(test_loader)):

            frames1_list = sample['clips1']
            frames2_list = sample['clips2']
            assert len(frames1_list) == len(frames2_list)

            labels1 = sample['labels1']
            labels2 = sample['labels2']
            label = torch.tensor(np.array(labels1) == np.array(labels2)).to(device)

            embeds1_list = []
            embeds2_list = []

            for i in range(len(frames1_list)):
                frames1 = frames_preprocess(frames1_list[i]).to(device, non_blocking=True)
                frames2 = frames_preprocess(frames2_list[i]).to(device, non_blocking=True)
                embeds1 = model(frames1, embed=True)
                embeds2 = model(frames2, embed=True)
                embeds1_list.append(embeds1)
                embeds2_list.append(embeds2)

            embeds1_avg = np.sum(embeds1_list) / len(embeds1_list)
            embeds2_avg = np.sum(embeds2_list) / len(embeds2_list)

            if dist == 'L1':
                # L1 distance
                pred = torch.sum(torch.abs(embeds1_avg - embeds2_avg), dim=1)
            elif dist == 'L2':
                # L2 distance
                pred = torch.sum((embeds1_avg - embeds2_avg) ** 2, dim=1)
            elif dist == 'NormL2':
                # L2 distance between normalized embeddings
                pred = torch.sum((F.normalize(embeds1_avg, p=2, dim=1) - F.normalize(embeds2_avg, p=2, dim=1)) ** 2, dim=1)
            elif dist == 'cos':
                # Cosine similarity
                pred = torch.cosine_similarity(embeds1_avg, embeds2_avg, dim=1)

            if iter == 0:
                preds = pred
                labels = label
                labels1_all = labels1
                labels2_all = labels2
            else:
                preds = torch.cat([preds, pred])
                labels = torch.cat([labels, label])
                labels1_all += labels1
                labels2_all += labels2


    fpr, tpr, thresholds = roc_curve(labels.cpu().detach().numpy(), preds.cpu().detach().numpy(), pos_label=0)
    auc_value = auc(fpr, tpr)
    wdr_value = compute_WDR(preds, labels1_all, labels2_all)


    return auc_value, wdr_value


def compute_WDR(preds, labels1, labels2):
    # compute weighted distance ratio
    #        weighted dist / # unmatched pairs
    # WDR = ---------------------------------
    #             dist / # matched pairs
    import json
    def read_json(file_path):
        with open(file_path, 'r') as f:
            data = json.loads(f.read())
        return data

    def compute_edit_dist(seq1, seq2):
        """
        计算字符串 seq1 和 seq1 的编辑距离
        :param seq1
        :param seq2
        :return:
        """
        matrix = [[i + j for j in range(len(seq2) + 1)] for i in range(len(seq1) + 1)]
        for i in range(1, len(seq1) + 1):
            for j in range(1, len(seq2) + 1):
                if (seq1[i - 1] == seq2[j - 1]):
                    d = 0
                else:
                    d = 2
                matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)
        return matrix[len(seq1)][len(seq2)]


    # Load steps info for the corresponding dataset
    label_bank_path = os.path.join('Datasets', cfg.DATASET.NAME, 'label_bank.json')
    label_bank = read_json(label_bank_path)
    # label_bank = read_json('Datasets/COIN-SV/label_bank.json')
    # label_bank = read_json('Datasets/Diving48-SV/label_bank.json')
    # label_bank = read_json('Datasets/CSV/label_bank.json')

    # Calcualte wdr
    labels = torch.tensor(np.array(labels1) == np.array(labels2))
    m_dists = preds[labels]
    um_dists = []
    for i in range(labels.size(0)):
        label = labels[i]
        if not label:
            # unmatched pair
            # NormL2 dist / edit distance
            um_dists.append(preds[i] / compute_edit_dist(label_bank[labels1[i]], label_bank[labels2[i]]))

    return torch.tensor(um_dists).mean() / m_dists.mean()


def eval():

    model = CAT(num_class=cfg.DATASET.NUM_CLASS,
                num_clip=cfg.DATASET.NUM_CLIP,
                dim_embedding=cfg.MODEL.DIM_EMBEDDING,
                pretrain=cfg.MODEL.PRETRAIN,
                dropout=cfg.TRAIN.DROPOUT,
                use_TE=cfg.MODEL.TRANSFORMER,
                use_SeqAlign=cfg.MODEL.ALIGNMENT,
                freeze_backbone=cfg.TRAIN.FREEZE_BACKBONE).to(device)

    if args.model_path == None:
        model_path = os.path.join(args.root_path, 'save_models')
    else:
        model_path = args.model_path

    start_time = time.time()

    if os.path.isdir(model_path):
        # Evaluate models
        logger.info('To evaluate %d models in %s' % (len(os.listdir(model_path)) - args.start_epoch + 1, model_path))

        best_auc = 0
        best_wdr = 0    # wdr of the model with best auc
        best_model_path = ''

        model_paths = os.listdir(model_path)
        try:
            model_paths.remove('.DS_Store')
            model_paths.remove('._.DS_Store')
        except:
            pass
        model_paths.sort(key=lambda x: int(x[6:-4]))

        for path in model_paths:
            if int(path[6:-4]) < args.start_epoch:
                continue
            checkpoint = torch.load(os.path.join(model_path, path))
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            auc, wdr = eval_one_model(model, args.dist)
            logger.info('Model is %s, AUC is %.4f, wdr is %.4f' % (os.path.join(model_path, path), auc, wdr))

            if auc > best_auc:
                best_auc = auc
                best_wdr = wdr
                best_model_path = os.path.join(model_path, path)

        logger.info('*** Best models is %s, Best AUC is %.4f, Best wdr is %.4f ***' % (best_model_path, best_auc, best_wdr))
        logger.info('----------------------------------------------------------------')

    elif os.path.isfile(model_path):
        # Evaluate one model
        logger.info('To evaluate 1 models in %s' % (model_path))
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        auc, wdr = eval_one_model(model, args.dist)
        logger.info('Model is %s, AUC is %.4f' % (model_path, auc))

    else:
        logger.info('Wrong model path: %s' % model_path)
        exit(-1)

    end_time = time.time()
    duration = end_time - start_time

    hour = duration // 3600
    min = (duration % 3600) // 60
    sec = duration % 60

    logger.info('Evaluate cost %dh%dm%ds' % (hour, min, sec))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='configs/eval_resnet_config.yml', help='config file path')
    parser.add_argument('--root_path', default=None, help='path to load models and save log')
    parser.add_argument('--model_path', default=None, help='path to load one model')
    parser.add_argument('--log_name', default='eval_log', help='log name')
    parser.add_argument('--start_epoch', default=1, type=int, help='index of the first evaluated epoch while evaluating epochs')
    parser.add_argument('--dist', default='NormL2')


    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    cfg = get_cfg_defaults()
    if args.config:
        cfg.merge_from_file(args.config)

    setup_seed(cfg.TRAIN.SEED)
    use_cuda = cfg.TRAIN.USE_CUDA and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    logger_path = os.path.join(args.root_path, 'logs')
    logger = setup_logger('Sequence Verification', logger_path, args.log_name, 0)
    logger.info('Running with config:\n{}\n'.format(cfg))

    test_loader = load_dataset(cfg)
    eval()