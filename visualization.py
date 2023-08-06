'''
This file is used to save the output image
'''
from sklearn.metrics import precision_recall_fscore_support as prfs
import torch.utils.data
from utils.parser import get_parser_with_args
from utils.helpers import get_test_loaders
from utils.helpers import (get_loaders, get_criterion,
                           load_model, initialize_test_metrics, get_mean_metrics,
                           set_test_metrics)
import os
from tqdm import tqdm
import cv2
from utils.losses import hybrid_loss
import logging


if not os.path.exists('./output_img'):
    os.mkdir('./output_img')

parser, metadata = get_parser_with_args()
opt = parser.parse_args()

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_loader = get_test_loaders(opt, batch_size=1)

path = './tmp/checkpoint_epoch_45.pt'    # the path of the model
model = torch.load(path)

model.eval()
index_img = 0

test_metrics = initialize_test_metrics()

with torch.no_grad():
    tbar = tqdm(test_loader,unit='batch')

    for batch_img1, batch_img2, labels in tbar:
        batch_img1 = batch_img1.float().to(dev)
        batch_img2 = batch_img2.float().to(dev)
        labels = labels.long().to(dev)
        cd_preds = model(batch_img1, batch_img2)
        cd_preds = cd_preds[-1]
        _, cd_preds = torch.max(cd_preds, 1)

        # 统计accuracy、recall、precision、F1
        cd_corrects = (100 *
                       (cd_preds.squeeze().byte() == labels.squeeze().byte()).sum() /
                       (labels.size()[0] * (opt.patch_size ** 2)))

        cd_val_report = prfs(labels.data.cpu().numpy().flatten(),
                             cd_preds.data.cpu().numpy().flatten(),
                             average='binary',
                             zero_division=0,
                             pos_label=1)

        val_metrics = set_test_metrics(test_metrics,
                                  cd_corrects,
                                  cd_val_report,
                                  )
        mean_val_metrics = get_mean_metrics(val_metrics)

        if True:
            # 输出预测结果
            cd_preds = cd_preds.data.cpu().numpy()
            cd_preds = cd_preds.squeeze() * 255

            file_path = './output_img/' + str(index_img).zfill(5)
            cv2.imwrite(file_path + '.png', cd_preds)

            index_img += 1

        # clear batch variables from memory
        del batch_img1, batch_img2, labels

    logging.basicConfig(level=logging.INFO)
    logging.info(" TEST METRICS" + str(mean_val_metrics))