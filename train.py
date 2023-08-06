import datetime
import torch
from sklearn.metrics import precision_recall_fscore_support as prfs
from utils.parser import get_parser_with_args
from utils.helpers import (get_loaders, get_criterion,
                           load_model, initialize_metrics, get_mean_metrics,
                           set_metrics)
import os
import logging
import json
from tensorboardX import SummaryWriter
from tqdm import tqdm
import random
import numpy as np


"""
Initialize Parser and define arguments
"""
parser, metadata = get_parser_with_args()
opt = parser.parse_args()

"""
Initialize experiments log
"""
logging.basicConfig(level=logging.INFO)
# writer = SummaryWriter(opt.log_dir + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

"""
Set up environment: define paths, download data, and set device
"""
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logging.info('GPU AVAILABLE? ' + str(torch.cuda.is_available()))

'''
固定seed，使模型的训练结果保持一致
'''
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(seed=777)


train_loader, val_loader = get_loaders(opt)

"""
Load Model then define other aspects of the model
"""
logging.info('LOADING Model')
model = load_model(opt, dev)   # 加载siam_nested_Unet
# 损失函数
criterion = get_criterion(opt) # hybrid：focal+dice
# 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate) # Be careful when you adjust learning rate, you can refer to the linear scaling rule
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)  #每八次迭代学习率乘0.5

"""
 Set starting values
"""
best_metrics = {'cd_f1scores': -1, 'cd_recalls': -1, 'cd_precisions': -1}
logging.info('STARTING training')
total_step = -1
"""
 Epoch
"""
for epoch in range(opt.epochs):
    train_metrics = initialize_metrics()  # loss 、recall 、learning_rate 字典
    val_metrics = initialize_metrics()

    """
    Begin Training
    """
    model.train()
    logging.info('SET model mode to train!')
    batch_iter = 0
    tbar = tqdm(train_loader,unit='batch')
    """
     step
     batch_iter迭代图片数量、total_step 批次数量
    """
    for batch_img1, batch_img2, labels in tbar:

        tbar.set_description("epoch {} info ".format(epoch) + str(batch_iter) + " -> " + str(batch_iter+opt.batch_size))
        batch_iter = batch_iter+opt.batch_size
        total_step += 1

        # Set variables for training
        batch_img1 = batch_img1.float().to(dev)
        batch_img2 = batch_img2.float().to(dev)
        labels = labels.long().to(dev)         #(batch_size,1,H,W)

        # print('标签', labels)

        # Zero the gradient
        optimizer.zero_grad()

        # Get model predictions, calculate loss, backprop
        cd_preds = model(batch_img1, batch_img2)  #(output1,2,3,4,output)

        # hybrid：focal+dice
        cd_loss = criterion(cd_preds, labels)

        loss = cd_loss
        loss.backward()
        optimizer.step()

        cd_preds = cd_preds[-1]  # 提取深监督的结果output作为预测结果(batch,2,H,W)



        _, cd_preds = torch.max(cd_preds, 1)    # 通道1、通道2的值为两类的预测值,值越大表示为该类的可能性越大,提取通道轴索引01分成两类(batch,1,H,W)

        # print('预测值', cd_preds)

        # Calculate and log other batch metrics
        # 准确率accuracy
        cd_corrects = (100 *
                       (cd_preds.squeeze().byte() == labels.squeeze().byte()).sum() /     # 使用 .byte() 将张量中的元素转换为 uint8 类型的整数，在进行比较时可以减少计算量，提高代码的效率。
                       (labels.size()[0] * (opt.patch_size**2)))                # 预测正确的像素数量/批次总像素数量  百分数
        # print('accuracy',cd_corrects)
        # 返回(precision,recall,F1)
        cd_train_report = prfs(labels.data.cpu().numpy().flatten(),     # 转为一维numpy数组
                               cd_preds.data.cpu().numpy().flatten(),
                               average='binary',  # 表示标签为二进制，只计算pos_label类的精度
                               zero_division=0,   # 分母为0时，返回零
                               pos_label=1)
        # print('PRF',cd_train_report[0],cd_train_report[1],cd_train_report[2])
        # 上述指标加入train_metrics字典
        train_metrics = set_metrics(train_metrics,
                                    cd_loss,
                                    cd_corrects,
                                    cd_train_report,
                                    scheduler.get_last_lr())

        # log the batch mean metrics
        mean_train_metrics = get_mean_metrics(train_metrics)

        # for k, v in mean_train_metrics.items():
        #     writer.add_scalars(str(k), {'train': v}, total_step)   # 添加scalar(标量)到tensorboard中

        # clear batch variables from memory
        del batch_img1, batch_img2, labels

    scheduler.step()
    logging.info("EPOCH {} TRAIN METRICS".format(epoch) + str(mean_train_metrics))

    """
    Begin Validation
    """
    model.eval()
    with torch.no_grad():
        for batch_img1, batch_img2, labels in val_loader:
            # Set variables for training
            batch_img1 = batch_img1.float().to(dev)
            batch_img2 = batch_img2.float().to(dev)
            labels = labels.long().to(dev)

            # Get predictions and calculate loss
            cd_preds = model(batch_img1, batch_img2)

            cd_loss = criterion(cd_preds, labels)

            cd_preds = cd_preds[-1]
            _, cd_preds = torch.max(cd_preds, 1)


            # Calculate and log other batch metrics
            cd_corrects = (100 *
                           (cd_preds.squeeze().byte() == labels.squeeze().byte()).sum() /
                           (labels.size()[0] * (opt.patch_size**2)))

            cd_val_report = prfs(labels.data.cpu().numpy().flatten(),
                                 cd_preds.data.cpu().numpy().flatten(),
                                 average='binary',
                                 zero_division=0,
                                 pos_label=1)

            val_metrics = set_metrics(val_metrics,
                                      cd_loss,
                                      cd_corrects,
                                      cd_val_report,
                                      scheduler.get_last_lr())

            # log the batch mean metrics
            mean_val_metrics = get_mean_metrics(val_metrics)

            # for k, v in mean_train_metrics.items():
            #     writer.add_scalars(str(k), {'val': v}, total_step)

            # clear batch variables from memory
            del batch_img1, batch_img2, labels

        logging.info("EPOCH {} VALIDATION METRICS".format(epoch)+str(mean_val_metrics))

        """
        Store the weights of good epochs based on validation results
        """
        if ((mean_val_metrics['cd_precisions'] > best_metrics['cd_precisions'])
                or
                (mean_val_metrics['cd_recalls'] > best_metrics['cd_recalls'])
                or
                (mean_val_metrics['cd_f1scores'] > best_metrics['cd_f1scores'])):

            # Insert training and epoch information to metadata dictionary
            logging.info('updata the model')
            metadata['validation_metrics'] = mean_val_metrics  # 向metadata中添加新item，更新最佳的验证指标

            # Save model and log
            if not os.path.exists('./tmp'):
                os.mkdir('./tmp')
            with open('./tmp/metadata_epoch_' + str(epoch) + '.json', 'w') as fout:
                json.dump(metadata, fout)

            torch.save(model, './tmp/checkpoint_epoch_'+str(epoch)+'.pt')

            # comet.log_asset(upload_metadata_file_path)
            best_metrics = mean_val_metrics


        print('An epoch finished.')
# writer.close()  # close tensor board
print('Done!')
