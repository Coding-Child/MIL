import os
import gc
import random
import logging
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.backends.cudnn as cudnn

from trainer.ModelEvaluator import evaluator


instance_loss_arr = list()
bag_loss_arr = list()
lr_per_step = list()

train_logger = logging.getLogger('train_logger')
train_logger.setLevel(logging.INFO)
train_handler = logging.FileHandler('train.log', mode='w')
train_handler.setFormatter(logging.Formatter('%(message)s'))
train_logger.addHandler(train_handler)

valid_logger = logging.getLogger('valid_logger')
valid_logger.setLevel(logging.INFO)
valid_handler = logging.FileHandler('valid.log', mode='w')
valid_handler.setFormatter(logging.Formatter('%(message)s'))
valid_logger.addHandler(valid_handler)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)


def trainer(model, num_epochs, train_loader, dev_loader, criterion_bag, criterion_instance,
            optimizer, scheduler=None, save_path='../model_ckpt'):
    min_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}') as pbar:
            for i, (img, label, pseudo_label) in enumerate(train_loader):
                img = img.cuda()
                label = label.cuda()
                pseudo_label = pseudo_label.cuda()

                bag_logits, instance_logits = model(img)

                if not isinstance(criterion_bag, nn.BCEWithLogitsLoss):
                    bag_loss = criterion_bag(bag_logits, label)
                    instance_loss = criterion_instance(instance_logits, pseudo_label)
                else:
                    bag_loss = criterion_bag(bag_logits, label.float())
                    instance_loss = criterion_instance(instance_logits, pseudo_label.float())

                optimizer.zero_grad()
                loss = (bag_loss * 0.5) + (instance_loss * 0.5)
                loss.backward()
                optimizer.step()

                if isinstance(model, nn.DataParallel) and isinstance(model, nn.DataParallel):
                    clip_grad_norm_(model.module.parameters(), max_norm=1.0)
                else:
                    clip_grad_norm_(model.parameters(), max_norm=1.0)

                if scheduler is not None:
                    scheduler.step()

                global instance_loss_arr, bag_loss_arr, lr_per_step
                instance_loss_arr.append(instance_loss)
                bag_loss_arr.append(bag_loss)

                pbar.update(1)
                pbar.set_postfix_str(f"Instance Loss: {instance_loss.item():.4f} | Bag Loss: {bag_loss.item():.4f}")

                train_logger.info(
                    f"Step {i + 1} of epoch {epoch + 1}\nInstance Loss: {instance_loss.item():.4f}\nBag Loss: {bag_loss.item():.4f}\nTotal Loss: {loss.item()}")

                del img, label, pseudo_label, bag_logits, instance_logits, instance_loss, bag_loss, loss
                gc.collect()
                torch.cuda.empty_cache()

        valid_auc, valid_f1, valid_instance_loss, valid_bag_loss, valid_loss = evaluator(model,
                                                                                         dev_loader,
                                                                                         criterion_bag,
                                                                                         criterion_instance,
                                                                                         )

        valid_logger.info(
            f"Epoch {epoch + 1}\nValidation AUC: {valid_auc:.4f}\nValidation F1 Score: {valid_f1:.4f}\nValidation Instance Loss: {valid_instance_loss.item():.4f}\nValidation Bag Loss: {valid_bag_loss.item():.4f}\nValidation Total Loss: {valid_loss.item():.4f}")

        if valid_loss < min_loss:
            print(f"Loss decreased! Save the model")
            os.makedirs('model_check_point', exist_ok=True)

            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), save_path + "/best_model.pt")
            else:
                torch.save(model.state_dict(), save_path + "/best_model.pt")

            min_loss = valid_loss

    if isinstance(model, nn.DataParallel) and isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), save_path + "/final_model.pt")
    else:
        torch.save(model.state_dict(), save_path + "/final_model.pt")

    return instance_loss_arr, bag_loss_arr, lr_per_step
