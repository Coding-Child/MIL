import gc

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import label_binarize

from util.metric import calculate_auroc, calculate_f1_score


def evaluator(model, data_loader, criterion_bag, criterion_instance):
    valid_preds = list()
    valid_labels = list()

    total_instance_loss = 0.0
    total_bag_loss = 0.0
    total_loss = 0.0

    model.eval()
    with torch.no_grad():
        with tqdm(total=len(data_loader), desc='Validation') as pbar:
            for i, (img, label, pseudo_label) in enumerate(data_loader):
                img = img.cuda()
                label = label.cuda()
                pseudo_label = pseudo_label.cuda()

                instance_logits, bag_logits = model(img)

                if isinstance(criterion_bag, nn.BCEWithLogitsLoss):
                    bag_loss = criterion_bag(bag_logits, label.float())
                    instance_loss = criterion_instance(instance_logits, pseudo_label.float())
                    valid_preds.extend(F.sigmoid(bag_logits).detach().cpu().numpy())
                else:
                    bag_loss = criterion_bag(bag_logits, label)
                    instance_loss = criterion_instance(instance_logits, pseudo_label)
                    valid_preds.extend(F.softmax(bag_logits).detach().cpu().numpy())

                loss = (bag_loss * 0.8) + (instance_loss * 0.2)

                total_instance_loss += instance_loss
                total_bag_loss += bag_loss
                total_loss += loss

                valid_labels.extend(label.detach().cpu().numpy())

                pbar.update(1)
                pbar.set_postfix_str(f"Instance Loss: {instance_loss.item():.4f} | Bag Loss: {bag_loss.item():.4f}")

                del img, label, pseudo_label, bag_logits, instance_logits, instance_loss, bag_loss, loss
                gc.collect()
                torch.cuda.empty_cache()

    valid_labels = label_binarize(valid_labels, classes=np.unique(valid_labels))
    valid_labels = np.array(valid_labels).ravel()
    valid_preds = np.array(valid_preds)

    is_binary = isinstance(criterion_bag, nn.BCEWithLogitsLoss)
    auc = calculate_auroc(is_binary, valid_labels, valid_preds)
    f1_score = calculate_f1_score(is_binary, valid_labels, valid_preds)

    avg_instance_loss = total_instance_loss / len(data_loader)
    avg_bag_loss = total_bag_loss / len(data_loader)
    avg_loss = total_loss / len(data_loader)

    del valid_labels, valid_preds
    gc.collect()
    torch.cuda.empty_cache()

    return auc, f1_score, avg_instance_loss, avg_bag_loss, avg_loss


def test(model, test_loader, criterion_bag, criterion_instance, save_path):
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(torch.load(save_path + '/final_model.pt'))
    else:
        model.load_state_dict(torch.load(save_path + '/final_model.pt'))

    final_test_auc, final_test_f1, final_instance_loss, final_bag_loss, _ = evaluator(model,
                                                                                      test_loader,
                                                                                      criterion_bag,
                                                                                      criterion_instance)

    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(torch.load(save_path + '/best_model.pt'))
    else:
        model.load_state_dict(torch.load(save_path + '/best_model.pt'))

    best_test_auc, best_test_f1, best_instance_loss, best_bag_loss, _ = evaluator(model,
                                                                                  test_loader,
                                                                                  criterion_bag,
                                                                                  criterion_instance)

    print(f'Best Model AUC: {best_test_auc * 100:.2f}% & Final Model AUC: {final_test_auc * 100:.2f}%')
    print(f'Best Model F1: {best_test_f1} & Final Model F1: {final_test_f1}')
    print(f'Best Model Instance Loss: {best_instance_loss:.4f} & Final Model Instance Loss: {final_instance_loss:.4f}')
    print(f'Best Model Bag Loss: {best_bag_loss:.4f} & Final Model Bag Loss: {final_bag_loss:.4f}')
