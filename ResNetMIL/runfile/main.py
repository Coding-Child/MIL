import glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as opt

from torchvision import transforms as T

from model.RTMIL import RTMIL
from Dataset.MILDataset import PathologyDataset, collate_fn
from loss.loss_fn import CenterLoss, MaxMarginLoss, SmoothSVMLoss
from trainer.ModelTrainer import trainer, seed_everything
from trainer.ModelEvaluator import test
from util.lr_scheduler import CosineWarmupScheduler
from util.augment import RandomRotation

import matplotlib.pyplot as plt


def main(args):
    seed_everything(42)

    lr = args.lr
    num_epochs = args.num_epochs
    num_classes = args.num_classes
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    train_num_sample = args.num_train_instance
    val_num_sample = args.num_val_instance
    test_num_sample = args.num_test_instance
    d_model = args.d_model
    num_heads = args.num_heads
    num_layers = args.num_layers
    d_ff = args.d_ff
    drop_prob = args.drop_prob
    csv_root = args.csv_root_dir
    save_path = args.save_path
    pretrained_path = args.pretrained_path
    pretrained = args.pretrained
    margin = args.margin
    loss = args.loss

    model = RTMIL(d_model=d_model,
                  d_ff=d_ff,
                  num_heads=num_heads,
                  num_layers=num_layers,
                  dropout=drop_prob,
                  pretrained=pretrained,
                  path=pretrained_path,
                  num_classes=num_classes)

    transform_train = T.Compose([T.RandomHorizontalFlip(p=0.5),
                                 T.Lambda(lambda img: RandomRotation(img)),
                                 T.ToTensor()
                                 ])

    transform_test = T.Compose([T.ToTensor()
                                ])

    train_dataset = PathologyDataset(csv_file=csv_root, num_samples=train_num_sample, val=False, test=False,
                                     transform=transform_train)
    dev_dataset = PathologyDataset(csv_file=csv_root, num_samples=val_num_sample, val=True, test=False,
                                   transform=transform_test)
    test_dataset = PathologyDataset(csv_file=csv_root, num_samples=test_num_sample, val=False, test=True,
                                    transform=transform_test)

    train_loader = DataLoader(train_dataset,
                              batch_size=train_batch_size,
                              shuffle=True,
                              drop_last=True,
                              collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset,
                            batch_size=train_batch_size,
                            shuffle=False,
                            drop_last=True,
                            collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset,
                             batch_size=test_batch_size,
                             shuffle=False,
                             drop_last=True,
                             collate_fn=collate_fn)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[0, 1])

    model.cuda()

    if isinstance(model, nn.DataParallel):
        optimizer = opt.Adam(model.module.parameters(), lr=lr, betas=(0.9, 0.98), amsgrad=True)
    else:
        optimizer = opt.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), amsgrad=True)

    scheduler = CosineWarmupScheduler(optimizer,
                                      warmup=int((len(train_loader) * num_epochs) * 0.1),
                                      max_iters=int(num_epochs * len(train_loader)))

    num_positive = glob.glob('cancer_data/1/**.jpg')
    num_negative = glob.glob('cancer_data/0/**.jpg')

    num_samples = torch.tensor([len(num_negative), len(num_positive)])
    weights = num_samples.sum() / num_samples
    weights = (weights / weights.sum()).cuda()

    if num_classes == 2:
        criterion_bag = nn.BCEWithLogitsLoss()
    else:
        criterion_bag = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    if loss == 'distance':
        print('Using CenterLoss')
        criterion_instance = CenterLoss(num_classes=num_classes,
                                        feat_dim=d_model,
                                        use_gpu=bool(torch.cuda.is_available()))
    elif loss == 'max':
        print('Using MaxMarginLoss')
        criterion_instance = MaxMarginLoss(margin=margin)
    elif loss == 'smooth':
        print("Using SmoothSVMLoss")
        criterion_instance = SmoothSVMLoss()
    else:
        criterion_instance = nn.BCEWithLogitsLoss()
    #
    instance_loss_step, bag_loss_step, lr_step = trainer(model=model,
                                                         num_epochs=num_epochs,
                                                         train_loader=train_loader,
                                                         dev_loader=dev_loader,
                                                         criterion_bag=criterion_bag,
                                                         criterion_instance=criterion_instance,
                                                         optimizer=optimizer,
                                                         scheduler=scheduler,
                                                         save_path=save_path
                                                         )

    test(model=model,
         test_loader=test_loader,
         criterion_bag=criterion_bag,
         criterion_instance=criterion_instance,
         save_path=save_path
         )

    plt.plot(instance_loss_step, label='instance pred loss')
    plt.plot(bag_loss_step, label='bag pred loss')
    plt.title("Training loss per step")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('Loss.jpg', facecolor='#eeeeee')
    plt.close()

    plt.plot(lr_step)
    plt.title("Learning Rate per step")
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.savefig('Learning_Rate.jpg', facecolor='#eeeeee')
    plt.close()
