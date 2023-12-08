import glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as opt

from torchvision import transforms as T

from model.ViTMIL import ViTMIL
from Dataset.MILDataset import PathologyDataset, collate_fn, data_split
from trainer.ModelTrainer import trainer, seed_everything
from trainer.ModelEvaluator import test
from util.lr_scheduler import CosineWarmupScheduler
from util.augment import RandomRotation, RandomErasing

import matplotlib.pyplot as plt


def main(args):
    seed_everything(42)

    lr = args.lr
    num_heads = args.num_heads
    dropout = args.dropout
    num_epochs = args.num_epochs
    num_classes = args.num_classes
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    train_num_sample = args.num_train_instance
    val_num_sample = args.num_val_instance
    test_num_sample = args.num_test_instance
    csv_root = args.csv_root_dir
    save_path = args.save_path

    model = ViTMIL(num_heads=num_heads, dropout=dropout)
    transform_train = T.Compose([T.RandomHorizontalFlip(p=0.5),
                                 T.Lambda(lambda img: RandomRotation(img)),
                                 T.RandomChoice([RandomErasing(fill_color=(225, 225, 225)),
                                                T.ElasticTransform(alpha=20.0, sigma=4.0, fill=225),
                                                T.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 0.1)),
                                                T.RandomGrayscale(p=0.2),
                                                T.RandomAutocontrast()]),
                                 T.ToTensor()
                                 ])

    transform_test = T.Compose([T.ToTensor()
                                ])

    train_data, val_data, test_data = data_split(csv_root)
    train_dataset = PathologyDataset(df=train_data, num_samples=train_num_sample, transform=transform_train)
    dev_dataset = PathologyDataset(df=val_data, num_samples=val_num_sample, test=True, transform=transform_test)
    test_dataset = PathologyDataset(df=test_data, num_samples=test_num_sample, test=True, transform=transform_test)

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

    scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    num_positive = glob.glob('cancer_data/1/**.jpg')
    num_negative = glob.glob('cancer_data/0/**.jpg')

    num_samples = torch.tensor([len(num_negative), len(num_positive)])
    weights = num_samples.sum() / num_samples
    weights = (weights / weights.sum()).cuda()

    if num_classes == 2:
        criterion_bag = nn.BCEWithLogitsLoss()
    else:
        criterion_bag = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    criterion_instance = nn.BCEWithLogitsLoss()

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
