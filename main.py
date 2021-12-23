import os
import random
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader

import option
from model import Model
from dataset import Dataset
from train import train
from test import test
from utils import Visualizer


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 并行gpu


if __name__ == '__main__':
    setup_seed(int(2333))  # 1577677170  2333
    viz = Visualizer(env='DeepMIL')

    args = option.parse_args()
    device = torch.device('cuda')

    model = Model(args.feature_size)
    model = model.to(device)
    for name, value in model.named_parameters():
        print(name, value.shape)

    train_ndataset = Dataset(args, test_mode=False, is_normal=True)
    train_adataset = Dataset(args, test_mode=False, is_normal=False)
    test_dataset = Dataset(args, test_mode=True)

    train_nloader = DataLoader(
        train_ndataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    train_aloader = DataLoader(
        train_adataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.00005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    auc = test(test_loader, model, args, viz, device)
    print(f'\nEpoch 0/{args.max_epoch}: auc: {auc}')

    for epoch in range(args.max_epoch):
        train(
            train_nloader,
            train_aloader,
            model,
            args.batch_size,
            optimizer,
            viz,
            args.plot_freq,
            device,
        )

        scheduler.step()

        torch.save(
            model.state_dict(),
            f'./ckpt/{args.feat_extractor}_{args.modality}_{epoch + 1}_.pkl',
        )

        auc = test(test_loader, model, args, viz, device)
        print(f'\nEpoch {epoch + 1}/{args.max_epoch}: auc: {auc}')
