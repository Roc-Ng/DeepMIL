from torch.utils.data import DataLoader

import torch.optim as optim
import torch
import os
import random
import numpy as np

def setup_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  #并行gpu


setup_seed(int(2333))  # 1577677170  2333

from model import Model
from dataset import Dataset
from train import train
from test import test
import option

from utils import Visualizer


torch.set_default_tensor_type('torch.cuda.FloatTensor')
viz = Visualizer(env='DeepMIL', use_incoming_socket=False)

if __name__ == '__main__':
    args = option.parser.parse_args()
    device = torch.device("cuda")  # 将torch.Tensor分配到的设备的对象

    train_nloader = DataLoader(Dataset(args, test_mode=False, is_normal=True),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    train_aloader = DataLoader(Dataset(args, test_mode=False, is_normal=False),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)

    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=1, shuffle=False,  ####
                              num_workers=args.workers, pin_memory=True)

    model = Model(args.feature_size)
    for name, value in model.named_parameters():
        print(name)

    torch.cuda.set_device(args.gpus)
    model = model.to(device)

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    optimizer = optim.Adam(model.parameters(),
                            lr=args.lr, weight_decay=0.00005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    auc = test(test_loader, model, args, viz, device)
    for epoch in range(args.max_epoch):
        scheduler.step()
        train(train_nloader, train_aloader, model, args.batch_size, optimizer, viz, device)
        if epoch % 1 == 0 and not epoch == 0:
            torch.save(model.state_dict(), './ckpt/'+args.model_name+'{}-i3d.pkl'.format(epoch))
        auc = test(test_loader, model, args, viz, device)
        print('Epoch {0}/{1}: auc:{2}\n'.format(epoch, args.max_epoch, auc))
    torch.save(model.state_dict(), './ckpt/' + args.model_name + 'final.pkl')
