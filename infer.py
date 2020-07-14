from torch.utils.data import DataLoader

import torch
import os

from model import Model
from dataset import Dataset
from test import test
import option
from utils import Visualizer
viz = Visualizer(env='DeepMIL', use_incoming_socket=False)
torch.set_default_tensor_type('torch.cuda.FloatTensor')

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    args = option.parser.parse_args()
    device = torch.device("cuda")  # 将torch.Tensor分配到的设备的对象

    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=1, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    model = Model(args.feature_size)
    for name, value in model.named_parameters():
        print(name)

    model_dict = model.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load('ckpt/deepmil10.pkl').items()})

    auc, ap = test(test_loader, model, args, viz, device)
    print(auc, ap)

