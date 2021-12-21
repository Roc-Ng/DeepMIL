import torch
from torch.utils.data import DataLoader
import option
from model import Model
from dataset import Dataset
from test import test
from utils import Visualizer

if __name__ == '__main__':
    viz = Visualizer(env='DeepMIL')
    args = option.parse_args()
    device = torch.device('cuda')

    test_dataset = Dataset(args, test_mode=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=args.workers,
                             pin_memory=True)

    model = Model(args.feature_size)
    model = model.to(device)
    for name, value in model.named_parameters():
        print(name, value.shape)
    model_dict = model.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load(args.ckpt).items()})

    auc = test(test_loader, model, args, viz, device)
    print(auc)
