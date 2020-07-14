import torch.utils.data as data
import numpy as np
from utils import process_feat
import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')


class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False):
        self.modality = args.modality
        self.is_normal = is_normal
        if test_mode:
            self.rgb_list_file = args.test_rgb_list
        else:
            self.rgb_list_file = args.rgb_list
        self.tranform = transform
        self.test_mode = test_mode
        self._parse_list()

    def _parse_list(self):

        self.list = list(open(self.rgb_list_file))
        if self.test_mode is False:
            if self.is_normal:
                self.list = self.list[810:]
            else:
                self.list = self.list[:810]

    def __getitem__(self, index):

        features = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)

        if self.tranform is not None:
            features = self.tranform(features)
        if self.test_mode:
            # name = os.path.basename(self.list[index].strip('\n'))
            return features
        else:
            features = process_feat(features, 32)
            return features

    def __len__(self):
        return len(self.list)