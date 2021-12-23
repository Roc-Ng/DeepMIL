import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='DeepMIL')
    parser.add_argument('--feat-extractor', default='i3d', choices=['i3d', 'c3d'])
    parser.add_argument('--feature-size', type=int, default=1024, help='size of feature (default: 2048)')
    parser.add_argument('--modality', default='RGB', help='the type of the input, RGB,AUDIO, or MIX')
    parser.add_argument('--rgb-list', default='list/ucf-i3d.list', help='list of rgb features ')
    parser.add_argument(
        '--test-rgb-list',
        default='list/ucf-i3d-test.list',
        help='list of test rgb features ',
    )
    parser.add_argument('--gt', default='list/gt-ucf.npy', help='file of ground truth ')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='number of instances in a batch of data (default: 16)',
    )
    parser.add_argument('--workers', default=8, help='number of workers in dataloader')
    parser.add_argument('--model-name', default='deepmil', help='name to save model')
    parser.add_argument('--plot-freq', type=int, default=2, help='frequency of plotting (default: 10)')
    parser.add_argument(
        '--max-epoch',
        type=int,
        default=50,
        help='maximum iteration to train (default: 100)',
    )
    parser.add_argument('--ckpt', default='ckpt/i3d_RGB_1_.pkl', help='ckpt for pretrained model')

    return parser.parse_args()
