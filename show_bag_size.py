import numpy as np
import option
from utils import process_feat


if __name__ == "__main__":
    args = option.parser.parse_args()

    rgb_list_file_train = args.rgb_list
    list_train = list(open(rgb_list_file_train))
    list_normal = list_train[810:]
    list_anomaly = list_train[:810]

    rgb_list_file_test = args.test_rgb_list
    list_test = list(open(rgb_list_file_test))

    total_train_bag_size_normal = 0
    total_train_bag_size_anomaly = 0
    feet_train_bag_size_normal = 0
    feet_train_bag_size_anomaly = 0
    for i in range(len(list_normal)):
        feature_normal = np.array(
            np.load(list_normal[i].strip('\n')), dtype=np.float32)
        total_train_bag_size_normal = total_train_bag_size_normal + \
            feature_normal.shape[0]
        feature_normal = process_feat(feature_normal, 32)
        feet_train_bag_size_normal = feet_train_bag_size_normal + \
            feature_normal.shape[0]

    for i in range(len(list_anomaly)):
        feature_anomaly = np.array(
            np.load(list_anomaly[i].strip('\n')), dtype=np.float32)
        total_train_bag_size_anomaly = total_train_bag_size_anomaly + \
            feature_anomaly.shape[0]
        feature_anomaly = process_feat(feature_anomaly, 32)
        feet_train_bag_size_anomaly = feet_train_bag_size_anomaly + \
            feature_anomaly.shape[0]

    total_test_bag_size = 0
    for i in range(len(list_test)):
        feature_test = np.array(
            np.load(list_test[i].strip('\n')), dtype=np.float32)
        total_test_bag_size = total_test_bag_size + \
            feature_test.shape[0]

    frame_gt_test = np.load(args.gt)

    print('-' * 10)
    print('feature size: {}'.format(args.feature_size))
    print('-' * 10)
    print('train list length normal: {}'.format(len(list_normal)))
    print('train list length anomaly: {}'.format(len(list_anomaly)))
    print('test list length: {}'.format(len(list_test)))
    print('-' * 10)
    print('feet train bag size normal: {}'.format(
        feet_train_bag_size_normal))
    print('feet train bag size anomaly: {}'.format(
        feet_train_bag_size_anomaly))
    print('total train bag size normal: {}'.format(
        total_train_bag_size_normal))
    print('total train bag size anomaly: {}'.format(
        total_train_bag_size_anomaly))
    print('-' * 10)
    print('total test bag size: {}'.format(total_test_bag_size))
    print('frame gt test size: {} (frames) = {} (bags) * {} (frames per bag)'.format(
        frame_gt_test.shape[0], total_test_bag_size, frame_gt_test.shape[0] / total_test_bag_size))
