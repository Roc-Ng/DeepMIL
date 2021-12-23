import numpy as np
import option
from utils import process_feat


def run():
    """Show dataset details
    See:
        Real-World Anomaly Detection in Surveillance Videos
        Sultani W, Chen C, Shah M
        https://openaccess.thecvf.com/content_cvpr_2018/papers/Sultani_Real-World_Anomaly_Detection_CVPR_2018_paper.pdf
    Related text:
        We compute C3D features for every 16-frame video clip followed by l2 normalization. To
        obtain features for a video segment, we take the average of all 16-frame clip features
        within that segment.
        We divide each video into 32 non-overlapping segments and consider each video segment 
        as an instance of the bag.
    """
    BAGS_PER_VIDEO = 32
    FRAMES_PER_CLIP = 16
    args = option.parse_args()

    rgb_list_file_train = args.rgb_list
    list_train = list(open(rgb_list_file_train))
    list_train_normal = list_train[810:]
    list_train_anomaly = list_train[:810]

    rgb_list_file_test = args.test_rgb_list
    list_test = list(open(rgb_list_file_test))

    train_clips_number_normal = 0
    train_clips_number_anomaly = 0
    train_bags_number_normal = 0
    train_bags_number_anomaly = 0

    feature_normal = np.array(np.load(list_train_normal[0].strip('\n')), dtype=np.float32)
    feature_size = feature_normal.shape[1]

    for i in range(len(list_train_normal)):
        feature_normal = np.array(np.load(list_train_normal[i].strip('\n')), dtype=np.float32)
        train_clips_number_normal = train_clips_number_normal + \
            feature_normal.shape[0]
        feature_normal = process_feat(feature_normal, BAGS_PER_VIDEO)
        train_bags_number_normal = train_bags_number_normal + \
            feature_normal.shape[0]

    for i in range(len(list_train_anomaly)):
        feature_anomaly = np.array(np.load(list_train_anomaly[i].strip('\n')), dtype=np.float32)
        train_clips_number_anomaly = train_clips_number_anomaly + \
            feature_anomaly.shape[0]
        feature_anomaly = process_feat(feature_anomaly, BAGS_PER_VIDEO)
        train_bags_number_anomaly = train_bags_number_anomaly + \
            feature_anomaly.shape[0]

    test_clips_number = 0
    for i in range(len(list_test)):
        feature_test = np.array(np.load(list_test[i].strip('\n')), dtype=np.float32)
        test_clips_number = test_clips_number + \
            feature_test.shape[0]

    test_ground_truth = np.load(args.gt)

    print('-' * 10)
    print(f'Feature size: {feature_size}')
    print('-' * 10)
    print(f'Length of train list normal: {len(list_train_normal)}')
    print(f'Length of train list anomaly: {len(list_train_anomaly)}')
    print(f'Length of test list: {len(list_test)}')
    print('-' * 10)
    print(f'Number of train video clips normal: {train_clips_number_normal}')
    print(
        f'Number of train bags normal: {train_bags_number_normal} = {len(list_train_normal)} (videos) * {BAGS_PER_VIDEO} (bags per video)'
    )
    print(f'Number of train video clips anomaly: {train_clips_number_anomaly}')
    print(
        f'Number of train bags anomaly: {train_bags_number_anomaly} = {len(list_train_anomaly)} (videos) * {BAGS_PER_VIDEO} (bags per video)'
    )
    print('-' * 10)
    print(f'Number of test clips: {test_clips_number}')
    print(
        f'Length of test ground truth: {test_ground_truth.shape[0]} (frames) = {test_clips_number} (video clips) * {FRAMES_PER_CLIP} (frames per video clip)'
    )


if __name__ == "__main__":
    run()
