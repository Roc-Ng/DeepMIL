import os
import glob

if __name__ == '__main__':
    root_path = 'D:/137/dataset/VAD/features/UCF_Crime/I3D/Train/RGB/'
    dirs = sorted(glob.glob(os.path.join(root_path, '*')))

    with open('list/ucf-i3d.list', 'w+') as f:
        normal = []
        for dir in dirs:
            files = sorted(glob.glob(os.path.join(root_path, dir, '*.npy')))
            for file in files:
                if '__' not in file:
                    if 'Normal_' in file:
                        normal.append(file)
                    else:
                        newline = file + '\n'
                        f.write(newline)

        for file in normal:
            newline = file + '\n'
            f.write(newline)
