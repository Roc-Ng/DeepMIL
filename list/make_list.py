import numpy as np
import os
import glob


'''
FOR UCF CRIME
'''
root_path = '/media/peng/Samsung_T5/c3d_features/train/RGB'
dirs = os.listdir(root_path)
print(dirs)
with open('ucf-c3d.list', 'w+') as f:
    normal = []
    for dir in dirs:
        files = sorted(glob.glob(os.path.join(root_path, dir, "*.npy")))
        for file in files:
            if '__0' in file:  ## comments
                if 'Normal_' in file:
                    normal.append(file)
                else:
                    newline = file+'\n'
                    f.write(newline)
    for file in normal:
        newline = file+'\n'
        f.write(newline)
