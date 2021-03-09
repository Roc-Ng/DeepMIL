import numpy as np
import os
import glob


'''
FOR UCF CRIME
'''
root_path = '/home/wucx/dataset/UCF-Crime/I3D/Test/RGB/'
dirs = os.listdir(root_path)
print(dirs)
with open('list/ucf-i3d-test.list', 'w+') as f:
    normal = []
    for dir in dirs:
        files = sorted(glob.glob(os.path.join(root_path, dir, "*.npy")))
        # print(files)
        for file in files:
            # if '__0' in file:  # WTF
            if '__' not in file:
                '''
                where we oversample each video frame with the “10-crop” augment, 
                “10-crop” means cropping images into the center, four corners, 
                and their mirrored counterparts. _0.npy is the center, _1~ _4.npy is the corners, 
                and _5 ~ _9 is the mirrored counterparts.
                '''
                if 'Normal_' in file:
                    normal.append(file)
                else:
                    newline = file+'\n'
                    f.write(newline)
    # print(normal)
    for file in normal:
        newline = file+'\n'
        f.write(newline)
