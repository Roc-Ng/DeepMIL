# DeepMIL Pytorch Version
Unofficial implemention of "Real-world Anomaly Detection in Surveillance Videos" CVPR2018

The feature extractor is here: https://github.com/DavideA/c3d-pytorch

we have released I3D features of UCF-Crime, which can be downloaded from: https://stuxidianeducn-my.sharepoint.com/:f:/g/personal/pengwu_stu_xidian_edu_cn/EvYcZ5rQZClGs_no2g-B0jcB4ynsonVQIreHIojNnUmPyA?e=xNrGxc

where we oversample each video frame with the “10-crop” augment, “10-crop” means cropping images into the center, four corners, and their mirrored counterparts. _0.npy is the center, _1~ _4.npy is the corners, and _5 ~ _9 is the mirrored counterparts. 

*To achieve better performance, we suggest use I3D features rather than C3D features.*

---

- **How to train**

  1. download or extract the features.
  2. use *make_list.py* in the *list* folder to generate the training and test list.
  3. change the parameters in option.py 
  4. run *main.py*

- **How to test**

  run *infer.py* and the model is in the ckpt folder.

---

We also released a audio-visual violence dataset named XD-Violence (ECCV2020), the project website is here: https://roc-ng.github.io/XD-Violence/ . We have released the I3D and VGGish features of our dataset as well as the code. 

---
**In order to make training process faster, we suggest use the following code to replace original code in train.py [Line 34]**
```python
model.train()
n_iter = iter(nloader)
a_iter = iter(aloader)
for i in range(30):  # 800/batch_size
    ninput = next(n_iter)
    ainput = next(a_iter)
```

Thanks for your attention!
