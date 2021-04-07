# DeepMIL Pytorch Version

Unofficial implemention of "Real-world Anomaly Detection in Surveillance Videos" CVPR2018

The feature extractor is here: https://github.com/DavideA/c3d-pytorch

*To achieve better performance, we suggest use I3D features rather than C3D features.*

---

we have released I3D features of UCF-Crime, which can be downloaded from: https://stuxidianeducn-my.sharepoint.com/:f:/g/personal/pengwu_stu_xidian_edu_cn/EvYcZ5rQZClGs_no2g-B0jcB4ynsonVQIreHIojNnUmPyA?e=xNrGxc

**where we oversample each video frame with the "10-crop" augment, "10-crop" means cropping images into the center, four corners, and their mirrored counterparts. *__0.npy* is the center, *__1.npy* ~ *__4.npy* is the corners, and *__5.npy* ~ *__9.npy* is the mirrored counterparts.**

---

**The file of ground truth is *list/gt-ucf.npy*, and the order of ground truth is the same as in *list/ucf-c3d.list*.**

---

- **How to train**

  1. download or extract the features.
  2. use *make_list.py* and *make_list-test.py* in the *list* folder to generate the train and test list.
  3. change the parameters in *option.py*
  4. run *main.py*

- **How to test**

  run *infer.py* and the model is in the ckpt folder.

---

We also released a audio-visual violence dataset named XD-Violence (ECCV2020), the project website is here: https://roc-ng.github.io/XD-Violence/ . We also have released the I3D and VGGish features of our dataset. 

Thanks for your attention!
