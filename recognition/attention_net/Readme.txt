-- Prerequistes:
* Python 3.6
* Tensorflow-gpu >= 1.1
* opencv2
* setproctitle
* matplotlib

--Download pretrained model:

Download : https://drive.google.com/open?id=1PuLCYVG457UOFzWHz4GuerTzWABZR0b6

DETECTION: unzip pixel_link_vgg_4s.zip into ${pixel_link_root}/model/

RECOGNITION: put 0_480000.pth into $attention_net_root$/


-- Usage:

1. Recognize the text from the detected boxes:

python Recognition_yang.py --detection_path $pixel_link_root$/Detection.pkl  --gpus 0

2. The results will be saved in the file: $attention_net_root$/Recognition.pkl


Note: 
If you use this code, please cite:

Liu, Yang, Zhaowen Wang, Hailin Jin, and Ian Wassell. "Synthetically supervised feature learning for scene text recognition." In Proceedings of the European Conference on Computer Vision (ECCV), pp. 435-451. 2018.

Deng, Dan, Haifeng Liu, Xuelong Li, and Deng Cai. "Pixellink: Detecting scene text via instance segmentation." In Thirty-Second AAAI Conference on Artificial Intelligence. 2018.
