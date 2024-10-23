# DHMNet：

## DHM-Net: Deep Hypergraph Modeling for Robust Feature Matching

## DHMNet implementation：
  
  Pytorch implementation of DHMNet

**Abstract**:

## Requirements:
  ```Please use Python 3.6, opencv-contrib-python (3.4.0.12) and Pytorch (>= 1.1.0). Other dependencies should be easily installed through pip or conda.```

## Explanation:
  ```If you need YFCC100M and SUN3D datasets, You can visit the code at https://github.com/zjhthu/OANet.git. We have uploaded the main code on 'core' folder.```

## Preparing Data：
  Please follow their instructions to download the training and testing data.
  ```
  bash download_data.sh raw_data raw_data_yfcc.tar.gz 0 8 ## YFCC100M
  tar -xvf raw_data_yfcc.tar.gz
  ```
  ```
  bash download_data.sh raw_sun3d_test raw_sun3d_test.tar.gz 0 2 ## SUN3D
  tar -xvf raw_sun3d_test.tar.gz
  bash download_data.sh raw_sun3d_train raw_sun3d_train.tar.gz 0 63
  tar -xvf raw_sun3d_train.tar.gz
  ```
  After downloading the datasets, the initial matches for YFCC100M and SUN3D can be generated as following. Here we provide descriptors   for SIFT (default), ORB, and SuperPoint.
  ```
  cd dump_match
  python extract_feature.py
  python yfcc.py
  python extract_feature.py --input_path=../raw_data/sun3d_test
  python sun3d.py
  ```



## Citing DHMNet
If you find the DHMNet code useful, please consider citing
 ```@article{chen2024dhm,
  title={DHM-Net: Deep Hypergraph Modeling for Robust Feature Matching},
  author={Chen, Shunxing and Xiao, Guobao and Guo, Junwen and Wu, Qiangqiang and Ma, Jiayi},
  journal={IEEE Transactions on Image Processing},
  year={2024},
  publisher={IEEE}
} ```
