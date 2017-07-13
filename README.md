# IMTD17 Datasets
*Imageset of Mobile Traffic Data 2017* (**IMTD17**) is a database of mobile application traffic in MNIST-like format.

## Introduction
**IMTD17** consists of 3 parts:
* Part 1: Training set of 50,000 unlabelled samples
* Part 2: Training set of 1,000 labelled samples
* Part 3: Test set of 10,000 labelled samples

The components of each part are as follows:

| Dataset    | Part 1 | Part 2 | Part 3 |
| :--------- | :----: | :----: | :----: |
| Labeled    | No     | Yes    | Yes    |
| Background | 45,000 | 0      | 9,000  |
| Mobile     | 5,000  | 1,000  | 1,000  |

The mobile traffic is generated from 12 different mobile applications:

|   App    | Version | Label |
| -------- | :-----: | :---: |
| Alipay   | 10.0.15 | 0     |
| Baidu    | 7.17.1  | 1     |
| Bilibili | 5.5.0   | 2     |
| CNTV     | 6.1.5   | 3     |
| JD       | 6.1.0   | 4     |
| Kugou    | 8.8.0   | 5     |
| QQ       | 7.0.0   | 6     |
| QQ Mail  | 5.3.0   | 7     |
| QQ Music | 7.3.1   | 8     |
| Taobao   | 6.7.2   | 9     |
| WeChat   | 6.5.8   | 10    |
| Weibo    | 7.5.1   | 11    |

***Note:*** The label number of background traffic is `12`.

6 files are available in this repository: 

| File Name                            | Description                    | Size (bytes) |
| :----------------------------------- | :----------------------------- | :----------- |
| IMTD17\\s1train-images-idx3-ubyte.gz | Unlabelled training set images | 6,138,712    |
| IMTD17\\s2train-images-idx3-ubyte.gz | Labelled training set images   | 55,204       |
| IMTD17\\s2train-labels-idx1-ubyte.gz | Labelled training set labels   | 572          |
| IMTD17\\test-images-idx3-ubyte.gz    | Labelled test set images       | 1,230,388    |
| IMTD17\\test-labels-idx1-ubyte.gz    | Labelled test set labels       | 1,489        |
| imtd.py                              | **IMTD17** helper functions module | 7,975        |
