# SVIP: Sequence VerIfication for Procedures in Videos
This repo is the official implementation of our CVPR 2022 paper: [*SVIP: Sequence VerIfication for Procedures in Videos*](https://arxiv.org/abs/2112.06447).

<image src="imgs/task.png" width="700">

---
### Getting Started
#### Prerequisites
- python 3.6
- pytorch 1.7.1
- cuda 10.2

#### Installation
1. Clone the repo and install dependencies.
    ```bash
    git clone https://github.com/svip-lab/SVIP-Sequence-VerIfication-for-Procedures-in-Videos.git
    cd VIP-Sequence-VerIfication-for-Procedures-in-Videos
    pip install requirements.txt 
    ```
2. Download the pretrained model.

    Link：[here](https://pan.baidu.com/s/1gUqVZRwt2Xq2rg8o5F-1Rg?pwd=2555)
    
    Extraction code：2555

---
### Datasets
Please refer to [here](https://github.com/svip-lab/SVIP-Sequence-VerIfication-for-Procedures-in-Videos/tree/main/Datasets) for detailed instructions.

---
### Training and Evaluation
We have provided the default configuration files for reproducing our results. Try these commands to play with this project. 
    
- For training:
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --config configs/train_resnet_config.yml
    ```
- For evaluation:
    ```bash
    CUDA_VISIBLE_DEVICES=0 python eval.py --config configs/eval_resnet_config.yml --root_path [model&log folder] --dist [L2/NormL2] --log_name [xxx]
    ```
    Note that we use **L2** distance while evaluating on COIN-SV, otherwise **NormL2**.

---
### Trained Models
We provide checkpoints for each dataset trained with this *re-organized* codebase. 

`Notice`: The reproduced performances are occassionally higher or lower (within a reasonable range) than the results reported in the paper.

<table>
    <tr>
    <th>Dataset</th><th>Split</th><th>Papar</th><th>Reproduce</th><th>ckpt</th>
    </tr>
    <tr>
        <td rowspan="2">COIN-SV</td>
        <td>val</td>
        <td>56.81, 0.4005</td><td>58.27, 0.4667</td><td rowspan="2"><a href="https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/qianych_shanghaitech_edu_cn/EV1sUUwj2qhOhjDUZjxH_MIBwAhtOmu-aj94oA5Ymjo3OQ?e=iwgmER">here</a></td>
    </tr>
    <td>test</td><td>51.13, 0.4098</td><td>51.55, 0.4658</td>
    <tr>
        <td rowspan="2">Diving48-SV</td>
        <td>val</td>
        <td>91.91, 1.0642</td><td>91.69, 1.0928</td><td rowspan="2"><a href="https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/qianych_shanghaitech_edu_cn/EemkvNVloT1Bl4GLwHyB9AsBYr1nA8CExP7AJO2mUiwQog?e=WirdVh">here</a></td>
    </tr>
    <td>test</td><td>83.11, 0.6009</td><td>84.28, 0.6193</td>
    <tr>
        <td>CSV</td>
        <td>test</td>
        <td>83.02, 0.4193</td><td>82.88, 0.4474</td><td><a href="https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/qianych_shanghaitech_edu_cn/EQTnwW6smH5OgjGpV3eOel4BCir7gBdnqs7nHj1WOS-Z3A?e=ZnNFP7">here</a></td>
    </tr>
</table>


---
### Citation
If you find this repo helpful, please cite our paper:
```
@inproceedings{qian2022svip,
  title={SVIP: Sequence VerIfication for Procedures in Videos},
  author={Qian, Yicheng and Luo, Weixin and Lian, Dongze and Tang, Xu and Zhao, Peilin and Gao, Shenghua},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={19890--19902},
  year={2022}
}
```
