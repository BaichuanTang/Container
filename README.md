<div align="center"><img src="assets/logo.png" width="350"></div>
<img src="assets/demo.png" >

## Introduction
本项目是复旦大学管理学院《大数据技术》的期末考核作业，是基于YOLOX框架实现的两阶段目标检测&识别模型，希望借此创造出商业价值。

<img src="assets/git_fig.png" width="1000" >



## Quick Start

<details>
<summary>Installation</summary>

Step1. Install YOLOX.
```shell
git clone git@github.com/BaichuanTang/Container.git
cd YOLOX
pip3 install -U pip && pip3 install -r requirements.txt
pip3 install -v -e .  # or  python3 setup.py develop
```

Step2. Install [pycocotools](https://github.com/cocodataset/cocoapi).

```shell
pip install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

Step3. MySQL数据库

- 安装版本5.7.28的MySQL
- 执行一遍`createTables.ipynb`，建库建表

Step4. 配置QtDesigner和PyQt5

```shell
pip install pyqt5
pip install pyqt5-tools
```

Step5. 安装TensorFlow和PyTorch

```shell
pip install tensorflow_gpu==2.3.0
pip install -r requirements.txt
```

</details>

<details>
<summary>Demo</summary>

Step1. Download a pretrained model from the benchmark table.

Step2. Use either -n or -f to specify your detector's config. For example:

```shell
python tools/demo.py image -n yolox-s -c /path/to/your/yolox_s.pth --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]
```
or
```shell
python tools/demo.py image -f exps/default/yolox_s.py -c /path/to/your/yolox_s.pth --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]
```
Demo for video:
```shell
python realtime.py webcam -n yolox-s  -c yolox_s.pth --path /path/to/your/video --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu --recog assets/simple_model
```

</details>



## Reference
```latex
 @article{yolox2021,
  title={YOLOX: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```
