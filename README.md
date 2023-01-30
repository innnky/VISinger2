# VISinger2

本仓库将VISinger2对接DiffSinger社区，兼容DiffSinger社区nomidi格式数据集、ds工程文件。相比于DiffSinger，本模型有着极快的合成速度，但不使用预训练模型情况下训练速度相对较慢，模型音质上限也低于DiffSinger

目前训练、推理代码还不是很易用，之后会逐步进行完善

## 数据集准备
先按照DiffSinger nomidi格式制作数据集，放入data目录下
+ 高质量数据集制作可以参照[DiffSinger数据集教程](https://www.yuque.com/sunsa-i3ayc/sivu7h/dx9xof9k1dg305aq) 

[//]: # (+ 低质量数据追求省事可以使用[自动化数据集制作脚本]&#40;https://github.com/innnky/audio-preprocessing-scripts&#41; （目前除了mfa部分基本可以做到一键完成）)
```shell
data
├───speaker0
│   └───raw
│        ├──wavs
│        └──transcriptions.txt
└───speaker1
    └───raw
         ├──wavs
         └──transcriptions.txt
```
之后依次执行
```shell
# 调整文件夹结构
python prepare_multispeaker.py
# 生成mel与pitch
python preprocess.py
# 生成多说话人配置
python preprocess_multispeaker.py
# 之后将上一部生成的spk2id粘贴到配置文件egs/visinger2/config.json中
```
## 训练
```shell
cd egs/visinger2
bash bash/train.sh 0
```
## 推理
修改 ds_inference.py 中ds工程、说话人、模型路径

python ds_inference.py
