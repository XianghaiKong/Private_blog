---
title: Onedl-MMSegmentation在5090上的环境配置及跑通自己的数据集
date: 2026-04-04 15:11:02
tags:
---

## 5090 深度学习环境配置

- CUDA: 12.8
- pytorch: 2.8.0
- python: 3.10.20

由于清华等国内镜像源没有 GPU 版本的 Pytorch，所以建议保留一个只有上述内容的虚拟环境，然后每次新开项目的时候克隆：

```bash
conda create -n xxx --clone xxx
```

然而由于 5090 使用全新架构，CUDA 最低也要 12.8 版本，pytorch 可选对应的适配版本。

由于 openmmlab 已经两年没有再维护更新 mmsegmentation，
即使是最新的 1.x 版本也难以支持目前的最新环境，且仅支持 Pytorch 1.x 版本，手动去改配置、从源码编译 mmcv 也可以成功，详细见 [mmcv issue #3283](https://github.com/open-mmlab/mmcv/issues/3283) 中的讨论即可。

但是我发现了一个团队正在重整当年 openmmlab 不再维护的代码：

[onedl-mmsegmentation](https://github.com/VBTI-development/onedl-mmsegmentation)

此版本可以适配当前最新的 Pytorch 2.x。

当然要注意官方主页的这两句话（不知道为什么这两句最关键的没有同步更新到具体的库中）：

<p>{% asset_img 1.png "官方说明: onedl-mmsegmentation 重要提示" %}</p>

*图 1：官方主页中的关键说明。*

## 完整的配置过程

```bash
conda create -n mmseg python=3.10 -y
conda activate mmseg

# --index-url 参数是此步骤成功的关键
pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -U onedl-mim
mim install onedl-mmengine

mim install onedl-mmcv==2.3.2
```

⭐️关键⭐️ 此处如果输出的是使用源代码包的安装日志如下：

<p>{% asset_img 2.png "源码包安装日志: mmcv 安装日志" %}</p>

*图 2：当出现源码包安装日志时，说明预构建包不匹配。*

如果你的输出是下面这种情况，则说明你的 pytorch、cuda 以及 python 版本没有严格对应的预构建包。这种情况下即使此时输出了安装成功，实际上也还是没有编译 mmcv。后续测试案例的时候会报错：

```text
No module named 'mmcv._ext'
```

此时又需要从源代码构建 onedl-mmcv 的办法。

或者直接使用 pip 安装，注意各个版本要严格对应好（主要是 cuda128 + pytorch2.8）：

```bash
pip install onedl-mmcv==2.3.2 -f https://mmwheels.onedl.ai/cu128-torch280/index.html
```

但是这种安装方式依旧报错：

```text
No module named 'mmcv._ext'
```

在 onedl-mmsegmentation 的 repo 提交了 issue，得到了[回复](https://github.com/VBTI-development/onedl-mmsegmentation/issues/21#issuecomment-4186971904)。具体来说开发者要求 Python 版本为 3.10 或 3.12，cuda 和 torch 版本跟我目前的说的一致，并提供了如下指令：

```bash
pip install onedl-mmcv==2.3.2 -f https://mmwheels.onedl.ai/cu128-torch280/index.html --only-binary=onedl-mmcv -v
```

接下来打算 clone 一个 Python 3.10 + CUDA 12.8 + Pytorch 2.8.0 的新环境，然后再重新 clone 一个 oneDL-mmsegmentation 项目来试一下上述指令能否成功。报错如下：

```text
Could not find a version that satisfies the requirement onedl-mmcv==2.3.2(from versions: none)
No matching distribution found for onedl-mmcv==2.3.2
```

这说明压根没找到匹配的预编译包，但是我的版本是明确对应的 python3.10 + cuda12.8 + torch2.8.0。这在他们 [mmcv 的官方安装文档](https://onedl-mmcv.readthedocs.io/en/latest/get_started/installation.html) 里也写了，并且能直接找到放预编译包的地址。

<p>{% asset_img 3.png "" %}</p>

*图 3：*

<p>{% asset_img 4.png "" %}</p>

*图 4：*

最终直接使用预编译包的直接地址去安装 mmcv 成功，即在图 4 中右键相应的 wheel 文件复制链接地址，然后：

```bash
pip install https://mmwheels-bucket.onedl.ai/cu128-torch280/onedl-mmcv/onedl_mmcv-2.3.3-cp310-cp310-manylinux_2_34_x86_64.whl
```

### 解决方案 2：从源代码编译 mmcv

这里最好直接把 mmcv 克隆到整个 mmsegmentation 项目下：

```bash
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout v2.1.0  # or your needed version

# Set CUDA environment
export FORCE_CUDA=1
export MMCV_WITH_OPS=1

# Compile and install
pip install -r requirements.txt
python setup.py build_ext --inplace
pip install -e .
```

安装 mmcv 成功后不要忘记从源代码安装 mmsegmentation！

```bash
git clone -b main https://github.com/vbti-development/onedl-mmsegmentation.git
cd onedl-mmsegmentation
pip install -v -e .
```

## 全面测试安装是否成功

接下来测试是否安装成功可以跑通：

```bash
cd ..  # 返回 mmsegmentation 主目录
mim download mmsegmentation --config pspnet_r50-d8_4xb2-40k_cityscapes-512x1024 --dest .

python demo/image_demo.py demo/demo.png configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth --device cuda:0 --out-file result.jpg
```

如果出现了 `result.jpg` 说明全部成功！

注意此时配置好环境后，就不能更改整个项目文件夹的名字，因为是从源码编译的，即 onedl-mmsegmentation。

接下来先在自带支持的数据集 ADE20K 上，跑通 Segformer 的训练与测试。先从官方文档给的链接下载 ADE20K 数据集，放到整个项目下的 `data/ade` 下，其余不用更改，内部的路径是对的。

```bash
conda activate mmseg

# 开始训练，并通过 --cfg-options 覆盖默认设置的 batch size 和 worker 数量
# 配置文件名的含义：用 mit-b0 的 backbone，8 张 gpu，每张 gpu 的 batch size 为 2，160k 次迭代次数，515x512 的图片尺寸
# 训练 batch=16（5090 其实可以开更大），验证 batch=1，数据预处理同时进行 8 个
python tools/train.py configs/segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512.py \
    --work-dir work_dirs/segformer_mit-b0_ade20k \
    --cfg-options train_dataloader.batch_size=16 val_dataloader.batch_size=1 train_dataloader.num_workers=8

# 或者如下更少次数的命令，更多的用法后续再去探索
python tools/train.py configs/segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512.py \
    --work-dir work_dirs/segformer_mit-b0_ade20k_fast \
    --cfg-options train_dataloader.batch_size=16 \
                  val_dataloader.batch_size=1 \
                  train_dataloader.num_workers=8 \
                  train_cfg.max_iters=40000 \
                  train_cfg.val_interval=2000
```

实时查看显卡和 CPU 状态：

```bash
# 每秒刷新一次
watch -n 1 nvidia-smi
htop
```

或者用 nvitop 工具查看显卡状态，按 `h` 可查看帮助文档：

```bash
# 装到 base 里，以后方便用
conda activate base
pip install nvitop
nvitop
```

关于测试，注意替换实际权重文件名：

```bash
# 评估精度，并保存预测的彩色分割图到指定文件夹
python tools/test.py configs/segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512.py \
    work_dirs/segformer_mit-b0_ade20k/best_mIoU_iter_xxx.pth \
    --out work_dirs/segformer_mit-b0_ade20k/results \
    --show-dir work_dirs/segformer_mit-b0_ade20k/vis_results
```

## 跑通自定义数据集（SmokeSeg 示例）

接下来说明如何跑通自己的分割数据集。

接下来以开源烟雾分割数据集 [SmokeSeg](https://github.com/LujianYao/FoSp) + 框架自带支持的模型 Segformer_mit-b5 为例子说明。此数据集已被制作成标准的 VOC 格式放在项目下 `data/SmokeSeg` 中。

### 第一步：注册 SmokeSeg 数据集（1.x 版本）

在 MMSegmentation 1.x 中，自定义数据集需要继承自 `BaseSegDataset`。

- 在 `mmseg/datasets/` 目录下新建 `smoke_voc.py`
- 在 `mmseg/datasets/__init__.py` 中导入并暴露此数据集类

### 第二步：构建 SegFormer-B5 模型的训练配置文件

在 `configs/segformer/` 目录下可以创建一个针对 SmokeSeg 的配置文件，比如 `segformer_mit-b5_8xb4-160k_smokeseg-515x512.py`。这里由于工具箱支持 segformer，所以可以直接继承基础的 SegFormer-B0 结构。

### 第三步：开始训练

```bash
python tools/train.py configs/segformer/xxxxx刚刚的配置文件
```

### 第四步：测试

```bash
python tools/test.py 配置文件 权重文件 --show-dir 叠加分割结果图路径（可选）
```
