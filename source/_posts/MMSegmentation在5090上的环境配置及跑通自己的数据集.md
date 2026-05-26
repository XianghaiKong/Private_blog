---
title: 在 RTX 5090 上配置 MMSegmentation 并跑通自定义数据集 (OneDL 版)
date: 2026-04-04 15:11:02
tags:
  - 深度学习
  - MMSegmentation
  - RTX 5090
  - 环境配置
  - 踩坑记录
categories: 深度学习
---

在配置基于 RTX 5090 的深度学习环境时，由于底层硬件架构较新，框架版本的匹配需要特别注意。本文主要记录了在 RTX 5090 上配置 MMSegmentation 环境的过程，以及如何使用自定义数据集跑通基础的训练和测试流程。

## 一、基础环境配置

RTX 5090 对 CUDA 版本有一定要求，经过测试，以下版本组合运行较为稳定：
- **CUDA**: 12.8 (5090 最低支持 12.8 版本的 CUDA)
- **PyTorch**: 2.8.0
- **Python**: 3.10 (建议 3.10.x 或 3.12.x)

> 💡 **个人习惯：建立基础克隆环境**
> 由于目前部分国内镜像源可能还没有完全同步最新的 GPU 版 PyTorch，我通常会先建一个只包含上述核心组件的“基础环境”。后续新建项目时直接克隆这个基础环境，可以避免重复下载和安装。
> ```bash
> conda create -n new_env_name --clone base_5090_env
> ```

---

## 二、框架选择：使用 OneDL-MMSegmentation

官方的 OpenMMLab 库 `mmsegmentation` 已经有较长一段时间未更新，最新的 1.x 版本在适配最新的硬件环境（如 CUDA 12.8）和 PyTorch 2.x 时会遇到不少兼容性问题。虽然可以通过手动修改配置、从源码编译 `mmcv` 来强行适配，但维护成本较高。

在调研后，我选择了第三方维护的复刻版本：**[onedl-mmsegmentation](https://github.com/VBTI-development/onedl-mmsegmentation)**。该库对最新的 PyTorch 2.x 提供了较好的支持。

<p>{% asset_img 1.png "官方说明: onedl-mmsegmentation 重要提示" %}</p>
*图 1：官方主页中的相关兼容性说明。*

---

## 三、完整的安装与配置流程

### 1. 创建环境与安装 PyTorch
```bash
conda create -n mmseg python=3.10 -y
conda activate mmseg

# 注意：需通过官方源安装指定 CUDA 12.8 版本的 PyTorch
pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 2. 安装基础依赖包
```bash
pip install -U onedl-mim
mim install onedl-mmengine
```

### 3. 安装 MMCV

在整个配置过程中，`mmcv` 的安装是比较容易出错的一步。主要有两种方案：

#### 方案 A：使用预编译 Wheel 包安装 (推荐)

最理想的方式是使用 `mim` 自动安装预编译包：
```bash
mim install onedl-mmcv==2.3.2
```

> ⚠️ **编译提示**：
> 如果在终端日志中看到 `Building wheel from source`，说明系统未能匹配到合适的预构建包，正在尝试本地编译。这种情况往往由于缺少 C++ 扩展编译环境，导致后续调用 `mmcv._ext` 时报错。

为了避免编译错误，可以根据 [OneDL-MMCV 官方文档](https://onedl-mmcv.readthedocs.io/en/latest/get_started/installation.html) 寻找对应的预编译包（如 Python 3.10 + CUDA 12.8 + PyTorch 2.8.0）。如果自动匹配失败，建议直接通过获取到的 .whl 链接进行安装：

<p>{% asset_img 3.png "预编译包列表" %}</p>
*图 2：官方发布的预编译包列表。*

<p>{% asset_img 4.png "" %}</p>
*图 3：复制对应版本的下载链接。*

```bash
# 示例：通过复制的 .whl 文件绝对链接直接安装
pip install https://mmwheels-bucket.onedl.ai/cu128-torch280/onedl-mmcv/onedl_mmcv-2.3.3-cp310-cp310-manylinux_2_34_x86_64.whl
```

#### 方案 B：从源代码编译

如果无法使用预编译包，也可以从源码编译，建议将代码克隆到工程的同级目录：

```bash
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout v2.1.0  # 切换到所需的分支

# 配置环境变量以编译 C++ 算子
export FORCE_CUDA=1
export MMCV_WITH_OPS=1

pip install -r requirements.txt
python setup.py build_ext --inplace
pip install -e .
```

### 4. 从源码安装 OneDL-MMSegmentation

确认 `mmcv` 正确安装并能导入 `mmcv._ext` 后，即可安装主框架：

```bash
git clone -b main https://github.com/vbti-development/onedl-mmsegmentation.git
cd onedl-mmsegmentation
pip install -v -e .
```

---

## 四、测试与验证环境

完成上述步骤后，可以使用官方提供的 Demo 进行推理测试，以验证环境是否正常配置：

```bash
cd onedl-mmsegmentation

# 获取测试用的配置和模型权重
mim download mmsegmentation --config pspnet_r50-d8_4xb2-40k_cityscapes-512x1024 --dest .

# 运行单张图片推理
python demo/image_demo.py demo/demo.png \
    configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py \
    pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth \
    --device cuda:0 --out-file result.jpg
```
如果当前目录生成了带有分割蒙版的 `result.jpg`，说明环境的安装配置已经初步完成。

---

## 五、在内置数据集 (ADE20K) 上的训练尝试

在切入自定义数据前，我先尝试用内置的 ADE20K 数据集走了一遍完整的训练和测试流程，以熟悉框架的调用逻辑。

1. **准备数据**：按照官方文档指引，解压 ADE20K 并放至 `data/ade` 目录。
2. **启动训练**：尝试使用 Segformer 模型。得益于 5090 充足的显存，可以将 batch size 适当调大以加速验证。

```bash
conda activate mmseg

# 覆盖默认超参数配置
python tools/train.py configs/segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512.py \
    --work-dir work_dirs/segformer_mit-b0_ade20k \
    --cfg-options train_dataloader.batch_size=16 val_dataloader.batch_size=1 train_dataloader.num_workers=8
```

也可以通过修改迭代参数进行一次快速测试：
```bash
python tools/train.py configs/segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512.py \
    --work-dir work_dirs/segformer_mit-b0_ade20k_fast \
    --cfg-options train_dataloader.batch_size=16 val_dataloader.batch_size=1 train_dataloader.num_workers=8 train_cfg.max_iters=40000 train_cfg.val_interval=2000
```

### 显卡状态监控
为了观察高负载下的硬件表现，可以在训练期间监控显卡状态：
```bash
# 方法 1：系统自带
watch -n 1 nvidia-smi

# 方法 2：使用第三方库 nvitop
conda activate base
pip install nvitop
nvitop
```

### 模型验证
训练完成后，评估精度并输出预测结果：
```bash
python tools/test.py configs/segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512.py \
    work_dirs/segformer_mit-b0_ade20k/best_mIoU_iter_xxx.pth \
    --out work_dirs/segformer_mit-b0_ade20k/results \
    --show-dir work_dirs/segformer_mit-b0_ade20k/vis_results
```

---

## 六、跑通自定义数据集：以 SmokeSeg 为例

结合自己课题，这里以开源的烟雾分割数据集 **[SmokeSeg](https://github.com/LujianYao/FoSp)** 为例，记录了如何将其接入 MMSegmentation 框架。

假设数据已转为标准 VOC 格式并存放在项目目录的 `data/SmokeSeg` 中。

### 1. 注册新数据集
在 MMSegmentation 1.x 规范中，自定义数据集需继承 `BaseSegDataset`。
- 在 `mmseg/datasets/` 下创建 `smoke_voc.py`，编写数据集读取逻辑。
- 在 `mmseg/datasets/__init__.py` 中导入并暴露该模块以便系统调用。

### 2. 编写训练配置文件
在 `configs/segformer/` 目录下新建对应 SmokeSeg 的配置文件，例如 `segformer_mit-b5_8xb4-160k_smokeseg-515x512.py`。
*(注: 可以直接利用框架的 `_base_` 机制继承官方的 SegFormer-B5 配置文件，然后在此基础上仅重写类别数、数据路径等参数。)*

### 3. 执行训练与可视化
调用新配置开始模型训练：
```bash
python tools/train.py configs/segformer/segformer_mit-b5_8xb4-160k_smokeseg-515x512.py
```

训练结束后进行测试并可视化：
```bash
python tools/test.py \
    configs/segformer/segformer_mit-b5_8xb4-160k_smokeseg-515x512.py \
    你的最优权重文件路径.pth \
    --show-dir vis_results_smokeseg/
```

到这里，整套分割库的本地化配置和运行流程基本走通了。接下来就可以在此框架上开展进一步的调参和对比实验。
