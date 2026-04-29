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

## 一、RTX 5090 深度学习基础环境配置

针对 RTX 5090 这样基于全新架构的显卡，对基础环境的要求非常严格，核心版本搭配如下：
- **CUDA**: 12.8 (5090 最低要求 12.8 版本的 CUDA)
- **PyTorch**: 2.8.0
- **Python**: 3.10 (建议 3.10.x 或 3.12.x)

> 💡 **建议：环境克隆法**
> 由于国内镜像源（如清华源）尚未完全收录最新的 GPU 版本 PyTorch，建议先创建一个仅包含上述核心组件的“基础虚拟环境”。以后每次新建项目时，直接克隆该环境即可，省时省力。
> ```bash
> conda create -n new_env_name --clone base_5090_env
> ```

---

## 二、框架选择：为什么是 OneDL-MMSegmentation？

官方的 OpenMMLab 已经停止维护更新 `mmsegmentation` 将近两年。即便是最新的 1.x 官方版本，也难以支持目前最新的硬件环境（如 CUDA 12.8）且仅支持 PyTorch 1.x。虽然可以通过手动修改配置、从源码编译 `mmcv` 来强行适配（详见 [mmcv issue #3283](https://github.com/open-mmlab/mmcv/issues/3283)），但过程极其繁琐。

幸运的是，我们找到了一个专门维护、重整 OpenMMLab 废弃代码的第三方团队库：
👉 **[onedl-mmsegmentation](https://github.com/VBTI-development/onedl-mmsegmentation)**

此版本完美适配当前最新的 PyTorch 2.x。不过需要注意其官方主页的这段重要说明（该说明目前仅在主页展示，未在库文件中同步更新）：

<p>{% asset_img 1.png "官方说明: onedl-mmsegmentation 重要提示" %}</p>

*图 1：官方主页中的关键说明。*

---

## 三、完整的安装与配置过程

### 1. 创建基础环境与安装 PyTorch
```bash
conda create -n mmseg python=3.10 -y
conda activate mmseg

# ⚠️ 此处 --index-url 参数是安装最新 GPU 版 PyTorch 的关键
pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 2. 安装基础依赖包
```bash
pip install -U onedl-mim
mim install onedl-mmengine
```

### 3. 核心难点：安装 MMCV

这是整个配置过程中最容易踩坑的环节。我们提供了两种解决方案。

#### 解决方案 A：使用预编译 Wheel 包安装 (推荐)

最理想的情况是直接通过 `mim` 安装：
```bash
mim install onedl-mmcv==2.3.2
```

> ⚠️ **避坑指南**：
> 运行上述命令后，请**务必检查终端输出的日志**。如果你看到类似下图这种“源码包构建 (Building wheel from source)”的日志：
> <p>{% asset_img 2.png "源码包安装日志: mmcv 安装日志" %}</p>
> *图 2：出现源码构建日志说明预编译包未成功匹配。*
>
> 这说明当前系统并没有找到与你的 Python/CUDA/PyTorch 版本严格对应的预构建包，它正在尝试本地编译。这种情况下，即使最终提示 `Successfully installed`，实际往往并没有正确编译 C++ 扩展，后续运行代码会报错 `No module named 'mmcv._ext'`。

**正确的预编译包安装姿势**：
根据 [OneDL-MMCV 官方文档](https://onedl-mmcv.readthedocs.io/en/latest/get_started/installation.html)，Python 3.10 + CUDA 12.8 + PyTorch 2.8.0 是有官方预编译包的。
若 `pip install` 自动匹配失败（如提示 `No matching distribution found`），请**直接通过 Wheel 文件的绝对地址安装**：

<p>{% asset_img 3.png "" %}</p>

*图 3：官方发布的预编译包列表。*

<p>{% asset_img 4.png "" %}</p>

*图 4：右键复制对应版本的下载链接。*

```bash
# 直接使用获取到的 .whl 文件链接进行安装
pip install https://mmwheels-bucket.onedl.ai/cu128-torch280/onedl-mmcv/onedl_mmcv-2.3.3-cp310-cp310-manylinux_2_34_x86_64.whl
```

#### 解决方案 B：从源代码编译 MMCV

如果由于种种原因预编译包不可用，你也可以选择从源码进行编译。建议将 mmcv 克隆到之后存放 `mmsegmentation` 的同一个父级目录下：

```bash
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout v2.1.0  # 切换到你需要的版本分支

# 设置 CUDA 环境变量以强制编译 C++ 算子
export FORCE_CUDA=1
export MMCV_WITH_OPS=1

# 编译并安装
pip install -r requirements.txt
python setup.py build_ext --inplace
pip install -e .
```

### 4. 从源码安装 OneDL-MMSegmentation

确保 `mmcv` 正确安装（且能导入 `mmcv._ext` 无报错）后，即可安装分割库本体：

```bash
git clone -b main https://github.com/vbti-development/onedl-mmsegmentation.git
cd onedl-mmsegmentation
pip install -v -e .
```

> ⚠️ **注意**：由于我们使用的是 `-e` (editable) 模式从源码安装，配置好环境后**切勿更改 `onedl-mmsegmentation` 文件夹的名称或路径**，否则会导致包引用失效。

---

## 四、全面测试与验证

为了确保环境无恙，我们可以运行一个官方提供的推理 Demo：

```bash
cd onedl-mmsegmentation  # 确保在项目主目录

# 下载配置文件与模型权重
mim download mmsegmentation --config pspnet_r50-d8_4xb2-40k_cityscapes-512x1024 --dest .

# 执行单张图片的推理脚本
python demo/image_demo.py demo/demo.png \
    configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py \
    pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth \
    --device cuda:0 --out-file result.jpg
```
🎉 **如果当前目录下成功生成了带有分割蒙版的 `result.jpg`，恭喜你，环境配置大功告成！**

---

## 五、在公开数据集 ADE20K 上的初步尝试

在跑自己的数据集之前，强烈建议先用框架内置支持的数据集（如 ADE20K）跑通训练与测试流程。

1. **准备数据**：从官方文档获取 ADE20K 数据集，解压并放置于 `data/ade` 目录下（内置的配置文件默认读取该路径）。
2. **启动训练**：尝试运行 Segformer 模型。
   *(配置文件命名规则：`segformer_mit-b0` (主干网络) + `8xb2` (8张卡每张batch设为2) + `160k` (总迭代次数) + `ade20k-512x512` (数据集与裁剪分辨率))*

```bash
conda activate mmseg

# 覆盖默认的超参数以加速验证：
# 由于 5090 显存极大，我们将 batch_size 调大到 16，并开启 8 个数据加载线程
python tools/train.py configs/segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512.py \
    --work-dir work_dirs/segformer_mit-b0_ade20k \
    --cfg-options train_dataloader.batch_size=16 val_dataloader.batch_size=1 train_dataloader.num_workers=8
```

也可以通过 `--cfg-options` 快速测试，缩短训练周期：
```bash
# 修改总迭代次数为 40k 次，每 2k 次验证一次
python tools/train.py configs/segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512.py \
    --work-dir work_dirs/segformer_mit-b0_ade20k_fast \
    --cfg-options train_dataloader.batch_size=16 \
                  val_dataloader.batch_size=1 \
                  train_dataloader.num_workers=8 \
                  train_cfg.max_iters=40000 \
                  train_cfg.val_interval=2000
```

### 实用工具拓展：显卡监控
你可以使用系统自带工具或进阶工具监控显卡满载状态：
```bash
# 方法 1：系统自带
watch -n 1 nvidia-smi
htop

# 方法 2：更直观的 nvitop (建议安装在 base 环境)
conda activate base
pip install nvitop
nvitop
```

### 模型测试与可视化
训练完成后，使用以下命令评估精度，并保存预测结果图：
*(注意将 `.pth` 替换为实际保存的最优权重文件名)*
```bash
python tools/test.py configs/segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512.py \
    work_dirs/segformer_mit-b0_ade20k/best_mIoU_iter_xxx.pth \
    --out work_dirs/segformer_mit-b0_ade20k/results \
    --show-dir work_dirs/segformer_mit-b0_ade20k/vis_results
```

---

## 六、跑通自定义数据集：以 SmokeSeg 为例

跑通官方数据集后，接下来迁移到自己的数据。
这里以开源的烟雾分割数据集 **[SmokeSeg](https://github.com/LujianYao/FoSp)** 为例。假设你已经将其转换为标准的 VOC 格式，并放置在项目目录 `data/SmokeSeg` 中。

### 第一步：注册新数据集
在 MMSegmentation 1.x 中，自定义数据集都需要继承基类 `BaseSegDataset`。
1. 在 `mmseg/datasets/` 目录下新建 `smoke_voc.py`，实现你的数据集读取逻辑。
2. 打开 `mmseg/datasets/__init__.py`，导入并暴露你的 `SmokeSegDataset` 模块。

### 第二步：构建对应的训练配置文件
前往 `configs/segformer/` 目录，创建一个专门针对 SmokeSeg 的配置文件，如 `segformer_mit-b5_8xb4-160k_smokeseg-515x512.py`。
*(Tip: 既然框架原生支持 Segformer，你可以直接利用 `_base_` 机制继承官方的基础 SegFormer-B5 结构，然后仅仅修改类别数、数据路径等必要信息。)*

### 第三步：开始训练
指向你的新配置文件，启动训练：
```bash
python tools/train.py configs/segformer/segformer_mit-b5_8xb4-160k_smokeseg-515x512.py
```

### 第四步：测试与可视化
```bash
python tools/test.py \
    configs/segformer/segformer_mit-b5_8xb4-160k_smokeseg-515x512.py \
    你的最佳权重文件路径.pth \
    --show-dir vis_results_smokeseg/
```
