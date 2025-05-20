```markdown

# CLIP-Fields 安装与训练全过程问题总结

---

## 1. 安装步骤（含依赖）

```bash
git clone --recursive https://github.com/notmahi/clip-fields
cd clip-fields

conda create -n cf python=3.8 -y
conda activate cf

# 安装 PyTorch（确保 CUDA 匹配）
conda install -y pytorch==1.12.0 torchvision torchaudio cudatoolkit=11.8 -c pytorch-lts -c nvidia

# 安装 Python 依赖
pip install -r requirements.txt
pip install jupyter

# 安装 LZFSE 解压库（如报错）
pip install git+https://github.com/yds540/lzfse-python

# 安装 SentenceTransformer、Transformers 等兼容版本
pip install sentence-transformers==2.2.2 transformers==4.21.2 huggingface_hub==0.10.1

# 下载数据并放置到主目录
(https://osf.io/famgv/files/osfstorage)

# ✅ Conda 安装 PyTorch + CUDA 11.8 常见问题总结与解决方案

---

## 🧨 问题 1：安装了 PyTorch 2.0.1 + CUDA 11.8，但运行时仍显示 PyTorch 2.4.1 + CUDA 12.1

### ❗现象

运行以下命令时：

```bash
python -c "import torch; print(torch.__version__)"
```

输出为：

```text
2.4.1+cu121  ❌
```

```bash
python -c "import torch; print(torch.version.cuda)"
```

输出为：

```text
12.1  ❌
```

---

### 📌 原因

系统中通过 `pip install torch` 安装过 PyTorch，位于：

```text
~/.local/lib/python3.8/site-packages/
```

该路径优先级高于 Conda 环境，污染了导入路径。

---

### ✅ 解决方法（⚠️ 请在 Conda 环境 **外** 执行）

```bash
pip uninstall -y torch torchvision torchaudio

rm -rf ~/.local/lib/python3.8/site-packages/torch*
rm -rf ~/.local/lib/python3.8/site-packages/torchvision*
rm -rf ~/.local/lib/python3.8/site-packages/torchaudio*
```

---

### ✅ 验证是否修复成功（请在 Conda 环境中执行）

```bash
python -c "import torch; print(torch.__version__)"
```

输出应为：

```text
2.0.1 ✅
```

```bash
python -c "import torch; print(torch.version.cuda)"
```

输出应为：

```text
11.8 ✅
```

```bash
python -c "import torch; print(torch.__file__)"
```

输出应为：

```text
/home/ubuntu/miniconda3/envs/cf/lib/... ✅
```

---

## 🧨 问题 2：nvcc 仍是系统默认的 CUDA 10.1，而不是 Conda 中安装的 CUDA 11.8

### ❗现象

```bash
which nvcc
```

输出：

```text
/usr/bin/nvcc ❌
```

```bash
nvcc --version
```

输出：

```text
Cuda compilation tools, release 10.1 ❌
```

---

### 📌 原因

你虽然安装了 `cuda-nvcc=11.8`，但没有设置 Conda 环境中的路径优先，因此仍然在用系统默认的 CUDA 10.1。

---

### ✅ 解决方法

在 Conda 环境中执行：

```bash
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
```

建议将这些添加到 `~/.bashrc` 中以便自动生效。

---

### ✅ 验证是否修复成功

```bash
which nvcc
```

输出应为：

```text
/home/ubuntu/miniconda3/envs/cf/bin/nvcc ✅
```

```bash
nvcc --version
```

输出应为：

```text
Cuda compilation tools, release 11.8 ✅
```

---

## 🧨 问题 3：编译 detectron2 或 torch-encoding 报错：找不到 `cuda_bf16.h`

### ❗错误信息

```text
fatal error: cuda_bf16.h: No such file or directory
```

---

### 📌 原因

你安装了 `cuda-nvcc=11.8`，但没有安装包含完整 CUDA headers 的 `cuda-cudart-dev=11.8`，导致头文件缺失。

---

### ✅ 解决方法

在 Conda 环境中安装完整 CUDA 11.8 工具链：

```bash
conda install -c nvidia \
    cuda-nvcc=11.8 \
    cuda-cudart=11.8 \
    cuda-cudart-dev=11.8 \
    cuda-libraries-dev=11.8 \
    cuda-tools=11.8
```

---

### ✅ 验证是否安装成功

```bash
find $CONDA_PREFIX/include -name cuda_bf16.h
```

输出应为：

```text
/home/ubuntu/miniconda3/envs/cf/include/cuda_bf16.h ✅
```

---


---

## 🧾 建议：添加 CUDA 环境变量到 `.bashrc`

```bash
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
```

---

✅ 以上为 Conda 下构建 PyTorch + CUDA 11.8 + Detectron2/torch-encoding 全套问题总结与解决方案。
```
# ✅ CLIP-Fields 安装与训练完整问题解决记录（Ubuntu + Conda）

---

## ✅ 系统环境要求

- Python 3.8
- CUDA 11.8（系统支持即可，无需精确匹配）
- 推荐使用 Conda 管理虚拟环境

---

## ✅ Conda 环境创建与 PyTorch 安装

最开始 PyTorch CUDA 版本不匹配：

### ❌ 错误示例：

```text
torch.version.cuda = None
torch.cuda.is_available() = False
```

### ✅ 正确安装命令（用 pip 装 PyTorch GPU 版）：

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

### ✅ 验证：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

```text
1.12.0+cu113
11.3
True
```

---

## ✅ 安装 PyTorch-Encoding 编译失败

### ❌ 报错 1：未定义符号（CPU-only PyTorch 编的）

```text
undefined symbol: _ZN2at4_ops15sum_dim_IntList4call...
```

### ✅ 解决：

使用 GPU 版本 PyTorch 后重新编译：

```bash
cd ~/clip-fields/PyTorch-Encoding
python setup.py clean
python setup.py install
```

---

### ❌ 报错 2：找不到 `nvcc`

```text
error: No such file or directory: .../envs/cf/bin/nvcc
```

### ✅ 解决：

```bash
conda install -c nvidia cuda-nvcc=11.3
```

或设置系统 CUDA：

```bash
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

---

### ❌ 报错 3：不支持的 GCC 版本

```text
#error -- unsupported GNU version! gcc versions later than 8 are not supported!
```

### ✅ 解决：

```bash
sudo apt install gcc-8 g++-8
export CC=/usr/bin/gcc-8
export CXX=/usr/bin/g++-8
```

---

## ✅ gridencoder 模块运行失败

### ❌ 报错：缺少 ninja

```text
RuntimeError: Ninja is required to load C++ extensions
```

### ✅ 解决：

```bash
conda install -y ninja
```

---

## ✅ CLIP-Fields 数据集下载问题

GitHub 上的 `nyu.r3d` 链接 404，改为从 OSF 下载：

```bash
wget https://osf.io/famgv/download -O nyu.r3d
```

---

## ✅ 编译验证完成后启动训练

```bash
python train.py dataset_path=nyu.r3d
```

训练即可正常开始。

---

## ✅ 推荐持久设置

写入 `.bashrc`（可选）：

```bash
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
export CC=/usr/bin/gcc-8
export CXX=/usr/bin/g++-8
```

---

## ✅ 成功 🎉

你现在已经成功解决了以下问题：

- CPU-only PyTorch 导致 `.so` 导入失败；
- `nvcc` 缺失导致扩展无法编译；
- 系统 CUDA 与 PyTorch CUDA 不一致；
- GCC 太高不被 nvcc 支持；
- 缺少 `ninja`；
- gridencoder 编译失败；
- 数据集下载失效。

你现在可以正常运行 CLIP-Fields 并开始训练。

---