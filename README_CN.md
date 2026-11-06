# 🐾 PAINR: 基于隐式神经表征 (INR) 的稀疏光声成像重建 ✨

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/CUDA-11.8%20%7C%2012.1-76B900?style=flat&logo=nvidia&logoColor=white" alt="CUDA"></a>
  <a href="https://github.com/NVlabs/tiny-cuda-nn"><img src="https://img.shields.io/badge/tiny--cuda--nn-v2.0-FF6F00?style=flat" alt="tiny-cuda-nn"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=flat" alt="License: MIT"></a>
  <a href="README.md"><img src="https://img.shields.io/badge/Docs-English%20README-pink?style=flat" alt="English Documentation"></a>
</p>

基于 PyTorch 与 `tiny-cuda-nn` 构建的轻量、高效、自监督隐式神经表征（INR）稀疏与有限视角光声成像（PAT / PACT）重建算法框架。

---

## 🌸 1. 项目简介 (What is PAINR?)

**光声层析成像 (Photoacoustic Tomography, PAT / PACT)** 是一种非常迷人的跨模态生物医学成像技术！✨ 它巧妙地将生物组织的**光学吸收高对比度**与**深层超声高穿透、高空间分辨率**融为一体，能够在不损伤生物体的前提下实现体内微血管的高清无创显像。

然而在临床与生物实验现场，由于骨骼解剖阻挡或高速扫描限制，探测器常常无法形成 360° 全视角环绕，从而面临严峻的**有限视角（Limited-view）**与**稀疏采样（Sparse-sampling）**难题：
- 💔 传统的滤波反投影（UBP）算法会留下严重的伪影和条纹噪声，血管边缘模糊难辨；
- 😿 常见监督深度学习方法又极度依赖成千上万对无伪影真值（Ground Truth）标签，而在实际医学扫描中几乎无法获取真实真值。

**PAINR** 带着可爱的**隐式神经表征 (Implicit Neural Representation, INR)** 给出了自监督解法！🎈

- 🧁 **纯自监督·零样本优化**：直接在单次采集的正弦图（Sinogram）上进行测试时优化，**不需要预训练权重、不需要外部数据集、更不需要真值图像**！
- 🌊 **物理驱动前向算子**：将基于圆弧 Radon 积分与时间一阶数值微商的光声波动传播模型，直接写进可微损失计算图。
- ⚡ **NVIDIA 哈希网格极速加速**：借助 `tiny-cuda-nn` 硬件加速哈希网格与 Fully-Fused MLP，数秒内即可捕捉丰富的高频微血管细节。
- 🎨 **连续坐标场任意放大**：摆脱离散体素限制，训练后可随心渲染 $512 \times 512$、$1024 \times 1024$ 或更高分辨率的清晰图像。

---

## 🎨 2. 核心架构与物理流程图 (How It Works)

光声正向传播将初始声压 $p_0(\mathbf{r})$ 与探测器测得的声压信号 $p(\mathbf{r}_d, t)$ 紧密相连：

$$p(\mathbf{r}_d, t) \propto \frac{\partial}{\partial t} \left[ \frac{1}{c_0 t} \int_{|\mathbf{r}-\mathbf{r}_d| = c_0 t} p_0(\mathbf{r}) \, \mathrm{d}\mathbf{r} \right]$$

PAINR 通过双半径同轴同心圆弧 ($r \pm \Delta r$) 对空间积分并求时间偏微分，把预测声压与测量正弦图进行反向微调：

```text
       ┌─────────────────────────────────────────────────────────────┐
       │             实测有限视角正弦图 (Sinogram 数据) 📡             │
       │                   p_meas(θ, t)                              │
       └──────────────────────────────┬──────────────────────────────┘
                                      │
                                      ▼  🐾 (MSE 损失函数)
┌─────────────────────────────────────┴─────────────────────────────────────┐
│                            物理驱动损失函数 🎯                             │
│             L = || p_meas(θ, t) - p_pred(θ, t) ||^2                       │
└─────────────────────────────────────▲─────────────────────────────────────┘
                                      │ 反向传播更新网络参数 💫
┌─────────────────────────────────────┴─────────────────────────────────────┐
│                        可微光声物理前向算子 🌊                            │
│  - 双半径同心圆弧坐标采样 (r + Δr, r - Δr)                                 │
│  - 探测器接收角射线积分：\int p0(r) dl                                     │
│  - 时间一阶数值微商：[Proj(r+Δr) - Proj(r-Δr)] / (2 * Δt)                 │
└─────────────────────────────────────▲─────────────────────────────────────┘
                                      │ 连续空间坐标查询 p0(x, y) 🔍
┌─────────────────────────────────────┴─────────────────────────────────────┐
│                    隐式神经表征网络 (INR 神经网络) 🧠                     │
│   - 多分辨率哈希网格编码 (tiny-cuda-nn, 14 级精细分辨率) ⚡                │
│   - 融合全连接网络 (Fully-Fused MLP: 64 神经元，2 个隐藏层，ReLU 激活)     │
└─────────────────────────────────────▲─────────────────────────────────────┘
                                      │ 输入目标空间坐标 (x, y) 📍
┌─────────────────────────────────────┴─────────────────────────────────────┐
│                        连续二维成像视场空间                                │
└───────────────────────────────────────────────────────────────────────────┘
                                      │ 训练完成后全图网格推理 🎉
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│               高质量初始声压分布图 p0(x, y) 🖼️ (无伪影、高保真)            │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ 3. 核心亮点 (Key Features)

- 🥑 **零样本·即开即用**：每一个扫描样本独立优化，不依赖外部历史数据。
- 🎯 **严谨物理波动先验**：准确还原声波传播延迟（$r = c_0 \cdot t$）与换能器接收孔径。
- 🎀 **天然抑制条纹伪影**：神经网络隐式平滑先验极大抑制了有限角度与稀疏采集产生的发散伪影。
- 🔍 **超分辨率连续查询**：可根据需求随时在任意细密网格上采样，轻松输出 $1024 \times 1024$ 甚至更大尺寸的高清大图。
- 📦 **开箱即用医学格式**：原生通过 SimpleITK 支持 NIfTI (`.nii`) 格式输入。

---

## 🗂️ 4. 仓库文件地图 (Repository Structure)

```text
🐾 PAINR/
├── ⚙️ configs/
│   └── config.json          # tiny-cuda-nn 哈希网格层数与网络超参数
├── 🧺 data/
│   └── raw/                 # NIfTI 格式测试正弦图
│       └── img1_128_Norm.nii    # 128 阵元归一化测试正弦图（默认示例）
├── 📜 dataset.py               # 数据加载类、正弦图采样与环形探测器射线坐标生成
├── 📐 utils.py                 # 笛卡尔网格生成与空间坐标旋转、弧线采样工具
├── 🚀 train.py                 # 主训练与重建测试脚本
├── 📋 requirements.txt         # 基础 Python 依赖包清单
├── 📄 LICENSE                  # MIT 开源协议文件
├── 📘 README.md                # 英文说明文档
└── 🌸 README_CN.md             # 中文说明文档
```

---

## 🍰 5. 保姆级环境配置指南 (Installation)

### 🐾 第一步：创建专属 Conda 环境

```bash
conda create -n painr python=3.10 -y
conda activate painr
```

### 🧸 第二步：安装对应 CUDA 的 PyTorch

请根据本地显卡与 CUDA Toolkit 版本选择（推荐 CUDA 12.1）：

```bash
# CUDA 12.1（强烈推荐）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 🎀 第三步：安装 `tiny-cuda-nn` 加速库

```bash
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

> 💡 **给 Windows 小伙伴的贴心避坑提示**：
> - 建议在 Windows 开始菜单中打开 **"x64 Native Tools Command Prompt for VS"** 终端执行上面这行命令，确保 C++ 编译器 `cl.exe` 处于环境变量中。
> - 如果编译时间偏长，可预先限定你的显卡架构，例如 RTX 30 系列运行 `set TCNN_CUDA_ARCHITECTURES=86`，RTX 40 系列运行 `set TCNN_CUDA_ARCHITECTURES=89`。

### 🍭 第四步：安装其余依赖包

```bash
pip install -r requirements.txt
```

### 🎈 第五步：测试环境是否大功告成！

```bash
python -c "import torch, tinycudann, SimpleITK; print('✨ CUDA 可用状态:', torch.cuda.is_available()); print('🎉 PAINR 环境配置完美！')"
```

---

## 🚀 6. 运行起来吧！(Quick Start)

### 🐾 启动重建优化

只需在根目录下执行：

```bash
python train.py
```

优化输出将自动保存在 `outputs/YYYYMMDD_HHMMSS/` 文件夹下：
- 🖼️ `latest_recon.png`：每个 Epoch 实时刷新的重建快照；
- 🏆 `final_recon.png`：优化结束后的最终 $1024 \times 1024$ 高清重建图；
- 📈 `tensorboard_logs/`：记录每次迭代的 Loss 下降曲线。

### 🍓 看看损失下降得有多漂亮 (TensorBoard)

```bash
tensorboard --logdir outputs
```
在浏览器打开 `http://localhost:6006`，就能看到损失像坐滑梯一样平稳下降啦！📉✨

---

## 🎛️ 7. 物理与实验参数小调节 (Parameter Knobs)

### 🧪 物理与硬件参数（位于 `train.py` 顶部）

所有关键参数都贴心地放在了 `train.py` 最显眼的位置，方便根据不同采集系统调节：

| 参数名称 | 默认值 | 物理含义与调节建议 |
| :--- | :--- | :--- |
| `sin_path` | `data/raw/img1_128_Norm.nii` | 输入正弦图路径（`.nii` 格式） |
| `num_sensors` ($\theta$) | `128` | 探测器阵元数量 / 采集角度数 |
| `sample_N` | `1200` | 每个传感器通道的时间采样点数 |
| `radius` | `0.04` ($40\text{ mm}$) | 环形探测器环的物理半径 |
| `field_of_view` | `0.04` ($40\text{ mm}$) | 探测视场（FOV）物理宽度 |
| `speed_of_sound` | `1500` ($\text{m/s}$) | 声波在传播介质中的声速（软组织/水约为 1500 m/s） |
| `time_step` | `5e-8` ($50\text{ ns}$) | 采集系统的单次采样时间步长 $\Delta t$ |
| `delta_t` | `1e-9` ($1\text{ ns}$) | 用于计算时间数值微商的极小偏移步长 |
| `num_rays` | `400` | 积分圆弧上的离散射线采样条数 |
| `grid_size` | `1024` | 最终输出图像的网格像素分辨率 ($1024 \times 1024$) |
| `num_epochs` | `80` | 自监督优化训练迭代总轮数 |
| `batch_size` | `8` | 训练单步批次选取的传感器通道数 |
| `lr` | `1e-3` | 初始学习率（配备 StepLR 动态衰减） |

### 🧠 神经网络配置（位于 `configs/config.json`）

```json
{
  "encoding": {
    "otype": "Grid",
    "type": "Hash",
    "n_levels": 14,
    "n_features_per_level": 8,
    "log2_hashmap_size": 24,
    "base_resolution": 1,
    "per_level_scale": 2,
    "interpolation": "Linear"
  },
  "network": {
    "otype": "FullyFusedMLP",
    "activation": "ReLu",
    "output_activation": "None",
    "n_neurons": 64,
    "n_hidden_layers": 2
  }
}
```

---

## 🧺 8. 如何跑自己的采集数据 (Custom Data)

想用 PAINR 重建您自己的光声采集数据吗？只需三步：

1. 📦 **格式准备**：将采集到的正弦图导出为 2D NIfTI (`.nii`) 数组，维度为 `(探测器数量, 时间采样点数)`；
2. 🧼 **数据归一化**：将信号强度归一化至合理数值区间（如 $[0, 1]$ 或标准化），有助于网络平稳收敛；
3. 📐 **硬件参数对齐**：在 `train.py` 中将 `radius`、`speed_of_sound` 和 `time_step` 修改为您实际实验硬件系统的标定参数。

---

## 💌 9. 论文引用 (Citation)

如果本项目代码、数据或方法对您的研究有所帮助，欢迎引用我们的论文：

```bibtex
@article{XIAO2025100685,
  title = {Limited-view photoacoustic imaging reconstruction via high-quality self-supervised neural representation},
  author = {Youshen Xiao and Yuting Shen and Sheng Liao and Bowei Yao and Xiran Cai and Yuyao Zhang and Fei Gao},
  journal = {Photoacoustics},
  volume = {42},
  pages = {100685},
  year = {2025},
  issn = {2213-5979},
  doi = {https://doi.org/10.1016/j.pacs.2025.100685},
  url = {https://www.sciencedirect.com/science/article/pii/S2213597925000047}
}
```

---

## 💖 10. 开源许可与特别鸣谢 (License & Thanks)

- 📄 基于 [MIT License](LICENSE) 开源协议发布，可自由学术研究与衍生修改。
- 🧸 特别感谢 NVIDIA 提供的 [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) 高性能神经渲染基座。
- 🍰 感谢 [PyTorch](https://pytorch.org/) 与 [SimpleITK](https://simpleitk.org/) 优秀的开源社区！
