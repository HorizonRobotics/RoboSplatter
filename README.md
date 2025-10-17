# RoboSplatter 🤖💫
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)]()

**面向机器人仿真的高斯溅射仿真框架 | Gaussian Splatting for Robotic Simulation**


## 🌟 核心特性 ✨ | Core Features

<!-- - **物理精准仿真**：基于MuJoCo的机器人动力学仿真引擎 | Physical-accurate simulation using the MuJoCo physics engine. -->
- **实时高斯渲染**：集成高效3D高斯溅射渲染管线 | Real-time Gaussian splatting rendering pipeline for 3D.
- **多模态感知**：支持RGB、Depth相机等多传感器仿真 | Multi-modal perception support with RGB, Depth cameras, and other sensors.

---

## 🛠️ 安装指南 | Installation Guide

### Pre-requests
- Python >= 3.10
- CUDA >= 11.8
- (Optional) [uv](https://docs.astral.sh/uv/) for faster environment setup

### 环境配置 ｜ Environment Configuration


```sh
# 1. Clone the repository
git clone https://github.com/HorizonRobotics/RoboSplatter.git
cd RoboSplatter

# 2. Create conda environment
# conda create -n robosplatter python=3.10 -y
# conda activate robosplatter

# 3. Install dependencies
pip install -e . #uv

```

### 下载资产 ｜ Download Assets

```sh
# 安装 huggingface_hub
# pip install huggingface_hub

python -m huggingface_hub.commands.huggingface_cli download HorizonRobotics/RoboSplatter --repo-type dataset --local-dir ./assets
# desk2.ply, golden_cup.ply, lab_table.ply, office.ply 等文件
```

## 🚀 运行指南 | Running Guide

### GS渲染 | GS Render

#### 渲染背景场景 | Render Background Scene
```sh
render-cli --data_file config/gs_data_basic.yaml \
  --camera_extrinsic "[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]" \
  --camera_intrinsic "[[606.6, 0.0, 326.3], [0.0, 607.6, 242.7], [0.0, 0.0, 1.0]]" \
  --image_height 480 \
  --image_width 640 \
  --device "cuda" \
  --output_dir "./output/background"
```

#### 批量渲染场景 | Render Scene Batch
```sh
python robo_splatter/scripts/render_scene_batch.py --data_file config/gs_data_fg_bg_mix.yaml \
  --camera_extrinsic "[[0, 1.5, 0, 0.0, -0.7071, 0.0, -0.7071], [0, 1.5, 0.0, 0.0, -0.5, 0.0, -0.866], [0, 1.5, 0.0, 0.0, -0.2588, 0.0, -0.9659], [0, 1.5, 0.0, 0.0, 0.0, 0.0, -1.0], [0, 1.5, 0.0, 0.0, 0.2588, 0.0, -0.9659], [0, 1.5, 0.0, 0.0, 0.5, 0.0, -0.866], [0, 1.5, 0.0, 0.0, 0.7071, 0.0, -0.7071], [0, 1.5, 0.0, 0.0, 0.866, 0.0, -0.5], [0, 1.5, 0.0, 0.0, 0.9659, 0.0, -0.2588], [0, 1.5, 0.0, 0.0, 1.0, 0.0, 0.0], [0, 1.5, 0.0, 0.0, 0.9659, 0.0, 0.2588], [0, 1.5, 0.0, 0.0, 0.866, 0.0, 0.5],[0, 1.5, 0, 0.0, -0.7071, 0.0, -0.7071]]" \
  --camera_intrinsic "[[256.0, 0.0, 512.0], [0.0, 256.0, 512.0], [0.0, 0.0, 1.0]]" \
  --image_height 1024 \
  --image_width 1024 \
  --coord_system MUJOCO \
  --output_dir "./output/mix_bg_fg_demo" \
  --gen_mp4_path "./output/mix_bg_fg_demo/render.mp4"
```

**输出文件**：
- 渲染图片将保存在指定的 `output_dir` 目录
- 如果指定了 `--gen_mp4_path`，将生成视频文件

## 🚗 目录结构 | Directory Structure

- **robo_splatter/**
  - **config/**: 仿真配置文件 | Simulation configuration files
  - **models/**: 3D GS数据结构及建模 | 3D GS data structures and modeling
  - **render/**: 3D GS场景渲染 | 3D GS scene configurations
  - **utils/**: 通用工具函数 | General utility functions
  - **scripts/**: 使用示例 | 3D GS example use cases
<!-- - **projects/**: 更多综合使用示例 | More comprehensive sim usage examples(Coming Soon) -->
---



## For developers only
```sh
pip install -e .[dev] && pre-commit install
```


## 🙏 致谢 | Acknowledgments

We utilize the rasterization kernel from [gsplat](https://github.com/nerfstudio-project/gsplat).
The design draws inspiration from [DriveStudio](https://github.com/ziyc/drivestudio) and [DISCOVERSE](https://github.com/TATP-233/DISCOVERSE).


## ⚖️ License

This project is licensed under the [Apache License 2.0](LICENSE). See the `LICENSE` file for details.


## 📚 Citation

If you use RoboSplatter in your research or projects, please cite:

```bibtex
@misc{wang2025embodiedgengenerative3dworld,
      title={EmbodiedGen: Towards a Generative 3D World Engine for Embodied Intelligence},
      author={Xinjie Wang and Liu Liu and Yu Cao and Ruiqi Wu and Wenkang Qin and Dehui Wang and Wei Sui and Zhizhong Su},
      year={2025},
      eprint={2506.10600},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2506.10600},
}
```
