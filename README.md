# PawGuard — 智能宠物行为分析系统

> 基于计算机视觉的猫狗检测与行为分析系统  
> A Computer Vision Course Project | 计算机视觉课程实践项目

---

## 📌 项目简介

PawGuard 是一个面向**家养宠物**与**流浪动物**双重场景的智能视觉分析系统，基于 YOLOv8 目标检测算法和 OpenCV 图像处理技术，实现猫狗目标的自动识别、个体身份确认和行为状态分析。

### 核心功能

| 功能模块 | 描述 | 状态 |
|---------|------|------|
| **目标检测** | 识别画面中的猫狗目标，输出位置与类别 | ✅ 已实现 |
| **身份确认** | 基于颜色直方图和轮廓特征匹配个体 | ⬜ 开发中 |
| **行为分析** | 判断行为状态（进食、睡觉、玩耍、异常静止） | ⬜ 开发中 |
| **实时监测** | 支持视频流实时检测与分析 | ⬜ 待实现 |

### 应用场景

- **家养宠物监控**：远程查看宠物状态，异常行为预警
- **宠物医院监护**：术后恢复监测，自动记录活动情况
- **流浪动物救助**：批量监测流浪动物健康状况

---

## 👥 项目团队

| 姓名 | 分工 | 负责模块 |
|------|------|---------|
| 郑云龙 | 技术负责人 | 身份识别模块、技术架构设计、技术文档 |
| 郭志杰 | 需求与报告 | 行为分析模块、需求分析、最终报告 |
| 朱晋 | 数据与测试 | 数据集整理、数据增强、测试与调试 |
| 范凯翔 | 算法对比 | 算法对比实验、模型优化、用户手册 |
| 农惠渝 | 统筹与前后端 | 环境搭建、模型测试、视频检测、演示准备、进度统筹 |

---

## 🛠 技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| Python | 3.9+ | 开发语言 |
| OpenCV | 4.8+ | 图像预处理、DNN模块、特征提取 |
| Ultralytics YOLO | 8.0+ | 目标检测（YOLOv8n） |
| NumPy | 1.21+ | 数值计算与数据处理 |

### OpenCV核心函数使用

| 函数 | 模块 | 用途 |
|------|------|------|
| `cv2.VideoCapture()` | 视频I/O | 摄像头/视频读取 |
| `cv2.dnn.readNetFromDarknet()` | DNN | YOLO模型加载 |
| `cv2.resize()` | 图像处理 | 图像缩放与标准化 |
| `cv2.calcHist()` | 图像处理 | 颜色直方图计算（身份识别） |
| `cv2.matchTemplate()` | 图像处理 | 模板匹配（特征比对） |
| `cv2.putText()` / `cv2.rectangle()` | 绘图 | 检测结果可视化 |

---

## 📁 项目结构

```
PawGuard/
├── docs/                              # 文档目录
│   ├── 实践报告.md                     # 课程实践报告
│   ├── 技术架构说明.md                  # 系统技术架构文档
│   ├── 技术路线与系统设计摘要.md         # 技术路线摘要
│   ├── 检测结果分析.md                  # 模型检测结果分析
│   ├── 后续分工安排.md                  # 项目分工与进度安排
│   ├── 视频识别方案.md                  # 视频识别技术方案
│   ├── GitHub上传指南.md                # GitHub上传操作指南
│   └── references/                    # 开源项目参考
│       ├── 开源项目调研.md              # 四个开源项目调研报告
│       ├── MMSkeleton/                # OpenMMLab骨骼理解工具包
│       ├── cat-vs-dogo/               # 猫狗品种分类器
│       ├── asbar/                     # 动物骨骼姿态识别
│       └── NHP-AR/                    # 猕猴动作识别系统
├── data/                              # 数据目录
│   ├── dataset/                       # 数据集
│   │   ├── Oxford-IIIT-Pet-dataset/   # Oxford-IIIT宠物数据集
│   │   └── kaggle-cat-vs-dog-dataset/ # Kaggle猫狗数据集
│   └── test.jpg                       # 测试图片
├── models/                            # 模型文件目录
│   ├── yolov8n.pt                     # YOLOv8n模型权重（自动下载）
│   ├── coco.names                     # COCO数据集类别名称
│   └── README.md                      # 模型下载说明
├── src/                               # 源代码目录
│   ├── ultralytics_demo.py            # YOLOv8检测Demo（主程序）
│   ├── demo_detect.py                 # OpenCV DNN检测Demo
│   └── download_models.py             # 模型自动下载脚本
├── demo/                              # 检测结果输出目录
│   ├── Oxford-IIIT-Pet-dataset/       # Oxford数据集检测结果
│   └── kaggle-cat-vs-dog-dataset/     # Kaggle数据集检测结果
├── .gitignore                         # Git忽略配置
├── requirements.txt                   # Python依赖列表
└── README.md                          # 项目说明文档（本文件）
```

---

## 🚀 快速开始

### 环境要求

- Python 3.9 或更高版本
- Windows 10/11 或 Linux/macOS
- 至少 4GB 内存（推荐 8GB）

### 安装依赖

```bash
# 克隆仓库
git clone https://github.com/KrisX-Vivian/PawGuard.git
cd PawGuard

# 安装依赖
pip install -r requirements.txt
```

### 下载模型文件

首次运行前，需要先下载 YOLO 模型文件：

```bash
# 方式1：使用自动下载脚本（推荐）
python src/download_models.py

# 方式2：手动下载
# 详见 models/README.md
```

### 运行检测Demo

#### 单张图片检测

```bash
python src/ultralytics_demo.py data/test.jpg
```

#### 批量检测（从数据集随机抽取）

```bash
python src/ultralytics_demo.py --batch \
  --dataset1 data/dataset/Oxford-IIIT-Pet-dataset/images \
  --dataset2 data/dataset/kaggle-cat-vs-dog-dataset \
  --count 10
```

#### 基于OpenCV DNN的检测（备用方案）

```bash
python src/demo_detect.py
```

---

## 📊 数据集

### 主力数据集

| 数据集 | 来源 | 规模 | 标注信息 | 用途 |
|--------|------|------|---------|------|
| Oxford-IIIT Pet Dataset | 牛津大学 | 7,390张 | 37个品种、边界框 | 训练与测试 |
| Kaggle Cat vs Dog Dataset | Kaggle | 10,028张 | 猫/狗二分类 | 补充测试 |

### 数据预处理

1. **统一尺寸**：缩放至 640×640 像素
2. **格式转换**：统一为 JPEG 格式
3. **类别过滤**：仅保留猫（COCO ID:15）和狗（COCO ID:16）
4. **数据增强**（计划）：旋转、翻转、亮度调整

---

## 📈 实验结果

### 目标检测性能

使用 YOLOv8n 模型在两个数据集上进行测试：

| 数据集 | 测试数量 | 准确率 | 平均置信度 | 备注 |
|--------|---------|--------|-----------|------|
| Oxford-IIIT Pet | 10张 | 90% | 86.8% | 1张误判（荷兰毛狮犬→猫） |
| Kaggle Cat vs Dog | 10张 | 90% | 70.6% | 1张漏检，2张低置信度 |
| **综合** | **20张** | **90%** | **78.7%** | - |

### 检测速度

| 模型 | 输入尺寸 | 推理时间 | 硬件环境 |
|------|---------|---------|---------|
| YOLOv8n | 640×640 | ~80ms/帧 | CPU (Intel i5) |
| YOLOv3-tiny | 416×416 | ~50ms/帧 | CPU (Intel i5) |

---

## 📅 项目进度

| 阶段 | 时间 | 内容 | 状态 |
|------|------|------|------|
| 第1周 | 已完成 | 选题确认、环境搭建、最小Demo跑通、数据集获取 | ✅ |
| 第2周 | 进行中 | 身份识别模块开发、行为分析模块开发、算法对比实验 | ⬜ |
| 第3周 | 计划中 | 前后端对接、测试与调试、模型优化、视频检测功能 | ⬜ |
| 第4周 | 计划中 | 整体调试、演示准备、最终报告、用户手册 | ⬜ |

---

## 🔧 算法对比

| 算法 | 模型大小 | 检测速度 | 准确率 | 适用场景 |
|------|---------|---------|--------|---------|
| YOLOv8n | 6.2MB | ~80ms/帧 | 90% | 实时检测（推荐） |
| YOLOv3-tiny | 35MB | ~50ms/帧 | ~85% | 轻量级部署 |
| YOLOv4-tiny | 24MB | ~60ms/帧 | ~88% | 平衡方案 |

---

## 📝 文档说明

| 文档 | 内容 | 位置 |
|------|------|------|
| 实践报告 | 课程要求的完整实践报告 | `docs/实践报告.md` |
| 技术架构 | 系统五层架构详细说明 | `docs/技术架构说明.md` |
| 检测结果 | 模型测试详细分析 | `docs/检测结果分析.md` |
| 分工安排 | 项目进度与任务分配 | `docs/后续分工安排.md` |
| 视频方案 | 视频识别技术方案 | `docs/视频识别方案.md` |

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

### 提交规范

- 使用清晰的提交信息
- 在 `docs/` 目录下更新相关文档
- 确保代码通过基础测试

---

## 📄 许可证

本项目基于 MIT 许可证开源，详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLOv8 目标检测框架
- [OpenCV](https://opencv.org/) - 计算机视觉库
- Oxford-IIIT Pet Dataset - 宠物数据集
- Kaggle - 猫狗分类数据集

---

**项目主页**：https://github.com/KrisX-Vivian/PawGuard  
**课程信息**：计算机视觉 | 2023级本科 | [学校名称]  
**最后更新**：2026年5月6日
