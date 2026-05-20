# NLP
NLP大作业：从零开始构建一个大语言模型

## 项目结构
```
```text
NLP/
├── .gitignore                   # Git 忽略配置文件
├── README.md                    # 项目说明文档
├── requirements.txt            # 项目依赖库列表
├── app.py                      # Web 交互应用启动入口
├── config.py                   # 全局参数与模型配置
├── checkpoints/                # 模型权重及训练检查点存放目录
├── data/                       # 数据集目录
│   ├── processed/              # 预处理后的结构化数据
│   └── raw/                    # 原始未处理数据
├── output/                     # 训练产出、日志及图表
│   └── instruct_loss_curve.png # 指令微调 Loss 训练曲线图
├── scripts/                    # 顶层任务执行脚本
│   ├── run_classifier.py       # 文本分类器训练脚本
│   ├── run_instruction_fintune.py # LLM 指令微调 (Instruction Tuning) 训练脚本
│   ├── run_instruction_test.py # 指令微调模型测试与生成效果评估
│   ├── run_pretrain.py         # 单卡模型预训练 (Pre-train) 脚本
│   ├── run_pretrain_multi.py   # 多卡分布式预训练脚本
│   └── test_classifier.py      # 分类器测试评估脚本
├── src/                        # 核心源代码目录 (Source)
│   ├── __init__.py
│   ├── data/                   # 数据处理与 Dataset 模块
│   │   ├── __init__.py
│   │   ├── cls_data.py         # 分类任务数据流处理
│   │   ├── cls_dataloader.py   # 分类任务专用 DataLoader
│   │   ├── data.py             # 数据处理基础类/通用函数，指令微调数据集加载
│   │   ├── data_loader.py      # 通用数据加载逻辑
│   │   ├── dataloader.py       # 基础 DataLoader 构建
│   │   └── instruction_dataset.py # 指令微调 Dataset 构建与 Prompt 拼接
│   ├── models/                 # 模型结构定义
│   │   ├── __init__.py
│   │   ├── attention.py        # 注意力机制实现 (Self-Attention / Masked Attention)
│   │   ├── gpt_model.py        # GPT 模型主体网络架构
│   │   └── transformer_block.py # Transformer Decoder Block 核心块实现
│   └── utils/                  # 训练与推理工具函数库
│       ├── __init__.py
│       ├── cls_finetune.py     # 分类微调辅助工具
│       ├── evaluation.py       # 模型指标评估 (Acc, Bleu, Rouge 等)
│       └── generation.py       # 解码生成算法 (Top-p, Top-k, Beam Search，指令微调工具 等)
└── templates/                  # 前端 HTML 模板目录
    └── index.html              # Web UI 交互界面

## 快速开始
```bash
# 1.安装依赖：
pip install -r requirements.txt

# 2.数据集加载，如果'data/processed/'目录为空，需要先运行data.py和data_loader.py
python src/data/data.py
python src/data/data_loader.py


# 3.预训练
python scripts/run_pretrain.py
