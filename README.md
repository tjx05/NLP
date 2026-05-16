# NLP
NLP大作业：从零开始构建一个大语言模型

## 项目结构
```
NLP/
├── config.py # 配置文件,模型超参数（vocab_size, n_layers等）
│
├── data/ # 数据集（不提交git）
│ ├── raw/ # 原始文本
│ └── processed/ # 预处理后的token张量
│
├── src/ # 核心源代码
│ ├── __init__.py 
│ │
│ ├── data/ # 数据模块
│ │ ├── __init__.py 
│ │ ├── data.py # 获得数据语料
│ │ ├── data_loader.py # 清洗获得的数据语料并分词
│ │ └── dataloader.py # 接收已分词好的,创建快捷式数据加载器
│ │
│ ├── models/ # 模型模块
│ │ ├── __init__.py 
│ │ ├── attention.py # 多头掩码注意力
│ │ ├── transformer_block.py # Transformer组件：Transformer Block+LayerNorm+FFN
│ │ └── gpt_model.py # 完整GPT模型
│ │
│ └── utils/ # 工具模块
│   ├── __init__.py 
│   ├── generation.py # 文本生成（贪心、温度+top-k）
│   └── evaluation.py # 评估与测评
│
├── scripts/ # 运行入口
│ ├── run_pretrain.py # 预训练主脚本(单卡)
│ ├── run_pretrain_multi.py # 分布式预训练主脚本
│ ├── run_classifier.py # 分类微调主脚本
│ └── run_demo.py # Gradio可视化界面
│
├── checkpoints/ # 保存的模型权重
├── output/ # 输出结果（损失曲线、生成样例）
└── README.md
```

## 快速开始
```bash
# 1.安装依赖：
pip install -r requirements.txt

# 2.数据集加载，如果'data/processed/'目录为空，需要先运行data.py和data_loader.py
python src/data/data.py
python src/data/data_loader.py


# 3.预训练
python scripts/run_pretrain.py
