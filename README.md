# NLP
NLP大作业：从零开始构建一个大语言模型

## 项目结构
```
NLP/
├── checkpoints
├── data
│   ├── cls_raw
│   │   ├── sms_spam_collection
│   ├── instrc_raw  json微调的数据集
│   ├── processed
│   ├── raw
├── output   里面有评测结果
├── scripts
│   ├── data
│   │   └── raw          这个不管
│   ├── run_classifier.py
│   ├── run_eval.py            评测脚本
│   ├── run_instruction_finetune_2.py   最终json数据集的微调脚本（这个是正确的）
│   ├── run_instruction_finetune_major.py  初期json数据集的微调脚本（这个废弃最后没用）
│   ├── run_instruction_fintune.py    普通任务（explain，list等）的微调脚本
│   ├── run_instruction_test.py  最终json数据集的测试脚本
│   ├── run_pretrain.py
│   ├── run_pretrain_multi.py
│   └── test_classifier.py
├── src
│   ├── data
│   │   ├── __init__.py
│   │   ├── cls_data.py
│   │   ├── cls_dataloader.py
│   │   ├── data.py
│   │   ├── data_loader.py
│   │   ├── dataloader.py
│   │   ├── instruct_finetune_data_generator.py  生成日常新闻json数据集的脚本
│   │   └── instruction_dataset.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── attention.py
│   │   ├── gpt_model.py
│   │   └── transformer_block.py
│   ├── utils
│   │   ├── __init__.py
│   │   ├── cls_finetune.py
│   │   ├── evaluation.py
│   │   ├── generation.py
│   │   └── metrics.py     评测函数
│   └── __init__.py
├── templates
├── app.py
├── config.py

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

