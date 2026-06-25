import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import torch
import tiktoken
from functools import partial

from config import cfg
from src.models.gpt_model import GPTModel
from src.data.dataloader import create_dataloader
from src.data.instruction_dataset import InstructionDataset, collate_fn
from src.data.cls_dataloader import get_cls_dataloaders
from src.utils.cls_finetune import load_and_modify_cls_model
from src.utils.generation import generate_instruction_response, generate_temp_topk, text_to_ids, ids_to_text
from src.utils.metrics import (
    calc_perplexity, calc_bleu_corpus, calc_instruction_corpus,
    calc_cls_metrics, calc_distinct, eval_json_extraction

)
from src.data.data import load_alpaca_data

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))



def load_gpt_model(checkpoint_name):
    model = GPTModel(
        vocab_size=cfg.vocab_size,
        embed_dim=cfg.embedding_dim,
        context_len=cfg.context_len,
        dropout=cfg.dropout,
        num_heads=cfg.num_heads,
        bias=cfg.bias,
        num_layers=cfg.num_layers,
    ).to(cfg.device)
    path = os.path.join(BASE_DIR, "checkpoints", checkpoint_name)
    model.load_state_dict(torch.load(path, map_location=cfg.device, weights_only=True))
    model.eval()
    print(f"  已加载: {path}")
    return model


def section(title):
    print("\n" + "=" * 55)
    print(f"  {title}")
    print("=" * 55)


# ── 预训练评测（PPL + Distinct）────────────────────────
def eval_pretrain():
    section("预训练模型评测 — PPL + Distinct")

    bin_file = os.path.join(BASE_DIR, "data", "processed", "data.bin")
    if not os.path.exists(bin_file):
        print(f"  ⚠️  找不到预训练数据文件: {bin_file}，跳过预训练评测")
        return None

    val_loader = create_dataloader(
        bin_file,
        split="val",
        batch_size=8,
        max_len=cfg.context_len,
        shuffle=False,
        pin_memory=False,
    )

    model = load_gpt_model("epoch_3.pth")
    tokenizer = tiktoken.get_encoding("gpt2")

    # PPL
    ppl = calc_perplexity(model, val_loader, cfg.device, num_batches=50)
    print(f"  预训练 PPL: {ppl}")

    # Distinct：跑10条续写
    prompts = [
        "The future of", "Once upon a time", "In recent years",
        "Scientists have discovered", "The best way to",
        "Hello, I am", "It is well known that", "The economy",
        "Artificial intelligence", "Human beings"
    ]
    eos_id = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
    generated = []
    for p in prompts:
        idx = text_to_ids(p, tokenizer).to(cfg.device)
        out = generate_temp_topk(model, idx, max_new_tokens=60,
                                 context_size=cfg.context_len,
                                 temp=0.8, top_k=40, eos_id=eos_id)
        generated.append(ids_to_text(out, tokenizer))

    distinct = calc_distinct(generated)
    print(f"  Distinct-1: {distinct['distinct-1']}")
    print(f"  Distinct-2: {distinct['distinct-2']}")

    return {"ppl": ppl, "distinct": distinct}


# ──分类微调评测（Accuracy / Precision / Recall / F1）──
def eval_classifier():
    section("分类微调评测 — Accuracy / Precision / Recall / F1")

    train_csv = os.path.join(BASE_DIR, "data", "cls_raw", "train.csv")
    val_csv = os.path.join(BASE_DIR, "data", "cls_raw", "validation.csv")

    if not os.path.exists(val_csv):
        print(f"    找不到分类数据文件: {val_csv}，跳过分类评测")
        return None

    GPT_CONFIG = {
        "vocab_size": cfg.vocab_size,
        "embed_dim": cfg.embedding_dim,
        "context_len": cfg.context_len,
        "dropout": cfg.dropout,
        "num_heads": cfg.num_heads,
        "bias": cfg.bias,
        "num_layers": cfg.num_layers,
    }

    # 先用预训练权重初始化架构（改输出层为2分类）
    cls_model = load_and_modify_cls_model(
        os.path.join(BASE_DIR, "checkpoints", "epoch_3.pth"),
        GPT_CONFIG, num_classes=2
    )
    # 再用分类微调权重覆盖
    ckpt_path = os.path.join(BASE_DIR, "checkpoints", "classifier_finetuned.pth")
    cls_model.load_state_dict(
        torch.load(ckpt_path, map_location=cfg.device, weights_only=True)
    )
    cls_model.to(cfg.device).eval()

    _, val_loader, _ = get_cls_dataloaders(train_csv, val_csv, batch_size=8)
    metrics = calc_cls_metrics(cls_model, val_loader, cfg.device)
    print(f"  Accuracy:  {metrics['accuracy']:.2%}")
    print(f"  Precision: {metrics['precision']:.2%}")
    print(f"  Recall:    {metrics['recall']:.2%}")
    print(f"  F1:        {metrics['f1']:.2%}")
    return metrics


# ── 指令微调评测（双数据源分流：普通指令流 + JSON 定向流）────────────
def eval_instruct():
    section("指令微调评测 — PPL / BLEU / 指令完成率 / JSON定向提取")

    tokenizer = tiktoken.get_encoding("gpt2")
    collate = partial(collate_fn, pad_token_id=tokenizer.eot_token)

    # 载入微调进化的终极大魔王权重
    model = load_gpt_model("begin_json_epoch5.pt")

    # ====================  数据源一：普通纯任务数据集（Alpaca） ====================
    # 用于测量：原有的 PPL、原有的 BLEU、原有的指令关键词完成率
    data = load_alpaca_data()
    data = [d for d in data if len(d["output"]) > 30]
    test_data_normal = data[-100:]  # 取末尾100条训练没见过的普通任务

    dataset_normal = InstructionDataset(test_data_normal, tokenizer, max_length=cfg.context_len)
    dataloader_normal = torch.utils.data.DataLoader(
        dataset_normal, batch_size=8, shuffle=False, collate_fn=collate
    )

    # 基于普通通用任务计算 PPL
    ppl = calc_perplexity(model, dataloader_normal, cfg.device)
    print(f"  标准指令微调 PPL (基于通用数据集): {ppl}")

    #基于普通通用任务批量生成回答（服务于原有指标）
    print("  正在生成通用任务测试回答（100条）...")
    normal_instructions, normal_references, normal_hypotheses = [], [], []
    for i, item in enumerate(test_data_normal):
        resp = generate_instruction_response(
            model, tokenizer,
            instruction=item["instruction"],
            input_text=item.get("input", ""),
            max_new_tokens=60,
            context_size=cfg.context_len,
            temperature=0.6,
            top_k=40,
            repetition_penalty=1.3,  # 通用文本生成允许使用惩罚项
            device=cfg.device,
        )
        normal_instructions.append(item["instruction"])
        normal_references.append(item["output"])
        normal_hypotheses.append(resp)



    #3-3. 精准回归：原有普通任务的指令完成率及其细分
    instr_scores = calc_instruction_corpus(normal_instructions, normal_hypotheses)
    print(f"  指令完成率（overall）: {instr_scores['overall']:.2%}")
    print(f"  各类别细分: {instr_scores}")

    # ==================== 数据源二：专项结构化 JSON 提取数据集 ====================
    # 用于测量：全新的 Field F1 和 Value F1
    with open("D:/NLP/NLP/data/instruct_test.json") as f:
        json_test = json.load(f)[-50:]
    json_result = eval_json_extraction(
        model, tokenizer, json_test, device=cfg.device
    )

    return {
        "ppl": ppl,
        "instruction_score": instr_scores,
        "json_extraction": json_result
    }


# ── 主函数 ────────────────────────────────────────────────
def main():
    print("\nNLP-LLM 评测系统启动")
    print(f"设备: {cfg.device}")

    results = {}

    pretrain_result = eval_pretrain()
    if pretrain_result is not None:
        results["pretrain"] = pretrain_result

    cls_result = eval_classifier()
    if cls_result is not None:
        results["classifier"] = cls_result

    instruct_result = eval_instruct()
    if instruct_result is not None:
        results["instruct"] = instruct_result

    # 保存结果
    out_dir = os.path.join(BASE_DIR, "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "eval_results_final.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n评测完成，结果已保存到: {out_path}")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()