# src/utils/metrics.py
import torch
import torch.nn.functional as F
import math
import json
from collections import Counter
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import torch




# ── 1. 困惑度 PPL ──────────────────────────────────────────
def calc_perplexity(model, data_loader, device, num_batches=None):
    """
    计算困惑度 PPL = exp(平均交叉熵loss)
    适用于预训练和指令微调模型
    ignore_index=-100 兼容指令微调的label mask
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0

    if num_batches is None:
        num_batches = len(data_loader)

    with torch.no_grad():
        for i, (input_ids, labels) in enumerate(data_loader):
            if i >= num_batches:
                break
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100
            )
            total_loss += loss.item()
            total_batches += 1

    avg_loss = total_loss / total_batches
    ppl = math.exp(avg_loss)
    return round(ppl, 4)


# ── 2. BLEU 分数─────────────────
def calc_bleu(reference: str, hypothesis: str) -> float:
    """单条 BLEU"""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    smoothie = SmoothingFunction().method1  # 防止短句得0分
    score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)
    return round(score, 4)


def calc_bleu_corpus(references: list, hypotheses: list) -> float:
    """整批 BLEU"""
    refs = [[r.lower().split()] for r in references]
    hyps = [h.lower().split() for h in hypotheses]
    score = corpus_bleu(refs, hyps)
    return round(score, 4)


# ── 3. 分类准确率 ──────────────────────────────────────────
def calc_accuracy(model, data_loader, device):
    """
    计算分类准确率
    model 输出 logits，取最后一个 token 作为分类依据（和 cls_finetune 一致）
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            logits = model(input_ids)[:, -1, :]  # 最后一个token
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return round(correct / total, 4) if total > 0 else 0.0


# ── 4. 指令完成效果（关键词命中率）────────────────────────
INSTRUCTION_KEYWORDS = {
    "explain": ["is a", "refers to", "means", "defined as",
                "used to", "which", "because"],
    "write": ["the", "a", "an", "is", "was", "were"],  # 写作类宽松判断
    "list": ["1.", "2.", "3.", "-", "first", "second", "third"],
}


def _detect_category(instruction: str) -> str:
    instr_lower = instruction.lower()
    for cat in INSTRUCTION_KEYWORDS:
        if cat in instr_lower:
            return cat
    return "write"  # 默认宽松判断


def calc_instruction_score(instruction: str, response: str) -> float:
    """
    单条指令完成度评估（0 或 1）
    检查 response 里是否包含与指令类型匹配的关键词
    """
    if not response.strip():
        return 0.0
    cat = _detect_category(instruction)
    keywords = INSTRUCTION_KEYWORDS[cat]
    resp_lower = response.lower()
    hit = any(kw in resp_lower for kw in keywords)
    return 1.0 if hit else 0.0


def calc_instruction_corpus(instructions: list, responses: list) -> dict:
    """
    对一批指令-回答对计算整体指令完成率
    返回总完成率 + 各类别细分
    """
    category_scores = {}
    for instr, resp in zip(instructions, responses):
        cat = _detect_category(instr)
        score = calc_instruction_score(instr, resp)
        if cat not in category_scores:
            category_scores[cat] = []
        category_scores[cat].append(score)

    result = {}
    all_scores = []
    for cat, scores in category_scores.items():
        avg = round(sum(scores) / len(scores), 4)
        result[cat] = avg
        all_scores.extend(scores)

    result["overall"] = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0
    return result


# ── 5.分类微调召回率────────────────────────
def calc_cls_metrics(model, data_loader, device):
    """计算分类完整指标：Accuracy / Precision / Recall / F1"""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(device)
            logits = model(input_ids)[:, -1, :]
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    tp = sum(p == 1 and l == 1 for p, l in zip(all_preds, all_labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(all_preds, all_labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(all_preds, all_labels))
    tn = sum(p == 0 and l == 0 for p, l in zip(all_preds, all_labels))

    accuracy = round((tp + tn) / len(all_labels), 4)
    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    }


# ── 6.预训练的生成多样性评测────────────────────────
def calc_distinct(texts: list) -> dict:
    unigrams, bigrams = [], []
    for text in texts:
        tokens = text.lower().split()
        unigrams.extend(tokens)
        bigrams.extend([tuple(tokens[i:i + 2]) for i in range(len(tokens) - 1)])  # 直接写死

    d1 = round(len(set(unigrams)) / len(unigrams), 4) if unigrams else 0.0
    d2 = round(len(set(bigrams)) / len(bigrams), 4) if bigrams else 0.0
    return {"distinct-1": d1, "distinct-2": d2}


# ── 7. 专门针对 JSON 强约束抽取的评测指标 ──────────────────────────
import json

# ── JSON 提取评测 ──────────────────────────────────────────

def parse_json_safe(text: str):
    """安全解析 JSON，失败返回 None"""
    try:
        return json.loads(text.strip())
    except:
        # 尝试找到 { } 之间的内容再解析
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except:
                return None
        return None


def calc_json_validity(hypotheses: list) -> float:
    """
    JSON 合法率：生成结果里能被成功解析为 JSON 的比例
    最基础的指标，衡量模型输出格式是否正确
    """
    valid = sum(1 for h in hypotheses if parse_json_safe(h) is not None)
    return round(valid / len(hypotheses), 4) if hypotheses else 0.0


def calc_field_coverage(references: list, hypotheses: list) -> dict:
    """
    字段覆盖率：生成的 JSON 里包含了多少标准答案要求的字段
    比如标准答案有 {patient_name, symptoms, diagnosis} 三个字段
    模型生成了 {patient_name, diagnosis} 两个，覆盖率 = 2/3
    """
    total_required = 0
    total_covered  = 0

    for ref_text, hyp_text in zip(references, hypotheses):
        ref = parse_json_safe(ref_text)
        hyp = parse_json_safe(hyp_text)
        if ref is None:
            continue
        required_keys = set(ref.keys())
        covered_keys  = set(hyp.keys()) if hyp else set()
        total_required += len(required_keys)
        total_covered  += len(required_keys & covered_keys)

    coverage = round(total_covered / total_required, 4) if total_required > 0 else 0.0
    return {"field_coverage": coverage,
            "total_required": total_required,
            "total_covered":  total_covered}


def calc_value_accuracy(references: list, hypotheses: list) -> dict:
    """
    值准确率：字段值和标准答案完全匹配的比例
    分两种：完全匹配（exact）和包含匹配（contains，更宽松）
    """
    exact_hits   = 0
    contain_hits = 0
    total_fields = 0

    for ref_text, hyp_text in zip(references, hypotheses):
        ref = parse_json_safe(ref_text)
        hyp = parse_json_safe(hyp_text)
        if ref is None or hyp is None:
            continue
        for key in ref:
            if key not in hyp:
                continue
            total_fields += 1
            ref_val = str(ref[key]).lower().strip()
            hyp_val = str(hyp[key]).lower().strip()
            if ref_val == hyp_val:
                exact_hits += 1
            if ref_val in hyp_val or hyp_val in ref_val:
                contain_hits += 1

    exact   = round(exact_hits   / total_fields, 4) if total_fields > 0 else 0.0
    contain = round(contain_hits / total_fields, 4) if total_fields > 0 else 0.0
    return {"exact_match": exact, "contain_match": contain, "total_fields": total_fields}


def eval_json_extraction(model, tokenizer, test_data: list, device="cpu",
                         max_new_tokens=150, context_size=1024,
                         temperature=0.3, top_k=20, repetition_penalty=1.2) -> dict:
    """
    JSON 提取任务完整评测入口
    test_data 格式：[{"instruction": "...", "input": "...", "output": "{...}"}, ...]
    output 字段是标准答案 JSON 字符串
    """
    from src.utils.generation import generate_json_response

    references, hypotheses = [], []
    print(f"  正在生成 {len(test_data)} 条 JSON 提取结果...")

    for i, item in enumerate(test_data):
        resp = generate_json_response(
            model, tokenizer,
            instruction=item["instruction"],
            input_text=item.get("input", ""),
            max_new_tokens=max_new_tokens,
            context_size=context_size,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            device=device,
        )
        references.append(item.get("output", "{}"))
        hypotheses.append(resp)
        if (i + 1) % 20 == 0:
            print(f"    已完成 {i+1}/{len(test_data)}")

    # 三个维度评测
    validity  = calc_json_validity(hypotheses)
    coverage  = calc_field_coverage(references, hypotheses)
    accuracy  = calc_value_accuracy(references, hypotheses)

    result = {
        "json_validity":   validity,          # 格式合法率
        "field_coverage":  coverage["field_coverage"],  # 字段覆盖率
        "exact_match":     accuracy["exact_match"],     # 值完全匹配率
        "contain_match":   accuracy["contain_match"],   # 值包含匹配率
        "total_samples":   len(test_data),
        "total_fields":    accuracy["total_fields"],
    }

    print(f"  JSON 合法率:   {validity:.2%}")
    print(f"  字段覆盖率:    {coverage['field_coverage']:.2%}")
    print(f"  值完全匹配率:  {accuracy['exact_match']:.2%}")
    print(f"  值包含匹配率:  {accuracy['contain_match']:.2%}")
    return result