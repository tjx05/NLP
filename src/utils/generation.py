import torch
import torch.nn.functional as F


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    最简单的贪心解码生成（每次选概率最高的token）

    输入:
        model:GPT模型
        idx: (batch,seq_len) 起始token ID
        max_new_tokens: 最多生成多少个新token
        context_size: 模型支持的最大上下文长度

    输出:
        idx: (batch,seq_len + generated) 生成的完整序列
    """
    for _ in range(max_new_tokens):
        # 只取最后 context_size 个token（防止超长）
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)  # (batch, seq_len, vocab_size)

            # 只关注最后一个token的输出
            logits_last = logits[:, -1, :]  # (batch, vocab_size)

            # 贪心解码
            probs = F.softmax(logits_last, dim=-1)
            idx_next = torch.argmax(probs, dim=-1, keepdim=True)  # (batch,1)

            # 拼接
            idx = torch.cat((idx, idx_next), dim=1)

    return idx


def generate_temp_topk(model, idx, max_new_tokens, context_size, temp=0, top_k=None, eos_id=None,
                       repetition_penalty=1.0):
    """
    带Temperature和Top-k采样的生成
    temperature: 1.0为标准，<1更确定，>1更随机
    top_k: 只从概率最高的k个token中采样
    repetition_penalty: >1.0时惩罚已出现的token，抑制重复（推荐1.1~1.5）
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # Repetition penalty：对已出现过的token降低logit
        if repetition_penalty != 1.0:
            for token_id in set(idx[0].tolist()):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

        # Top-k过滤
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1:]
            logits = torch.where(logits < min_val, -torch.inf, logits)

        # Temperature缩放
        if temp > 0:
            logits = logits / temp
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            # tem=0时退化为贪心解码
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and idx_next.item() == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def generate_instruction_response(model, tokenizer, instruction, input_text="",
                                  max_new_tokens=100, context_size=1024,
                                  temperature=0.7, top_k=40, device="cpu",
                                  repetition_penalty=1.3):
    """指令微调推理：格式化 prompt 后调用 generate_temp_topk"""
    if input_text.strip():
        prompt = (f"### Instruction:\n{instruction}\n\n"
                  f"### Input:\n{input_text}\n\n"
                  f"### Response:\n")
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

    idx = text_to_ids(prompt, tokenizer).to(device)
    eos_id = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

    out = generate_temp_topk(model, idx, max_new_tokens, context_size,
                             temp=temperature, top_k=top_k, eos_id=eos_id,
                             repetition_penalty=repetition_penalty)

    full_text = ids_to_text(out, tokenizer)
    # 只截取 Response 之后的部分
    return full_text.split("### Response:\n")[-1].strip()


def text_to_ids(text, tokenizer):
    """文本 → token IDs（自动添加batch维度）"""
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    return torch.tensor(encoded).unsqueeze(0)


def ids_to_text(ids, tokenizer):
    """token IDs → 文本（自动移batch维度）"""
    flat = ids.squeeze(0)
    decoded = tokenizer.decode(flat.tolist())
    return decoded

# import tiktoken
# from src.models.gpt_model import GPTModel
# tokenizer=tiktoken.get_encoding('gpt2')
# start_context='Hello, I am'
# encoded=tokenizer.encode(start_context)
# encoded_tensor=torch.tensor(encoded).unsqueeze(0)

# print(encoded_tensor)
# model=GPTModel(50257,768,1024,0.1,12,False,12)
# out=generate_text_simple(
#     model,
#     idx=encoded_tensor,
#     max_new_tokens=6,
#     context_size=1024
# )
# print(out)