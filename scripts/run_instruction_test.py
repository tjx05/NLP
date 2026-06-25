# scripts/run_json_test.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import tiktoken

from config import cfg
from src.models.gpt_model import GPTModel
from src.utils.generation import generate_instruction_response
from src.utils.generation import generate_json_response

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ========== 测试用例 ==========
TEST_CASES = [
  {
    "instruction": "Find every mention belonging to the categories 'name', 'agency', 'area', 'year'. Output a JSON object where each field contains unique entity values; use 'N/A' for missing categories.",
    "input": "Barbara Harff and Ted Gurr defined genocide as \"the promotion and execution of policies by a state or its agents which result in the deaths of a substantial portion of a group.\" Daniel D. Polsby and Don B. Kates, Jr. commented on these distinctions in later studies."
  },
  {
    "instruction": "Locate all references to 'person', 'org', 'region', 'year' in the document. Provide unique normalized entities for each category; use 'N/A' where appropriate.",
    "input": "Beyoncé attended St. Mary's Elementary School in Fredericksburg, Texas, and later performed with the school's choir. Dance instructor Darlette Johnson discovered her singing talent, and John Lennon was cited as an influence during her early training."
  },
  {
    "instruction": "Extract the following entities from the text: 'people', 'org', 'place', 'timestamp'. Return each as a list of unique, normalized strings; if none exists, use 'N/A'.",
    "input": "In 2013, the Infectious Disease Society of America (IDSA) reported that the weak antibiotic pipeline does not match bacteria's increasing ability to develop resistance. The number of new antibiotics approved per year in the United States remains low."
  },
  {
    "instruction": "Analyze the text and extract: patient_name, symptoms, diagnosis. Return the result strictly in JSON format.",
    "input": "Patient Michael Turner presented with fever, persistent cough, and shortness of breath. After examination, he was diagnosed with community-acquired pneumonia."
  },
  {
    "instruction": "Convert the unstructured text into structured JSON containing: party_A, party_B, agreement_type, effective_date.",
    "input": "This Software Licensing Agreement is entered into between CloudSphere Technologies and Apex Retail Group, effective January 1, 2026."
  },
  {
    "instruction": "Read the document carefully and retrieve the following entities: candidate_name, university, graduation_year.",
    "input": "Sarah Johnson graduated from Stanford University in 2022 with a Master's degree in Computer Science and currently works as a Machine Learning Engineer."
  },

]

def load_model(checkpoint_name):
    model = GPTModel(
        vocab_size=cfg.vocab_size,
        embed_dim=cfg.embedding_dim,
        context_len=cfg.context_len,
        dropout=cfg.dropout,
        num_heads=cfg.num_heads,
        bias=cfg.bias,
        num_layers=cfg.num_layers
    ).to(cfg.device)

    checkpoint_path = os.path.join(BASE_DIR, "checkpoints", checkpoint_name)
    if not os.path.exists(checkpoint_path):
        print(f" 找不到权重文件: {checkpoint_path}，跳过")
        return None

    model.load_state_dict(torch.load(checkpoint_path, map_location=cfg.device, weights_only=True))
    model.eval()
    print(f"Loaded: {checkpoint_path}\n")
    return model


def run_test(model, tokenizer, label):
    print("=" * 65)
    print(f"  Checkpoint: {label}")
    print("=" * 65)

    for item in TEST_CASES:
        response = generate_json_response(
            model, tokenizer,
            instruction=item["instruction"],
            input_text=item["input"],
            max_new_tokens=150,
            context_size=cfg.context_len,
            temperature=0.0,
            top_k=20,
            repetition_penalty=1.0,
            device=cfg.device
        )
        print(f"[Instruction] {item['instruction']}")
        print(f"[Input]       {item['input']}")
        print(f"[Response]    {response}")
        print("-" * 65)
    print()


def main():
    tokenizer = tiktoken.get_encoding("gpt2")

    # 依次测试三个 epoch 的权重，找出最好的
    for ckpt_name in ["begin_json_epoch5.pt"]:
        model = load_model(ckpt_name)
        if model is not None:
            run_test(model, tokenizer, label=ckpt_name)


if __name__ == "__main__":
    main()