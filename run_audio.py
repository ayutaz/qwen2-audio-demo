#!/usr/bin/env python
# run_audio_logits_numeric.py
import argparse, pathlib, torch, torchaudio
from transformers import (AutoProcessor, Qwen2AudioForConditionalGeneration,
                          LogitsProcessor, LogitsProcessorList)

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
LABELS   = ["喜び", "怒り", "悲しみ", "恐れ", "驚き", "嫌悪", "中立", "その他"]
NUM_IDS  = list(range(len(LABELS)))          # [0,1,2,3,4,5,6,7]

# ---------- 1) CLI ----------
ap = argparse.ArgumentParser()
ap.add_argument("audio")
ap.add_argument("text")
ap.add_argument("--8bit", action="store_true")
args = ap.parse_args()

# ---------- 2) processor / model ----------
proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
tok  = proc.tokenizer
load_kw = dict(device_map="auto", trust_remote_code=True)
load_kw["load_in_8bit" if args.__dict__["8bit"] else "torch_dtype"] = (
    True if args.__dict__["8bit"] else torch.float16
)
model = Qwen2AudioForConditionalGeneration.from_pretrained(MODEL_ID, **load_kw).eval()

# ---------- 3) 音声 ----------
wave, sr = torchaudio.load(args.audio)
target_sr = proc.feature_extractor.sampling_rate
if sr != target_sr:
    wave = torchaudio.functional.resample(wave, sr, target_sr)
wav_np = wave.squeeze().cpu().numpy()

# ---------- 4) プロンプト ----------
system_msg = (
    "日本語の感情を分類します。"
    "次の数値から **1 つだけ**返してください。\n"
    + "\n".join(f"{i}:{lbl}" for i, lbl in enumerate(LABELS))
)
prompt = proc.apply_chat_template(
    [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": [
            {"type": "audio", "audio": wav_np},
            {"type": "text",  "text": args.text},
            {"type": "text",  "text": "感情番号は？"}
        ]}
    ],
    tokenize=False, add_generation_prompt=True,
)

inputs = proc(text=prompt, audio=wav_np, sampling_rate=target_sr,
              return_tensors="pt").to(model.device)

# ---------- 5) ロジット制限 (0〜7 のみ) ----------
class AllowedNums(LogitsProcessor):
    def __init__(self, ids): self.ids = torch.tensor(ids)
    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.ids] = 0
        return scores + mask

proc_list = LogitsProcessorList([AllowedNums(NUM_IDS)])

gen = model.generate(**inputs,
                     logits_processor=proc_list,
                     max_new_tokens=1,
                     temperature=0.0, do_sample=False)

num_id = gen[0, -1].item()
print(LABELS[num_id])          # 最終的に日本語ラベルで表示
