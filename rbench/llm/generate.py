"""Thin wrapper over a Hugging Face instruct model: batched, greedy, chat template.

Greedy decoding makes every answer reproducible. `truncated` marks answers that hit
max_new_tokens: the canary goes at the END of the reply, so a cut-off answer can hide
an obeyed instruction -- report the truncated share next to every ISR.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Generation:
    text: str
    truncated: bool          # stopped at max_new_tokens, not at an end-of-turn token


class HFGenerator:
    def __init__(self, model_name: str, device: str = "cpu", max_new_tokens: int = 256,
                 batch_size: int = 8) -> None:
        # Imported lazily so `import rbench` works without loading an LLM stack.
        import torch
        import transformers
        from packaging.version import Version
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.padding_side = "left"                # decoder-only: pad on the left
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        # `dtype` replaced `torch_dtype` in transformers 4.56
        key = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **{key: dtype}).to(device)
        self.model.eval()
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos]) | {self.tok.eos_token_id}
        self.eos_ids.discard(None)

    def _chat(self, user_message: str) -> str:
        return self.tok.apply_chat_template([{"role": "user", "content": user_message}],
                                            tokenize=False, add_generation_prompt=True)

    def generate_batch(self, user_messages: list[str]) -> list[Generation]:
        """Generate answers for one batch (up to batch_size prompts)."""
        import torch

        batch = [self._chat(m) for m in user_messages]
        enc = self.tok(batch, return_tensors="pt", padding=True,
                       add_special_tokens=False).to(self.device)
        with torch.no_grad():
            gen = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                      do_sample=False, pad_token_id=self.tok.pad_token_id)
        out: list[Generation] = []
        for row in gen[:, enc["input_ids"].shape[1]:].tolist():
            ended = any(t in self.eos_ids for t in row)
            out.append(Generation(text=self.tok.decode(row, skip_special_tokens=True).strip(),
                                  truncated=not ended and len(row) >= self.max_new_tokens))
        return out

    def generate(self, user_messages: list[str]) -> list[Generation]:
        out: list[Generation] = []
        for s in range(0, len(user_messages), self.batch_size):
            out.extend(self.generate_batch(user_messages[s:s + self.batch_size]))
        return out
