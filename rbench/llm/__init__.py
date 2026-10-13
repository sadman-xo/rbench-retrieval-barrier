"""The LLM stage, for ONE measurement only (Phase 4d E4): does the model obey a
retrieved instruction? No agent, no tools, no full RAG app."""
from .generate import Generation, HFGenerator
from .prompt import build_prompt

__all__ = ["Generation", "HFGenerator", "build_prompt"]
