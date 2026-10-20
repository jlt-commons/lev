"""AgentDecision-4B (JevAgentHarness/AgentDecision-4B, a LoRA adapter on
Qwen/Qwen3-4B) as one GGUF for lev's thinker: the adapter merged into the
base revision it was trained on (decision_config.json), as the model
card's merge_and_unload does, then llama.cpp's convert_hf_to_gguf.

    python bench/agentdecision_gguf.py ADAPTER_DIR BASE_DIR OUT.gguf [q8_0|f16|bf16]

ADAPTER_DIR holds the Hub files (adapter, tokenizer, calibration.json);
BASE_DIR is Qwen/Qwen3-4B at revision 1cfa9a7208912126459214e8b04321603b3df60c.
The merged checkpoint is written next to the adapter (ADAPTER_DIR/merged)
and kept, since bench/agentdecision_golden.py reads it. Needs torch,
transformers and peft (uv run --with peft).
"""

import json
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"


def main() -> None:
    adapter, base, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    outtype = sys.argv[4] if len(sys.argv) > 4 else "q8_0"
    rec = json.loads((adapter / "decision_config.json").read_text())
    if rec.get("base_model_revision") != REVISION:
        sys.exit(f"the adapter was trained on revision {rec.get('base_model_revision')}, this script on {REVISION}")
    merged = adapter / "merged"
    if not (merged / "config.json").exists():
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        model.save_pretrained(merged, safe_serialization=True)
        # the adapter's tokenizer and chat template, as used in training
        AutoTokenizer.from_pretrained(adapter).save_pretrained(merged)
    convert = Path(__file__).resolve().parent.parent / "native" / "llama.cpp" / "convert_hf_to_gguf.py"
    subprocess.run([sys.executable, str(convert), str(merged), "--outtype", outtype, "--outfile", str(out)], check=True)


if __name__ == "__main__":
    main()
