"""Golden fixtures for lev's agentdecision prompt (test/lev/agentdecision_test.clj).

Runs AgentDecision-4B the way its research server does (jevharness
training/serve_jev.py Engine._plan for the option lines, training/data.py
render and decision_model.chat_text for the prompt, the label logits at the
next token, calibration.json's temperatures), on the merged checkpoint
bench/agentdecision_gguf.py writes, in f32. Writes
golden/agentdecision/requests.json:

  [{"name" "state" "questions" {qid {...}} "prompts" {qid "<chat text>"}
    "logits" {qid [label logits ...]} "probs" {qid [calibrated ...]}} ...]

the logits in prompt order (a noul's yes first), the chat text as the
tokenizer renders it, so the test checks lev's prompt byte for byte too.

    python bench/agentdecision_golden.py ADAPTER_DIR JEVHARNESS_RESEARCH_SRC [golden/agentdecision]

JEVHARNESS_RESEARCH_SRC is github.com/JevAgentHarness/JevAgentHarness's
research/src (imported for _plan and render, so the prompts are the
authors'). Needs torch and transformers.
"""

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REQUESTS = [
    {"name": "ticket",
     "state": {"subject": "Duplicate billing", "body": "We were billed twice. Refund before Friday or we cancel."},
     "questions": {
         "department": {"type": "choice", "instructions": "Which department should handle this?",
                        "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs", "other": None}},
         "urgency": {"type": "score", "instructions": "How urgent is this?",
                     "criteria": ["not urgent", "soon", "critical"]},
         "churn_risk": {"type": "noul", "instructions": "Does the customer threaten to leave?"},
         "is_phishing": {"type": "noul", "instructions": "Is this phishing?",
                         "criteria": {"true": "a scam", "false": "legitimate"}}}},
    {"name": "model card",
     "state": ("TASK: A customer wrote in asking us to ship their pending order (express if they are gold tier, "
               "otherwise standard), but did not say who they are.\n\nHISTORY:\n[0] ask_user({\"question\": \"Could "
               "you please provide your name or customer ID so I can locate your pending order?\", \"missing\": "
               "\"customer_name\"}) -> Customer: Oh, sorry -- my name is Ivan Kim.\n\nCURRENT OBSERVATION:\n"
               "Customer: Oh, sorry -- my name is Ivan Kim."),
     "questions": {
         "tool": {"type": "choice", "instructions": "Which tool should the agent call next?",
                  "criteria": {
                      "find_user": "Look up a customer by full name; returns user_id and tier.",
                      "search_users": "List customers whose last name starts with a prefix (display names only, no ids).",
                      "get_orders": "List a customer's orders filtered by status.",
                      "ship_order": "Ship a pending order.",
                      "ask_user": ("Ask the customer a clarifying question and wait for the reply. Use it only when "
                                   "information needed to act is missing or ambiguous."),
                      "finish": "Call when the task is complete. Provide the final answer (or a short summary)."}},
         "done": {"type": "noul", "instructions": "Is the task complete?"}}},
    {"name": "many options",
     "state": "Pick the tool numbered 23.",
     "questions": {"tool": {"type": "choice", "instructions": "Which tool?",
                            "criteria": {f"tool_{i}": f"tool number {i}" for i in range(30)}}}},
]


def main() -> None:
    adapter, src = Path(sys.argv[1]), Path(sys.argv[2])
    out = Path(sys.argv[3] if len(sys.argv) > 3 else "golden/agentdecision")
    sys.path.insert(0, str(src))
    from jevharness.decision.types import OPTION_LABELS
    from jevharness.training.data import render
    from jevharness.training.decision_model import chat_text
    from jevharness.training.serve_jev import Engine, _bucket

    tok = AutoTokenizer.from_pretrained(adapter)
    model = AutoModelForCausalLM.from_pretrained(adapter / "merged", dtype=torch.float32).eval()
    cal = json.loads((adapter / "calibration.json").read_text())
    codes = [tok.encode(c, add_special_tokens=False)[0] for c in OPTION_LABELS]

    def temperature(kind: str, k: int) -> float:
        b = cal.get("buckets", {})
        return float(b.get(f"{kind}:{_bucket(k)}", b.get(_bucket(k), cal.get("global", 1.0))))

    results = []
    for r in REQUESTS:
        state = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"], ensure_ascii=False)
        prompts, logits, probs = {}, {}, {}
        for qid, q in r["questions"].items():
            kind, instr, opts, _ = Engine._plan(q)
            prompt, _ = render({"state": state, "question": instr, "options": opts})
            text = chat_text(tok, prompt)
            ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids[:, -4096:]
            with torch.no_grad():
                z = model(input_ids=ids).logits[0, -1].float()
            zl = z[codes[: len(opts)]]
            prompts[qid] = text
            logits[qid] = zl.tolist()
            probs[qid] = torch.softmax(zl / temperature(kind, len(opts)), -1).tolist()
        results.append({**r, "prompts": prompts, "logits": logits, "probs": probs})

    out.mkdir(parents=True, exist_ok=True)
    (out / "requests.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print("wrote", out / "requests.json")


if __name__ == "__main__":
    main()
