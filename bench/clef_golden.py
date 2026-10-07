"""Golden fixtures for lev.clef's joint schema head (test/lev/clef_test.clj).

Runs Clef's own JointSchemaHead (joint_schema_model.py from the release,
in f32) on synthetic inputs: random final hidden states for a made-up
record (three questions, one of each type, with token spans laid out as
encode_record lays them) and a small random output embedding the record's
token ids index. Writes golden/clef/:

  hidden.f32   [n x 5120] hidden states
  lm_head.f32  [v x 5120] the output embedding rows the ids index
  head.edn     {:n :v :input-ids [...] :questions [{:type :span [s e]
               :option-spans [[s e] ...]}] :logits [[...] ...]}
  encode.edn   {:request "<JSON>" :input-ids [...] :questions [{:id :type
               :span :option-spans :option-ids}]}: encode_record on a
               request, with the release's own tokenizer (tokenizer.json)

    jolt clef-golden                                   # LEV_TEST_CLEF, else ~/src/models/clef
    python bench/clef_golden.py /path/to/clef [golden/clef]

Needs torch, safetensors and transformers; the head alone is loaded, not
the backbone.
"""

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


def main(clef_dir, out_dir):
    sys.path.insert(0, clef_dir)
    from joint_schema_model import EncodedQuestion, EncodedRecord, JointSchemaHead

    path = Path(clef_dir)
    head = JointSchemaHead(**json.loads((path / "joint_head_config.json").read_text()))
    head.load_state_dict(load_file(path / "joint_head.safetensors"), strict=True)
    head = head.float().eval()

    torch.manual_seed(7)
    n, d, v = 64, head.hidden_norm.normalized_shape[0], 50
    # final-normed hidden states of a real Qwen run are large in a few
    # dimensions; a scaled normal with some heavy channels stands in
    hidden = torch.randn(n, d) * 2.0
    hidden[:, :8] *= 40.0
    lm_head = torch.randn(v, d) * 0.02
    input_ids = torch.randint(0, v, (n,))

    # spans as encode_record makes them: the question's instruction, then
    # each option's JSON, in schema order (noul, choice, score)
    questions = [
        EncodedQuestion("q_noul", 0, (20, 25), ((26, 30), (31, 34)), ("true", "false")),
        EncodedQuestion("q_choice", 1, (36, 40), ((41, 44), (45, 47), (48, 52)), ("a", "b", "c")),
        EncodedQuestion("q_score", 2, (53, 56), ((57, 58), (58, 60), (60, 61), (61, 63)), ("0", "1", "2", "3")),
    ]
    record = EncodedRecord(tuple(input_ids.tolist()), tuple(questions), "golden")
    with torch.inference_mode():
        logits = head(hidden.unsqueeze(0), input_ids.unsqueeze(0), torch.ones(1, n, dtype=torch.long),
                      [record], lm_head)[0]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hidden.numpy().astype("<f4").tofile(out / "hidden.f32")
    lm_head.numpy().astype("<f4").tofile(out / "lm_head.f32")
    types = {0: "noul", 1: "choice", 2: "score"}

    def edn_vec(xs):
        return "[" + " ".join(xs) + "]"

    qs = edn_vec(
        "{:type \"%s\" :span [%d %d] :option-spans %s}"
        % (types[q.question_type], *q.question_span, edn_vec("[%d %d]" % s for s in q.option_spans))
        for q in questions
    )
    lg = edn_vec(edn_vec(repr(float(x)) for x in row.tolist()) for row in logits)
    (out / "head.edn").write_text(
        ";; bench/clef_golden.py: Clef's JointSchemaHead (f32) on synthetic inputs\n"
        "{:n %d :v %d\n :input-ids %s\n :questions %s\n :logits %s}\n"
        % (n, v, edn_vec(str(i) for i in input_ids.tolist()), qs, lg)
    )
    print("wrote", out, [[round(float(x), 4) for x in row] for row in logits])
    write_encode(clef_dir, out)


ENCODE_REQUEST = {
    "state": {"ticket": {"subject": "Checkout down", "body": "Orders fail with a 500 since 9am.",
                         "customer": {"tier": "gold", "name": "Zoë"}}, "attempts": 2},
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle the message?",
            "criteria": {"technical": "Bugs or outages", "billing": "Payments or invoices", "account": None},
        },
        "urgency": {"type": "score", "instructions": "How urgent is it?",
                    "criteria": ["Can wait", "This week", {"level": "today", "sla_hours": 4}]},
        "outage": {"type": "noul", "instructions": "Is a service down?",
                   "criteria": {"true": "Something is unavailable."}},
    },
}


def write_encode(clef_dir, out):
    from joint_schema_model import QUESTION_TYPES, encode_record
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(clef_dir)
    encoded = encode_record(tokenizer, ENCODE_REQUEST)
    types = {v: k for k, v in QUESTION_TYPES.items()}

    def edn_vec(xs):
        return "[" + " ".join(xs) + "]"

    def edn_str(x):
        return json.dumps(x, ensure_ascii=False)

    qs = edn_vec(
        "{:id %s :type \"%s\" :span [%d %d] :option-spans %s :option-ids %s}"
        % (edn_str(q.question_id), types[q.question_type], *q.question_span,
           edn_vec("[%d %d]" % s for s in q.option_spans), edn_vec(edn_str(i) for i in q.option_ids))
        for q in encoded.questions
    )
    (out / "encode.edn").write_text(
        ";; bench/clef_golden.py: Clef's encode_record with the release tokenizer\n"
        "{:request %s\n :input-ids %s\n :questions %s}\n"
        % (edn_str(json.dumps(ENCODE_REQUEST, ensure_ascii=False)), edn_vec(str(i) for i in encoded.input_ids), qs)
    )
    print("wrote encode.edn,", len(encoded.input_ids), "tokens")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "golden/clef")
