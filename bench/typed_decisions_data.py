"""Build the typed-decisions files the bench and lev.calibrate read, from
LocalLLaMA/typed-decisions on the Hub (Apache-2.0; its parquet files, no auth):
400 test and 1,200 train states from four workflows (invoice processing,
security incidents, customer service, agent-trace observability), five
typed questions each, gold = the mean of three teacher samples.

  bench/data/typed-decisions-test.jsonl    {id workflow state questions gold}
  bench/data/typed-decisions-train.jsonl   the same, for fitting
  bench/data/typed-decisions-calib.jsonl   train as lev.calibrate cases:
                                           {state questions labels targets}

`labels` is each gold's argmax (a choice's option, a score's level index, a
noul's boolean); `targets` the gold distribution in answer order, which
lev.calibrate fits against (soft cross-entropy) when present.

  uv run --with pyarrow python bench/typed_decisions_data.py
"""
import io, json, os, urllib.request

import pyarrow.parquet as pq

DS = "LocalLLaMA/typed-decisions"
OUT = os.path.join(os.path.dirname(__file__), "data")


def rows(split):
    u = f"https://huggingface.co/datasets/{DS}/resolve/main/all/{split}-00000-of-00001.parquet"
    return pq.read_table(io.BytesIO(urllib.request.urlopen(u).read())).to_pylist()


def case(r):
    return {"id": r["id"], "workflow": r["workflow"], "state": json.loads(r["state"]),
            "questions": json.loads(r["questions"]), "gold": json.loads(r["gold"])}


def answer_order(q):
    """The gold's options in the order lev answers them: a choice's criteria
    keys, a score's levels, a noul's [false, true]."""
    t = q["type"]
    if t == "noul":
        return ["false", "true"]
    if t == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    crit = q["criteria"]
    return list(crit) if isinstance(crit, list) else list(crit.keys())


def calib(c):
    labels, targets = {}, {}
    for qid, q in c["questions"].items():
        g = c["gold"][qid]
        keys = answer_order(q)
        targets[qid] = [g["probabilities"].get(k, 0.0) for k in keys]
        lab = g["label"]
        labels[qid] = (lab == "true") if q["type"] == "noul" else (int(lab) if q["type"] == "score" else lab)
    return {"id": c["id"], "state": c["state"], "questions": c["questions"], "labels": labels, "targets": targets}


def write(name, items):
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print("wrote", path, len(items))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    test = [case(r) for r in rows("test")]
    train = [case(r) for r in rows("train")]
    write("typed-decisions-test.jsonl", test)
    write("typed-decisions-train.jsonl", train)
    write("typed-decisions-calib.jsonl", [calib(c) for c in train])
