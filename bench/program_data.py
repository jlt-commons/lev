"""Build the labelled sets bench/program.clj scores lev.program against:
real noisy inputs, each split into a train half (what a program's author
may look at) and a test half (what it is scored on).

  bench/data/bgl-{train,test}.jsonl     LogHub's BGL_2k sample (Blue Gene/L
                                        RAS log lines, hand-labelled alert
                                        categories, "-" -> "normal"); the
                                        lines of each label alternate between
                                        the halves, in log order
  bench/data/toxic-{train,test}.jsonl   google/civil_comments: 200 toxic
                                        (toxicity >= 0.5) and 200 non-toxic
                                        (< 0.1) comments a half, train from
                                        the validation split, test from the
                                        test split, seeded

Each line is {id text label}.

  uv run --with pyarrow python bench/program_data.py
"""
import csv, io, json, os, random, urllib.request

import pyarrow.parquet as pq

OUT = os.path.join(os.path.dirname(__file__), "data")
BGL = "https://raw.githubusercontent.com/logpai/loghub/master/BGL/BGL_2k.log_structured.csv"
CC = "https://huggingface.co/datasets/google/civil_comments/resolve/refs%2Fconvert%2Fparquet/default/{}/0000.parquet"


def write(name, rows):
    with open(os.path.join(OUT, name), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(name, len(rows))


def bgl():
    text = urllib.request.urlopen(BGL).read().decode("utf-8")
    seen, halves = {}, ([], [])
    for r in csv.DictReader(io.StringIO(text)):
        label = "normal" if r["Label"] == "-" else r["Label"]
        i = seen.get(label, 0)
        seen[label] = i + 1
        line = " ".join([r["Date"], r["Node"], r["Time"], r["Type"], r["Component"], r["Level"], r["Content"]])
        halves[i % 2].append({"id": "bgl-" + r["LineId"], "text": line, "label": label})
    write("bgl-train.jsonl", halves[0])
    write("bgl-test.jsonl", halves[1])


def toxic(split, out, seed):
    t = pq.read_table(io.BytesIO(urllib.request.urlopen(CC.format(split)).read()), columns=["text", "toxicity"])
    rows = [r for r in t.to_pylist() if 20 <= len(r["text"]) <= 600]
    rng = random.Random(seed)
    pos = rng.sample([r for r in rows if r["toxicity"] >= 0.5], 200)
    neg = rng.sample([r for r in rows if r["toxicity"] < 0.1], 200)
    cases = [{"text": r["text"], "label": "toxic"} for r in pos] + [{"text": r["text"], "label": "ok"} for r in neg]
    rng.shuffle(cases)
    write(out, [dict(id=f"{split}-{i}", **c) for i, c in enumerate(cases)])


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    bgl()
    toxic("validation", "toxic-train.jsonl", 1)
    toxic("test", "toxic-test.jsonl", 2)
