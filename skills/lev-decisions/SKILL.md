---
name: lev-decisions
description: Make typed, calibrated decisions about text or JSON with lev, a local decision engine: classify (choice), rate (score) or check a yes/no statement (noul), with probabilities you can threshold. Use it to triage tickets and emails, route requests, moderate posts, screen prompts for jailbreaks or injections, or any step where the agent needs a quick judgement to act on instead of reasoning it out in text. Works through lev's MCP server (the `decide` tool) or its HTTP API.
license: Apache-2.0
---

# Typed decisions with lev

lev answers typed questions about a **state** (a message, an email, a ticket, any JSON) with a
calibrated answer to each question. Two kinds of model sit behind it:

- the **encoders** (`english`, `multilingual`, `typed-decisions`): one forward pass for all of a
  call's questions, about 100 ms on a laptop CPU;
- **thinkers** (GGUF chat models, e.g. `qwen3.5-4b`): slower (~170 ms a question on a GPU, seconds
  with thinking) and much more accurate on questions that need a deduction.

It never writes text. Reach for it when the answer is one of a known set of outcomes and you will
act on it: route, label, block, escalate, pick a template. Keep reasoning in text for open-ended
work.

## When a typed decision is the right tool

| Situation | Use |
|---|---|
| Label, route or filter many items (tickets, emails, messages, rows) | `decide` on each item |
| A yes/no gate before an action (spam, a jailbreak, a refund request) | a `noul` question |
| A rating compared with a threshold (urgency, severity, frustration) | a `score` question |
| One of N categories, teams or intents | a `choice` question |
| Summaries, explanations, answers that need new text | not this skill |

## How to call it

1. **MCP** (server `lev`, set up with `claude mcp add lev -- /path/to/lev-server --mcp`):
   - `decide` `{state, questions, model?, constraints?, thinking?, escalate?}`: the same body and
     answer as `POST /v1/systemone`.
   - `run_workflow` `{name, input, model?, options?}`: a ready-made question set on its input.
   - `list_models`, `list_workflows`; resources `lev://workflows/<name>` show each workflow's
     questions, to reuse or adapt.
2. **HTTP** (`jolt -M:serve` or `./lev-server`, `http://127.0.0.1:8080` by default), compatible
   with TypeSafe's Jev API:
   ```sh
   curl -s http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' \
     -d '{"state": "…", "questions": {…}}'
   curl -s http://127.0.0.1:8080/v1/workflows/triage -d '{"input": "I was charged twice."}'
   ```

## Picking a model

Leave `model` out to route by content to an encoder (`english` for English, `multilingual`
otherwise). Name a thinker when the question needs reading between the lines, or better, let the
encoder answer and escalate only what it is unsure of:

```json
{"state": "…", "questions": {…}, "escalate": {"model": "qwen3.5-4b", "threshold": 0.5}}
```

On the adversarial authored144 set that answers 92% at 270 ms a case, against 61% for the encoder
alone and 95% for the thinker alone. `list_models` shows what this server has.

## Workflows

Bundled question sets, run with `run_workflow` or `POST /v1/workflows/<name>`:

- `triage`: intent, is_urgent, frustration (0–3), refund_requested, churn_risk. Input: a
  customer `message`.
- `email`: category, spam, phishing, urgency, needs a reply. Input: `{subject, body, from}`, or
  the body as a string; the body is cleaned of quotes and signatures.
- `guard`: jailbreak, prompt_injection, sensitive_data, harm_severity, topic. Input: a `prompt`.
- `moderation`: toxic, harassment, threat, spam, severity. Input: a `post`.
- `llm-router`: difficulty, domain, needs_tools, is_sensitive. Input: a `request`.
- `security`: event type, active threat, severity, with constraints. Input: an `event`.

A bare string input is taken as the field the workflow names.

## Writing your own questions

Questions are an object keyed by an id you choose. Their order and the order of the options are
kept, and are part of the model's input:

```json
{
  "team": {
    "type": "choice",
    "instructions": "Which team should handle `ticket`?",
    "criteria": {
      "billing": "invoices, charges, refunds, plans",
      "engineering": "bugs, outages, errors, integrations",
      "sales": "pricing, demos, upgrades",
      "other": "anything else"
    }
  },
  "urgency": {
    "type": "score",
    "instructions": "How urgent is `ticket`?",
    "criteria": ["can wait", "this week", "today", "right now: an outage or a deadline"]
  },
  "wants_refund": {
    "type": "noul",
    "instructions": "Does the customer ask for their money back?"
  }
}
```

- **choice**: a short description for every option, mutually exclusive options, a catch-all
  (`other`).
- **score**: 3–5 levels, lowest first, each described concretely. The answer is the expected
  level (it can fall between levels), with a `legend`.
- **noul**: one statement that is true or false. No double negatives, no two questions in one.
- Name the state's field in backticks (`` `ticket` ``) when the state is an object.
- Keep instructions short and literal.
- `constraints` tie answers together (a benign event is no threat); see `lev://workflows/security`.

## Reading answers and acting on them

| `type` | Fields |
|---|---|
| `choice` | `choice`, `confidence` (0–1), `probabilities` per option |
| `score` | `score` (expected level), `confidence`, `legend`, `probabilities` per level |
| `noul` | `noul`: the probability that the statement is true, and `confidence` |

Act on an answer only when it is clear, and escalate otherwise:

- `choice`: act when `confidence` ≥ 0.6; below that, treat the top two options as candidates,
  escalate to a thinker, or ask a human.
- `noul`: treat ≥ 0.8 as yes and ≤ 0.2 as no; in between, escalate or look closer.
- `score`: compare `score` with your threshold and check `confidence` first.

Say in your output which model answered and the probability behind each action, so a person can
audit it.

## Example: triage a batch of tickets

1. For each ticket, call `run_workflow` with `{"name": "triage", "input": ticket_text}`, or
   `decide` with your own `team` / `urgency` questions.
2. Route by `intent`, flag `is_urgent` ≥ 0.8 or `frustration` ≥ 2, and send `churn_risk` ≥ 0.8 to
   a retention queue. Put every ticket whose intent has low confidence on a "needs review" list, or
   re-ask those with `escalate`.
3. Report a table: ticket, action, and the probabilities behind it.
