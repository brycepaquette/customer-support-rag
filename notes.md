# Customer Support RAG — Evaluation & Design Notes

## Correction (Day 15, 2026-06-05) — Q8 is a corpus bug, not a retrieval bug

While porting the Day 13 questions into `evals/eval_set.json`, I went to look up the ground-truth header name in `corpus/celonis-apis_req-tracing.html` and found the file has **zero `<p>` tags**. It's a 289 KB shell of JS bundle + nav chrome with a single `<h1>API Request Tracing</h1>` and no body content. The page is JS-rendered and the crawler captured an empty shell.

A corpus-wide audit (`<p>` count per file) found **~40 docs (~13% of the corpus) are content-empty stubs**, including `celonis-apis_oauth-authentication.html`, all the `_models_*` schema reference pages, and most glossaries.

**Implications:**

- **Q8's diagnosis below is wrong.** The system refused correctly — there is no header name to retrieve. BM25 can't retrieve content that doesn't exist. The "rare term lost to high-freq token" classification doesn't apply here.
- **Backlog item #1 (BM25) loses one of its three justifying instances.** It still helps MCP and "export" cases, but it's no longer "three confirmed cases of this class."
- **New backlog item (top priority candidate): headless rescrape.** Re-ingest the stub pages with Playwright / a headless browser so JS-rendered content is captured. Without this, the eval upper bound is capped — questions whose ground truth lives only on a stub page can never be answered.
- **Eval-set treatment:** Q4, Q6, and Q8 are encoded as `question_type="edge"` with the literal refusal string as `expected_answer`. Refusal correctness is now a first-class metric.

The original Day 13 analysis is preserved below as the historical record. Strike-through would be dishonest; the misdiagnosis itself is the lesson — answers verified from the corpus only, not from intuition about what the corpus "should" contain.

---

## Day 13 — 10-question eval (2026-06-05)

**Setup:** top_k=3, sentence-transformers all-MiniLM-L6-v2, chromadb (cosine), claude-sonnet-4-6, refusal-mode system prompt, hybrid confidence = `min(claude_self_confidence, top_sim)`.

**Tally:** 8 correct answers, 2 correct refusals, 1 wrong refusal, 0 hallucinations.

| # | Question | Type | Conf | Verdict |
|---|---|---|---|---|
| 1 | How do I authenticate with the Celonis API? | how-to | 0.746 | Excellent — cited 2 sources |
| 2 | How can I install a REST Client? | how-to | 0.389 | Solid; hybrid cap from weak top-1 |
| 3 | What are the rate limits for the API? | factual | 0.588 | Strong; synthesized across 2 sources |
| 4 | How do I export data from a data pool? | edge | 0.000 | Correct refusal (retrieved ingestion docs) |
| 5 | How to retrieve a record by its ID? | how-to | 0.311 | Drew from rank-2; hybrid honestly capped |
| 6 | What is the capital of France? | OOD | 0.000 | Correct refusal |
| 7 | What HTTP status code on rate limit exceeded? | factual | 0.508 | Correct "429" |
| 8 | What header is used for request tracing? | factual | 0.000 | **Wrong refusal — retrieval failure** |
| 9 | What is the SCIM API used for? | conceptual | 0.529 | Strong |
| 10 | Difference between basic auth and OAuth in Celonis APIs? | conceptual | 0.693 | Best synthesis of the run |

### Failure deep-dive: Q8 "What header is used for request tracing?"

**Classification:** retrieval failure (corpus gap = none — `celonis-apis_req-tracing.html` exists).

**What happened:** top-3 returned three `cpm/developer/services/graphic/request/json` chunks (scores 0.391 / 0.347 / 0.340). The correct doc never surfaced. Claude refused correctly given the irrelevant context.

**Root cause:** "request" is a high-frequency token across the corpus (HTTP requests, JSON requests, request bodies). Dense embeddings let the generic word dominate; the rare, discriminative term "tracing" gets averaged out. all-MiniLM-L6-v2 (general-web-trained) can't distinguish "request tracing" as a feature name from incidental co-occurrence of the two words.

**Pattern:** third instance of this failure class this month —
- Day 11: "MCP API" (rare acronym lost to generic context)
- Day 11: "export from data pool" (wrong-direction retrieval)
- Day 13: "request tracing" (rare term lost to high-freq token)

**Prescription (Week 3):** BM25 hybrid retrieval. BM25 weights "tracing" by IDF (likely <10 chunks contain it vs. hundreds for "request"), so `req-tracing.html` would rank near the top. Combine with dense via reciprocal rank fusion. Standard prescription for technical docs.

### Wins worth keeping as evidence

- **Q5 — rank-2 rescue.** Rank-1 was a wrong `models/pool` chunk (0.311); rank-2 was the correct `retrieve-record-data` doc (0.281). Claude drew from rank-2 and cited only it. Hybrid confidence honestly capped at 0.311 = top_sim. Vindicates k≥3 over k=1.
- **Q10 — best synthesis.** Claude built a structured contrast between basic auth and OAuth from 3 chunks that never explicitly compare the two. Inferred from facts spread across auth + migration docs. Upper bound of what dense retrieval + Claude can do without a re-ranker.
- **Q4 + Q6 — refusal contract holds under two different stresses:** wrong-direction retrieval (Q4 returned ingestion docs for an "export" question) and pure OOD (Q6). Both refused at confidence 0.0.

### Confidence calibration

Confidence tracks quality directionally. Floor (refusals = 0.0) is hard. Ceiling (great answer ≈ 0.75) is soft — Claude rarely self-reports >0.97 even when correct, and hybrid min() pulls the score down toward retrieval similarity which tops out around 0.7 for this embedding model. Not a user-facing score — a downstream-trust signal for routing/escalation.

### Cost (from Langfuse export, 10 questions)

- **Total run cost:** $0.0929 (input 19,126 tokens; output 2,367 tokens)
- **Per-question avg:** $0.0093
- **Range:** $0.0056 (Q7, short factual) — $0.0152 (Q1, detailed answer)
- **Implied:** ~$0.93/day at 100 queries/day, ~$9.30/day at 1000. Generation dominates; embedding is rounding error.

---

## Improvement backlog (priority order)

> Updated Day 15: headless rescrape promoted to #1 after corpus-stub discovery. See correction at top.

0. **Headless rescrape of stub pages (Day 15 discovery)** — ~40 corpus files captured as JS-rendered shells with no body content (`req-tracing`, `oauth-authentication`, all `_models_*`, most glossaries). Without this, the corpus has a hard recall ceiling.
1. **BM25 hybrid retrieval** — fixes "MCP" / "export" class (2 confirmed instances; Q8 reclassified as corpus bug, see correction).
2. **Cross-encoder re-rank on top-10** — would fix Q5 rank-2 case and similar near-misses without changing the index.
3. **Strip boilerplate from chunks** — repeated nav/footer/"Copy for LLM" text dilutes embeddings; visible in low scores across the board.
4. **k=5 instead of k=3** — cheap; might let Claude synthesize from rank-4/5 chunks.
5. **Per-doc-family embeddings** — `cpm/developer/*` and `celonis-apis/*` may need separate handling; corpus mixes very different document shapes.

---

## Design reflection (feeds Week 4 README "Design Decisions")

### Chunking strategy

**What we did:** sentence-window splitting at ~512 chars with overlap, one chunk per logical paragraph where possible. 304 chunks across 60+ HTML docs.

**Why:** small enough to fit several into a Claude context window at k=3 (~2k input tokens total), large enough to carry self-contained meaning. HTML structure was discarded — pure text.

**What's wrong with it:** boilerplate (nav links, "Copy for LLM" banners, "Open in ChatGPT" buttons) leaks into chunks and dilutes the embeddings. Most chunks have ~30% noise tokens. A proper HTML-aware splitter that preserves headings as metadata would likely raise top_sim across the board by 0.1-0.2.

### Embedding model

**What we picked:** sentence-transformers all-MiniLM-L6-v2 (384-dim, ~22M params, free, local).

**Why:** zero API cost, fast (<1s for 304 chunks), no rate limits, good enough for a Day 10 demo.

**Trade-off:** trained on general web text. Doesn't know that "SCIM", "MCP", "req-tracing" are first-class entities in this corpus. Every failure this month traces back to this: rare technical terms can't punch through generic context.

**What I'd change:** either (a) BM25 hybrid (cheapest fix, addresses the root failure mode without re-embedding), or (b) a code/docs-trained embedding model (`bge-base-en-v1.5`, `nomic-embed-text-v1.5`). (a) first because it's a 1-day change; (b) only if (a) plateaus.

### Retrieval gaps

- No re-ranker. The rank-2 rescue (Q5) is a one-off — Claude can't always pull a correct answer from rank-2 when rank-1 dominates the context. A cross-encoder re-rank would let us drop the hybrid `min()` floor and trust top-1 more.
- No query rewriting. Q8 "What header is used for request tracing?" could be rewritten by Claude to "x-trace-id correlation-id Celonis HTTP tracing header" before embedding. Expands the keyword surface area for dense retrieval.
- No multi-hop. Anything requiring stitching across 4+ documents is out of reach.

### What I'd do differently if starting over

1. **HTML-aware chunking first.** Preserving section headings as chunk metadata is essentially free and would lift retrieval across the entire eval.
2. **BM25 from day 1.** I deferred it as a "Week 3 optimization." It's actually a baseline, not an optimization — dense-only retrieval fails on rare technical terms by design.
3. **Confidence as a contract, not a number.** The hybrid `min(claude_conf, top_sim)` works but the units are arbitrary. A calibrated score (Platt scaling against a held-out eval) would be more honest.
4. **Eval set built before the system, not after.** Several Day 13 questions were retro-fitted to test specific failure modes I already knew about. A pre-committed eval set would have caught Q8 on Day 11.

### What worked

- Refusal-mode prompt + hybrid confidence: 0 hallucinations across 10 questions including two adversarial probes (Q4 wrong-direction, Q6 OOD). The refusal contract holds.
- Server-side source filter against retrieved URLs: prevents Claude from inventing citation URLs even when it tries.
- Langfuse `@observe` instrumentation: per-question cost/latency/token counts available without writing logging code. Made the cost audit a 5-minute job.
