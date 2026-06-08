# Colby Professor Reviews — RAG System (Project 1)

A retrieval-augmented question answering system over student reviews of Colby
College professors. Ask a natural-language question and it retrieves the most
relevant reviews, then answers using only those reviews with the source files
cited.

**How to run:**

```bash
# one-time setup (already done in .venv)
.venv/bin/python -m pip install -r requirements.txt

# rebuild the pipeline if needed
.venv/bin/python ingest.py   # data/ -> chunks.json
.venv/bin/python embed.py    # chunks.json -> ChromaDB

# launch the web UI
.venv/bin/python app.py      # http://localhost:7860
```

A real `GROQ_API_KEY` must be in `.env` for generation to work.

---

## Domain

Student reviews of Colby College professors, sourced from Rate My Professors.
This knowledge is valuable because the official course catalog only describes
*what* a class coverrs, it says nothing about teaching style, grading
strictness, workload, or whether a professor actually helps you learn. Students
get that information from peers, but there's no way to query across many
professors at once. This system lets a student ask one question (e.g. "who
gives helpful feedback?") and get a synthesized answer drawn from dozens of
real reviews.

---

## Document Sources

Each professor's reviews are stored as one plain-text file in `data/`. I
couldn't scrape the live pages with `requests`/BeautifulSoup because Rate My
Professors renders reviews client-side with JavaScript, so I pulled the review
text through RMP's GraphQL API and saved one file per professor (name,
department, then each review with its class / quality / difficulty line).

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | Rate My Professors — David W. Findlay | Reviews (.txt) | data/davidfindlay.txt (id 132637) |
| 2 | Rate My Professors — Robert T. Bluhm Jr. | Reviews (.txt) | data/robertbluhm.txt (id 342085) |
| 3 | Rate My Professors — Kyla Pohl | Reviews (.txt) | data/kylapohl.txt (id 3159202) |
| 4 | Rate My Professors — Julie T. Millard | Reviews (.txt) | data/juliemillard.txt (id 952332) |
| 5 | Rate My Professors — Nora Youngs | Reviews (.txt) | data/norayoungs.txt (id 2232428) |
| 6 | Rate My Professors — Daniel H. Cohen | Reviews (.txt) | data/danielcohen.txt (id 63976) |
| 7 | Rate My Professors — Nikky-Guninder K. Singh | Reviews (.txt) | data/nikkysingh.txt (id 64321) |
| 8 | Rate My Professors — Adam Howard | Reviews (.txt) | data/adamhoward.txt (id 328533) |
| 9 | Rate My Professors — Elizabeth Ketner | Reviews (.txt) | data/elizabethketner.txt (id 2054082) |
| 10 | Rate My Professors — Jonathan H. McCoy | Reviews (.txt) | data/jonathanmccoy.txt (id 1350504) |

Together the ten professors span Economics, Philosophy, Religious Studies,
Chemistry, Writing, Physics, Education, and Math, so the corpus covers a range
of subjects and teaching styles. 207 reviews total.

---

## Chunking Strategy

**Chunk size:** 256 tokens

**Overlap:** 50 tokens

**Why these choices fit your documents:** RMP reviews are short (1–5 sentences
each), so a single review is far too small to be a chunk on its own, it loses
context and the embeddings become noisy. Instead of cutting blindly every N
tokens, my chunker (`ingest.py`) packs whole reviews together until adding the
next one would exceed 256 tokens, which fits roughly 2–4 reviews per chunk.
This keeps every review intact (no chunk ends mid-sentence) and keeps related
opinions about the same professor together. The 50-token overlap carries the
last review of one chunk into the next so a review on a boundary isn't isolated
from its neighbors. Token counts are measured with MiniLM's own tokenizer so
"256 tokens" means the same thing here as it does at embedding time.

Preprocessing before chunking: HTML-entity unescaping (the API text still
contained `&#39;`, `&quot;`, `&#128016;`), stripping any stray HTML tags, and
collapsing extra blank lines. The professor's name is prepended to every chunk
so a retrieved chunk is self-contained and can't be misattributed.

**Final chunk count:** 93 chunks (avg ~219 tokens, range 126–263).

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers`, stored and
queried through ChromaDB (a persistent local collection). It's small, fast,
runs locally with no API cost, and is a solid general-purpose sentence
embedder — appropriate for a class project with a small corpus. Retrieval uses
**top-k = 5**.

**Production tradeoff reflection:** For real users I'd consider a larger,
API-hosted model like OpenAI's `text-embedding-3-large`. The upsides:
better accuracy on the informal, slang-heavy language students use ("ruthless
grader," "easy A," "carry the curve") and a much larger context window. The
downsides are real though — per-query latency and cost, plus the privacy
question of sending review text to a third party. The biggest concrete limit I
hit with MiniLM is its **256-token max sequence length**: anything longer is
silently truncated, which is exactly why I capped my chunk size at 256 (see
Spec Reflection). A model with a larger context window would let me use bigger
chunks and keep more reviews together per embedding.

---

## Grounded Generation

**System prompt grounding instruction:** The model (Groq
`llama-3.3-70b-versatile`, `temperature=0`) is given a system prompt that
*enforces* grounding rather than suggesting it. The actual instructions:

- "Answer using ONLY the information in the context documents the user provides."
- "Do NOT use any outside or prior knowledge about these professors or Colby."
- "If the context does not contain enough information to answer the question,
  reply with exactly this sentence and nothing else: *I don't have enough
  information on that.*"
- "Do not invent professors, classes, ratings, or opinions that are not in the
  context."

The retrieved chunks are formatted into a numbered context block
(`[Document N - source: <file>]`) and passed in the user message, so the model
sees exactly which file each piece of evidence came from.

**How source attribution is surfaced in the response:** Attribution is done
**in code, not left to the model**. After generation, `ask()` collects the
unique source files from the chunks that were actually retrieved and returns
them as a `sources` list — so the citation reflects what the retriever really
pulled, even if the model forgets to cite. If the model returns the exact
"not enough information" sentence, the sources list is emptied so the UI
doesn't show reviews it didn't actually use.

---

## Evaluation Report

All five questions were run through the live system. Responses are summarized;
quoted phrases below were spot-checked against the source files and are
verbatim from real reviews.

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | Which Colby professors are known for being engaging in lectures? | Names of professors described as engaging/enthusiastic | Howard, McCoy, Findlay — quotes McCoy as "funny and animated," Findlay as an "exciting lecturer." Sources: adamhoward, jonathanmccoy, davidfindlay | Relevant | Accurate |
| 2 | Which professors have the lightest workload? | Names of professors with manageable workloads | Picks Findlay's MACROTHEORY (difficulty 1), Bluhm, McCoy — reasons from *difficulty ratings*, even noting "difficulty and workload are not always the same thing." Sources: davidfindlay, robertbluhm, norayoungs, jonathanmccoy | Partially relevant | Partially accurate |
| 3 | Which professors are recommended for students new to a subject? | Names of approachable/clear professors | Howard, Cohen, McCoy — grounded on Cohen's "easy-going philosophy class … for the incoming freshman" and McCoy focusing on "concepts rather than equations." Sources: adamhoward, davidfindlay, danielcohen, jonathanmccoy | Relevant | Partially accurate |
| 4 | What do students say about grading fairness at Colby? | Summary of fair/unfair grading reviews | Balanced summary: Findlay "tough grader but … demands excellence," Millard a "ruthless grader" vs. another calling her "difficult but fair." Sources: davidfindlay, juliemillard, danielcohen | Relevant | Accurate |
| 5 | Which professors give the most helpful feedback on assignments? | Names noted for detailed/useful feedback | Ketner "always gives good feedback" (grounded); also names Findlay and Youngs but with hedging ("implies," "suggests"). Sources: davidfindlay, elizabethketner, nikkysingh, adamhoward, norayoungs | Partially relevant | Partially accurate |

**Retrieval quality:** Relevant / Partially relevant / Off-target
**Response accuracy:** Accurate / Partially accurate / Inaccurate

Q1 and Q4 are clean wins — the retrieved chunks directly contain the answer and
every claim is traceable to a quote. Q3 and Q5 are partially accurate because
the model adds reasonable but unsupported inferences (e.g. inferring Youngs
gives good *feedback* from a review that only says she explains material well).
Q2 is the clearest failure and is analyzed below.

---

## Failure Case Analysis

**Question that failed:** "Which professors have the lightest workload?" (Q2)

**What the system returned:** It answered by comparing **difficulty ratings**
(picking Findlay's MACROTHEORY at difficulty 1) and even admitted "difficulty
and workload are not always the same thing" before answering anyway. It only
considered 4 of the 10 professors, and the top-ranked retrieved chunk was
actually Findlay's EC338 at **difficulty 4**,  not a light class at all.

**Root cause (tied to a specific pipeline stage):** This is a **retrieval**
failure caused by the mismatch between a superlative/aggregation question and
top-k similarity search. "Lightest workload" requires *comparing all 10
professors*, but retrieval only returns the 5 chunks whose embeddings are
nearest to the phrase "lightest workload" — it never sees the other professors,
so the model can't actually rank the whole field. Worse, the reviews rarely use
the word "workload" explicitly, so the embedding model matches on the closest
available signal (difficulty language and ratings) and the LLM is forced to use
difficulty as a proxy for workload. The result is an answer drawn from whichever
5 chunks happened to rank highest, not from a real survey of who has the
lightest load. This is the "cross-professor" limitation I flagged in
planning.md showing up in practice.

**What you would change to fix it:** Two options. (1) For aggregation questions,
retrieve *per professor* — run the query with a metadata filter for each
professor and take the top chunk from each, so all 10 are represented before the
LLM compares them. (2) Store the numeric difficulty/quality ratings as
structured metadata and sort/aggregate on them directly instead of hoping the
embedding picks them up. Simply raising top-k would help a little but wouldn't
fix the root problem, since the corpus is 93 chunks and 5–10 still won't
guarantee coverage of every professor.

---

## Spec Reflection

**One way the spec helped you during implementation:** The chunking section and
architecture diagram in `planning.md` turned a vague "build a RAG system" into
five concrete, testable stages (ingest → chunk → embed → retrieve → generate).
Because I'd written down "2–4 reviews per chunk" and "answer from retrieved
context only" in advance, I knew exactly what each script had to do and what
"correct" looked like before writing any code — and the five evaluation
questions gave me a ready-made test set to check the finished pipeline against.

**One way your implementation diverged from the spec, and why:** Two divergences.
First, `planning.md` originally specified **300-token chunks**, but `all-MiniLM-
L6-v2` has a 256-token maximum sequence length, so a 300-token chunk would have
its last ~44 tokens silently truncated at embedding time and never represented
in the vector. I lowered the chunk size to **256** so every token actually gets
embedded, and updated planning.md with the reason. Second, the plan assumed
scraping RMP with `requests` + BeautifulSoup; that doesn't work because the site
is JavaScript-rendered, so I pulled the reviews through RMP's GraphQL API and
saved them as `.txt` files instead — same data, different ingestion path.

---

## AI Usage

**Instance 1 — ingestion and chunking**

- *What I gave the AI:* My Documents table and Chunking Strategy section from
  planning.md (300 tokens / 50 overlap, "2–4 reviews per chunk"), plus the
  problem that scraping RMP returned empty pages.
- *What it produced:* It diagnosed that RMP is JavaScript-rendered and pulled
  the reviews via the GraphQL API instead, then wrote a chunker.
- *What I changed or overrode:* I directed it to pack **whole reviews** together
  up to the token limit rather than splitting mechanically every N tokens (so no
  chunk ends mid-sentence), and to **prepend the professor's name** to every
  chunk to prevent cross-professor misattribution. I also had it switch the
  chunk size from 300 to **256** once we confirmed MiniLM truncates at 256.

**Instance 2 — grounded generation**

- *What I gave the AI:* My grounding requirement (answer from retrieved context
  only, cite sources) and the request to wire retrieval to Groq's
  `llama-3.3-70b-versatile`.
- *What it produced:* A `query.py` with a system prompt and an `ask()` function,
  and a Gradio `app.py`.
- *What I changed or overrode:* I insisted that **source attribution be
  programmatic** — built from the files of the chunks actually retrieved — rather
  than trusting the model to cite them, since an LLM can hallucinate or omit
  citations. I also pinned the exact decline sentence ("I don't have enough
  information on that.") and set `temperature=0` so answers stay factual and
  repeatable, and made the UI hide sources when the model declines.
