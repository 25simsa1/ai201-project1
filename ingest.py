"""
Milestone 3 - document pipeline.

Loads the professor review files from data/, cleans them up, and splits
them into chunks for the embedding step (Milestone 4).
Chunk size / overlap come from planning.md (300 tokens, 50 overlap).
"""

import os
import re
import html
import json
import random

from transformers import AutoTokenizer

DATA_DIR = "data"
CHUNK_SIZE = 256       # tokens - set to MiniLM's max seq length (see planning.md)
CHUNK_OVERLAP = 50     # tokens, from planning.md

# I count tokens with MiniLM's own tokenizer since that's the model that will
# actually embed these chunks in Milestone 4, so "300 tokens" means the same
# thing here as it will there.
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


def count_tokens(text):
    return len(tokenizer.encode(text, add_special_tokens=False))


def load_documents():
    """Read every .txt file in data/ and return a list of documents.

    The first line of each file is "Professor: <name>", so I grab that for
    metadata and use the rest of the file as the actual review text.
    """
    docs = []
    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.endswith(".txt"):
            continue

        path = os.path.join(DATA_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()

        # pull the professor name off the header line if it's there
        professor = fname.replace(".txt", "")
        first_line = raw.split("\n", 1)[0]
        if first_line.startswith("Professor:"):
            professor = first_line.replace("Professor:", "").strip()

        docs.append({
            "source": fname,
            "professor": professor,
            "text": raw,
        })

    return docs


def clean_text(text):
    """Clean a raw document.

    The reviews came from Rate My Professors so some of them still have HTML
    entities like &#39; or &quot; in them. I unescape those and tidy up the
    whitespace, but I keep the review text, the professor name and the
    Class/Quality/Difficulty lines because those are real content.
    """
    # turn &#39; -> ' , &quot; -> " , &amp; -> & etc.
    text = html.unescape(text)

    # drop any stray html tags just in case some slipped through
    text = re.sub(r"<[^>]+>", "", text)

    # normalize whitespace: collapse 3+ blank lines down to one blank line,
    # and strip trailing spaces on each line
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def chunk_document(doc):
    """Split one cleaned document into ~300 token chunks with ~50 token overlap.

    The reviews are separated by blank lines, so instead of cutting blindly
    every 300 tokens I pack whole reviews together until I'd go over the limit.
    That keeps each review intact (planning.md says 2-4 reviews per chunk) and
    stops a chunk from ending in the middle of a sentence. I also stick the
    professor's name at the top of every chunk so a retrieved chunk can stand
    on its own and doesn't get confused with another professor.
    """
    professor = doc["professor"]
    prefix = f"Professor {professor}:\n"

    # the header (Professor/Department/Source) is the first block - drop it and
    # split the rest into review blocks on the blank lines
    blocks = [b.strip() for b in doc["text"].split("\n\n") if b.strip()]
    review_blocks = [b for b in blocks if not b.startswith("Professor:")]

    chunks = []
    current = []
    current_tokens = 0

    def flush():
        if not current:
            return
        text = prefix + "\n\n".join(current)
        chunks.append({
            "text": text,
            "source": doc["source"],
            "professor": professor,
            "n_tokens": count_tokens(text),
        })

    for block in review_blocks:
        bt = count_tokens(block)

        # if adding this review would blow past the chunk size, close the
        # current chunk first
        if current and current_tokens + bt > CHUNK_SIZE:
            flush()

            # carry the last review(s) over as overlap (up to ~50 tokens)
            overlap = []
            t = 0
            for b in reversed(current):
                overlap.insert(0, b)
                t += count_tokens(b)
                if t >= CHUNK_OVERLAP:
                    break
            current = overlap
            current_tokens = sum(count_tokens(b) for b in current)

        current.append(block)
        current_tokens += bt

    flush()
    return chunks


def build_chunks():
    """Run the whole pipeline: load -> clean -> chunk. Returns all chunks."""
    docs = load_documents()
    for d in docs:
        d["text"] = clean_text(d["text"])

    all_chunks = []
    for d in docs:
        all_chunks.extend(chunk_document(d))

    # tag each chunk with an id now that we have the full list
    for i, c in enumerate(all_chunks):
        c["chunk_id"] = i

    return all_chunks


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents from {DATA_DIR}/")

    chunks = build_chunks()
    print(f"Produced {len(chunks)} chunks total "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})\n")

    # sanity check the count against the milestone's 50-2000 guideline
    if len(chunks) < 50:
        print("WARNING: fewer than 50 chunks - chunks might be too big")
    elif len(chunks) > 2000:
        print("WARNING: more than 2000 chunks - chunks might be too small")

    sizes = [c["n_tokens"] for c in chunks]
    print(f"token counts: min={min(sizes)}, max={max(sizes)}, "
          f"avg={sum(sizes) // len(sizes)}\n")

    # print 5 random chunks and inspect them (the checkpoint step)
    random.seed(0)
    for c in random.sample(chunks, 5):
        print(f"----- chunk {c['chunk_id']} | {c['source']} | "
              f"{c['n_tokens']} tokens -----")
        print(c["text"])
        print()

    # save the chunks so Milestone 4 can embed them without re-running this
    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print("Saved chunks to chunks.json")
