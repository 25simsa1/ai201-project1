"""
Milestone 3 - document pipeline.

Loads the professor review files from data/, cleans them up, and splits
them into chunks for the embedding step (Milestone 4).
Chunk size / overlap come from planning.md (300 tokens, 50 overlap).
"""

import os
import re
import html

DATA_DIR = "data"


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


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents from {DATA_DIR}/\n")

    # clean them all
    for d in docs:
        d["text"] = clean_text(d["text"])

    # print one cleaned document and read it (per the milestone instructions)
    sample = docs[0]
    print(f"--- {sample['source']} (professor: {sample['professor']}) ---")
    print(sample["text"][:800])
