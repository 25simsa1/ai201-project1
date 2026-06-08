"""
Milestone 5 - grounded generation.

Ties retrieval (embed.search) to the LLM. The whole point here is grounding:
the model is only allowed to answer from the chunks we retrieved, and if those
chunks don't cover the question it has to say so instead of making something up.
ask() returns the answer plus the list of source files the chunks came from.
"""

import os

from dotenv import load_dotenv
from groq import Groq

from embed import search, TOP_K

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
LLM_MODEL = "llama-3.3-70b-versatile"   # from planning.md

# exact sentence the model must use when the context doesn't cover the question
NO_ANSWER = "I don't have enough information on that."

SYSTEM_PROMPT = f"""You are a Q&A assistant for student reviews of Colby College professors.

Follow these rules exactly:
- Answer using ONLY the information in the context documents the user provides.
- Do NOT use any outside or prior knowledge about these professors or Colby.
- If the context does not contain enough information to answer the question,
  reply with exactly this sentence and nothing else: "{NO_ANSWER}"
- When you do answer, name the professor(s) the information is about and base
  every statement on the reviews in the context.
- Do not invent professors, classes, ratings, or opinions that are not in the
  context."""


def build_context(chunks):
    """Format the retrieved chunks into a numbered context block for the prompt."""
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(f"[Document {i} - source: {c['source']}]\n{c['text']}")
    return "\n\n".join(blocks)


def ask(question):
    """Retrieve, generate a grounded answer, and return it with its sources."""
    chunks = search(question, k=TOP_K)
    context = build_context(chunks)

    user_message = (
        f"Context documents:\n\n{context}\n\n"
        f"Question: {question}"
    )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,   # keep it factual / repeatable, don't get creative
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    answer = response.choices[0].message.content.strip()

    # Source attribution is done in code, not left to the model: list the unique
    # files the retrieved chunks actually came from. If the model declined to
    # answer, don't show sources it didn't really use.
    if answer == NO_ANSWER:
        sources = []
    else:
        sources = []
        for c in chunks:
            if c["source"] not in sources:
                sources.append(c["source"])

    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    # test grounded generation end to end (Milestone 5 checkpoint)
    tests = [
        "Which Colby professors are known for being engaging in lectures?",
        "What do students say about Professor Findlay's grading?",
        "Which professors have a light workload?",
        "What is the best dining hall at Colby?",   # not in our documents
    ]
    for q in tests:
        result = ask(q)
        print(f"Q: {q}")
        print(f"A: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print("-" * 70)
