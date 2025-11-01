import os
from typing import Optional

import streamlit as st
from openai import OpenAI

# Read key/model from Streamlit secrets first, then env as fallback.
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
DEFAULT_MODEL = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")

_client: Optional[OpenAI] = None


def client() -> OpenAI:
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it in Streamlit Secrets or environment."
            )
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def answer_with_openai(
    context: str,
    question: str,
    model: Optional[str] = None,
    temperature: float = 0.2,
) -> str:
    """
    Compose a grounded answer using ONLY the provided context.
    - Returns concise bullets; asks user to add docs if context is empty.
    """
    if not context.strip():
        return (
            "I don't have enough information in the knowledge base to answer this. "
            "Please upload relevant docs and rebuild the index."
        )

    system = (
        "You are a senior performance engineer. "
        "Answer ONLY using the provided context. "
        "Be concise (<= 6 bullets). "
        "Use inline citations like [filename.ext] where helpful. "
        "If the answer is not in the context, say so."
    )
    user = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer with bullets and short rationale."

    mdl = model or DEFAULT_MODEL
    resp = client().chat.completions.create(
        model=mdl,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content.strip()
