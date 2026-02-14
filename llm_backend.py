# llm_backend.py
# Factory module for LLM and embedding backends.
# Supports Ollama and LM Studio (OpenAI-compatible).

import logging

import config

def get_embeddings():
    """Return an embeddings instance based on the configured backend."""
    backend = getattr(config, "BACKEND", "ollama")

    if backend == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            base_url=config.OLLAMA_BASE_URL,
            model=config.EMBEDDING_MODEL,
        )
    elif backend == "lm_studio":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            base_url=getattr(config, "LM_STUDIO_BASE_URL", "http://localhost:1234/v1"),
            model=config.EMBEDDING_MODEL,
            # LM Studio doesn't require an API key, but the client expects one
            api_key="lm-studio",
        )
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'ollama' or 'lm_studio'.")


def get_llm(**kwargs):
    """Return a chat LLM instance based on the configured backend.
    
    Additional kwargs are passed through to the LLM constructor
    (e.g. temperature, streaming).
    """
    backend = getattr(config, "BACKEND", "ollama")

    if backend == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            base_url=config.OLLAMA_BASE_URL,
            model=config.LLM_MODEL,
            **kwargs,
        )
    elif backend == "lm_studio":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            base_url=getattr(config, "LM_STUDIO_BASE_URL", "http://localhost:1234/v1"),
            model=config.LLM_MODEL,
            # LM Studio doesn't require an API key, but the client expects one
            api_key="lm-studio",
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'ollama' or 'lm_studio'.")
