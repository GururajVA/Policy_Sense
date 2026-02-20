import os
from llama_cpp import Llama, LlamaGrammar
import json
from typing import List, Dict, Any
import re

MODEL_PATH = os.getenv("MODEL_PATH", "model/llama-3.2-1b-instruct-q4_k_m.gguf")
_llm_instance: Llama | None = None
_json_grammar: LlamaGrammar | None = None


def _get_llm() -> Llama:
    global _llm_instance
    if _llm_instance is None:
        num_threads = max(1, int(os.getenv("LLM_THREADS", os.cpu_count() or 8)))
        batch_size = max(32, int(os.getenv("LLM_BATCH", "256")))
        _llm_instance = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,
            n_threads=num_threads,
            n_batch=batch_size,
            f16_kv=True,
            use_mmap=True,
            use_mlock=False,
            verbose=False,
        )
    return _llm_instance


def _get_json_grammar() -> LlamaGrammar | None:
    """Constrain outputs to our specific JSON schema."""
    global _json_grammar
    if _json_grammar is None:
        try:
            grammar_str = r'''
root ::= json

json ::= "{" ws "\"decision\"" ws ":" ws decision ws "," ws "\"amount\"" ws ":" ws integer ws "," ws "\"justification\"" ws ":" ws string ws "}"

decision ::= "\"approved\"" | "\"rejected\"" | "\"unknown\""

integer ::= digit+

string ::= '"' char* '"'
char ::= [^"\\] | '\\' ["\\/bfnrt]

ws ::= [ \t\n\r]*
digit ::= [0-9]
'''
            _json_grammar = LlamaGrammar.from_string(grammar_str)
        except Exception as e:
            # If grammar creation fails, return None to disable grammar constraint
            _json_grammar = None
    return _json_grammar


def make_decision(query: str, parsed_query: Dict[str, Any], retrieved_clauses: List[Dict[str, Any]]) -> str:
    if not retrieved_clauses:
        # Minimal JSON when nothing retrieved
        return json.dumps({
            "decision": "unknown",
            "amount": 0,
            "justification": "No relevant clauses found."
        })

    context = "\n".join([f"Clause: {r['chunk']}" for r in retrieved_clauses])
    prompt = f"""
You are an insurance policy expert. Using the following user query and policy clauses, decide if the procedure is covered, the payout amount, and provide justification referencing the clauses.

User Query: {query}
Parsed Query: {json.dumps(parsed_query)}
Policy Clauses:
{context}

Return ONLY a compact JSON object with keys: decision (approved/rejected/unknown), amount (number), justification (reference clause numbers or short rationale).
""".strip()

    llm = _get_llm()
    grammar = _get_json_grammar()
    if grammar is not None:
        output = llm(prompt, max_tokens=128, temperature=0.1, grammar=grammar)
    else:
        output = llm(prompt, max_tokens=128, temperature=0.1, stop=["\n\n", "</s>"])
    text = output['choices'][0]['text'].strip()

    # Must be valid JSON by grammar; still verify just in case
    try:
        return json.dumps(json.loads(text))
    except Exception:
        return json.dumps({
            "decision": "unknown",
            "amount": 0,
            "justification": "Model returned invalid JSON despite grammar."
        })


def initialize_llm() -> bool:
    """Load the LLM into memory at startup for lower first-token latency."""
    try:
        _ = _get_llm()
        return True
    except Exception:
        return False