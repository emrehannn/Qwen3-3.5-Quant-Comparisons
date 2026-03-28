"""
NeedleBench: Long-context Retrieval and Reasoning Benchmark

Based on NeedleBench: Can LLMs Do Retrieval and Reasoning in 1 Million Context Window?"
Paper: arXiv:2407.11963

Implements the official NeedleBench evaluation tasks:
    - S-RT (Single Retrieval): One needle inserted at specified depth inside
      a long haystack built from en_haystack_texts; asks needle's retrieval_question
    - M-RT (Multi Retrieval): Multiple needles spread from a start depth inside
      a long haystack; asks each needle's retrieval_question
    - M-RS (Multi-fact Reasoning): Sample derivations inserted as needles
      into a long haystack; asks the reasoning question

Metrics — ALL values stored as fractions in [0, 1]. Multiply by 100 for display.
    - accuracy for S-RT and M-RS  (correct / total)
    - precision / recall / f1 for M-RT
    - composite: O = 0.4*S-RT + 0.3*M-RT(f1) + 0.3*M-RS

Scoring design for S-RT and M-RS:
    composite_retrieval_score = max(levenshtein_soft_score, predicted_coverage_score, substr_score)

    levenshtein_soft_score: character-level edit distance similarity.
    predicted_coverage_score: fraction of predicted tokens that appear in the reference,
        active only when the prediction contains >= 3 tokens. This correctly handles
        models that return concise extractions ('The Galaxy Melody Band') rather than
        echoing the full sentence template, without rewarding trivially short outputs.
    substr_score: safe exact-phrase match to fix single-word false negatives.
        Both scores are stored per-trial for full transparency.

    M-RT uses token-set F1, which is already format-agnostic — no change needed.

Depth variation:
    Every task is evaluated at multiple depth percentages (default 5%-90%).
    Haystack content is held constant across depth conditions per sample (rng reseeded
    per depth loop), so observed accuracy differences isolate needle position effects
    rather than haystack variation.
    Results carry a `depth_breakdown` dict so architecture-level depth
    degradation is directly visible in the output JSON and downstream plots.

DATA REQUIREMENT: This benchmark requires the official OpenCompass NeedleBench dataset.
Dataset downloaded automatically from HuggingFace:
    https://huggingface.co/datasets/opencompass/NeedleBench


Reference:
    Li et al. "Can LLMs Do Retrieval and Reasoning in Information-Dense Context?"
    arXiv:2407.11963 (2024)
"""

import json
import re
import Levenshtein
from pathlib import Path
from tqdm import tqdm
from llama_cpp import Llama
from datasets import load_dataset
import random


TASK_TYPES     = ["S-RT", "M-RT", "M-RS"]
DEFAULT_DEPTHS = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9]

# Minimum number of predicted tokens required for predicted_coverage_score to
# activate. Guards against trivially short outputs (e.g. 'the moon') that would
# score 1.0 precision against almost any long reference sentence.
_COVERAGE_MIN_TOKENS = 3


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_needlebench_subset(subset, split="test", language_filter=None):
    """Load one NeedleBench subset from HuggingFace."""
    print(f"Loading NeedleBench subset: {subset}")
    print(f"  Source: https://huggingface.co/datasets/opencompass/NeedleBench")
    try:
        dataset = load_dataset("opencompass/NeedleBench", subset, split=split)
    except Exception as e:
        raise RuntimeError(
            f"FAILED TO LOAD '{subset}'.\nError: {e}\n\n"
            f"pip install datasets huggingface_hub\n"
            f"https://huggingface.co/datasets/opencompass/NeedleBench\n"
            f"This benchmark WILL NOT RUN with synthetic/fake data."
        )
    if language_filter and "language" in dataset.column_names:
        dataset = dataset.filter(lambda x: x.get("language", "") == language_filter)
        print(f"  Filtered to '{language_filter}': {len(dataset)} samples")
    if len(dataset) == 0:
        raise RuntimeError(f"Subset '{subset}' is empty after filtering.")
    print(f"  Loaded {len(dataset)} samples")
    return dataset


# ---------------------------------------------------------------------------
# Scoring & Extraction
# ---------------------------------------------------------------------------

def extract_structured_answer(text: str) -> str:
    """
    Strips internal reasoning and extracts the final answer from <answer> tags.
    Falls back to the raw cleaned text if the model failed to use tags.
    """
    # 1. Strip <think> tags completely
    clean_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    # 2. Extract text inside <answer>...</answer>
    match = re.search(r'<answer>(.*?)</answer>', clean_text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback if tags are missing
    return clean_text


def levenshtein_soft_score(predicted: str, reference: str) -> float:
    """
    Character-level edit distance similarity.
    score = 1 - d(P, R) / max(|P|, |R|)
    Returns 1.0 for exact match, 0.0 for completely different. In [0, 1].
    """
    if not predicted or not reference:
        return 0.0
    pred = predicted.lower().strip()
    ref  = reference.lower().strip()
    if pred == ref:
        return 1.0
    distance = Levenshtein.distance(pred, ref)
    max_len  = max(len(pred), len(ref))
    return max(0.0, 1.0 - distance / max_len) if max_len > 0 else 0.0


def predicted_coverage_score(predicted: str, reference: str) -> float:
    """
    Token precision of prediction against reference: how many of the predicted
    tokens appear in the reference token set.

    Returns 0.0 if the prediction has fewer than _COVERAGE_MIN_TOKENS tokens,
    to prevent trivially short outputs from scoring artificially high.

    Example:
        predicted = "The Galaxy Melody Band"          (4 tokens)
        reference = "The first band to perform on the Moon was the Galaxy Melody Band."
        coverage  = 4/4 = 1.0   (all predicted tokens are in the reference)

    This correctly rewards a concise extraction that contains the right content,
    while the _COVERAGE_MIN_TOKENS guard rejects outputs like "the moon" (2 tokens)
    that would otherwise trivially score 1.0 against a long reference.

    All returned values in [0, 1].
    """
    if not predicted or not reference:
        return 0.0
    pred_tokens = set(predicted.lower().strip().split())
    ref_tokens  = set(reference.lower().strip().split())
    if len(pred_tokens) < _COVERAGE_MIN_TOKENS:
        return 0.0
    if not ref_tokens:
        return 0.0
    tp = len(pred_tokens & ref_tokens)
    return tp / len(pred_tokens)


def is_negation_or_refusal(text: str) -> bool:
    """Checks if the model generated a 'not found' or refusal response."""
    text = text.lower().strip()
    if text == "not found":
        return True
    
    patterns = [
        "not mentioned", "not stated", "not included",
        "doesn't say", "does not say", "no information",
        "not specify", "does not specify", "cannot answer",
        "no mention", "does not contain"
    ]
    # Only penalize if the text is relatively short (< 20 words).
    # This avoids killing a valid, long comparative sentence that 
    # just happens to contain the phrase "not mentioned".
    if len(text.split()) < 20 and any(p in text for p in patterns):
        return True
    return False


def _is_exact_phrase_match(short_text: str, long_text: str) -> bool:
    """
    Checks if short_text is a contiguous phrase safely inside long_text.
    Includes special handling for Chinese/Japanese/Korean (CJK) where word
    boundaries (\b) are not appropriate for concatenated characters.
    """
    if len(short_text) < 3:
        return False

    # If the text is entirely ASCII, or ends with ASCII word chars, use \b.
    # Otherwise (CJK), simple substring containment is safer because CJK
    # does not use spaces between words.
    is_ascii = bool(re.match(r'^[\x00-\x7F]*$', short_text))
    
    if is_ascii:
        pattern = r'\b' + re.escape(short_text) + r'\b'
        return bool(re.search(pattern, long_text))
    else:
        return short_text in long_text


def composite_retrieval_score(predicted: str, reference: str) -> float:
    """
    Final score for S-RT and M-RS trials.
    Includes a guard against false positive refusals and a
    safe exact-phrase match to fix single-word false negatives.
    """
    if not predicted or not reference:
        return 0.0

    pred_clean = predicted.lower().strip()
    ref_clean  = reference.lower().strip()

    # 1. Guard against False Positives (Verbose Refusals)
    if is_negation_or_refusal(pred_clean) and not is_negation_or_refusal(ref_clean):
        return 0.0

    # 2. Existing metrics
    lev = levenshtein_soft_score(predicted, reference)
    cov = predicted_coverage_score(predicted, reference)

    # 3. Guard against False Negatives (Single-word vs Full sentence)
    pred_nopunct = re.sub(r'[^\w\s]', '', pred_clean).strip()
    ref_nopunct  = re.sub(r'[^\w\s]', '', ref_clean).strip()

    substr_score = 0.0
    if _is_exact_phrase_match(pred_nopunct, ref_nopunct) or \
       _is_exact_phrase_match(ref_nopunct, pred_nopunct):
        substr_score = 1.0

    return max(lev, cov, substr_score)


def calculate_precision_recall_f1(predicted: str, reference: str) -> tuple:
    """Token-set P/R/F1. All returned values in [0, 1]."""
    if not predicted or not reference:
        return 0.0, 0.0, 0.0

    pred_clean = predicted.lower().strip()
    ref_clean  = reference.lower().strip()

    # 1. Guard against False Positives (Verbose Refusals)
    if is_negation_or_refusal(pred_clean) and not is_negation_or_refusal(ref_clean):
        return 0.0, 0.0, 0.0

    pred_tokens = set(pred_clean.split())
    ref_tokens  = set(ref_clean.split())
    
    if not pred_tokens or not ref_tokens:
        return 0.0, 0.0, 0.0
        
    tp        = len(pred_tokens & ref_tokens)
    precision = tp / len(pred_tokens)
    recall    = tp / len(ref_tokens)
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def _tokenize(llm, text):
    return llm.tokenize(text.encode("utf-8"))

def _detokenize(llm, tokens):
    return llm.detokenize(tokens).decode("utf-8", errors="ignore")


# ---------------------------------------------------------------------------
# Haystack construction
# ---------------------------------------------------------------------------

def _build_haystack_tokens(llm, haystack_texts, target_tokens, rng):
    shuffled = list(haystack_texts)
    rng.shuffle(shuffled)
    tokens = []
    for text in shuffled:
        if len(tokens) >= target_tokens:
            break
        tokens.extend(_tokenize(llm, text))
    return tokens[:target_tokens]


def build_haystack_with_needle(llm, haystack_texts, needle, target_tokens, depth, rng):
    """Insert one needle at `depth` fraction into a haystack of ~target_tokens."""
    needle_tokens   = _tokenize(llm, needle)
    available       = max(0, target_tokens - len(needle_tokens))
    haystack_tokens = _build_haystack_tokens(llm, haystack_texts, available, rng)
    insert_pos      = max(0, min(int(len(haystack_tokens) * depth), len(haystack_tokens)))
    full_tokens     = haystack_tokens[:insert_pos] + needle_tokens + haystack_tokens[insert_pos:]
    return _detokenize(llm, full_tokens)


def build_haystack_with_multiple_needles(llm, haystack_texts, needles, target_tokens, start_depth, rng):
    """
    Insert multiple needles into a haystack.
    First needle at start_depth; rest spread evenly up to 0.9.
    Insertion positions are computed against the pre-insertion haystack length
    to avoid compounding index shifts.
    """
    n = len(needles)
    if n == 0:
        return _detokenize(llm, _build_haystack_tokens(llm, haystack_texts, target_tokens, rng))
    if n == 1:
        depths = [start_depth]
    else:
        end_depth = min(0.9, max(start_depth + 0.1, 0.9))
        depths    = [start_depth + (end_depth - start_depth) * i / (n - 1) for i in range(n)]

    needle_token_lists = [_tokenize(llm, nd) for nd in needles]
    available          = max(0, target_tokens - sum(len(t) for t in needle_token_lists))
    haystack_tokens    = _build_haystack_tokens(llm, haystack_texts, available, rng)

    hs_len      = len(haystack_tokens)
    insert_data = [(max(0, min(int(hs_len * d), hs_len)), t)
                   for d, t in zip(depths, needle_token_lists)]
    insert_data.sort(key=lambda x: x[0], reverse=True)   # reverse so indices stay valid

    result = list(haystack_tokens)
    for pos, ntoks in insert_data:
        result = result[:pos] + list(ntoks) + result[pos:]
    return _detokenize(llm, result)


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_PROMPT = (
    "Below is a long document. Read it carefully and answer the question.\n"
    "Document:\n{context}\n\n"
    "Question: {question}\n"
    "Instructions: You must wrap your final, concise answer inside <answer> and </answer> tags. "
    "If the document does not contain the answer, reply with exactly <answer>NOT FOUND</answer>.\n"
    "Response:"
)


# ---------------------------------------------------------------------------
# Task evaluators
# ---------------------------------------------------------------------------

def evaluate_single_retrieval(llm, needle_dataset, haystack_texts,
                               ctx_len, depth_percentages, num_samples):
    """
    S-RT: embed one needle at each depth, ask its retrieval_question.

    Scoring: composite_retrieval_score (max of levenshtein and predicted_coverage).
    correct = True if composite score >= 0.5.
    Both the composite score and the raw levenshtein score are stored per trial
    for full transparency.

    accuracy = correct / total, in [0, 1].
    """
    RESERVE = 300
    indices = list(range(len(needle_dataset)))
    random.seed(42); random.shuffle(indices)
    if num_samples is not None:
        indices = indices[:num_samples]

    results_by_depth = {d: [] for d in depth_percentages}
    all_results      = []
    target_tokens    = ctx_len - RESERVE

    for depth in depth_percentages:
        # rng is reseeded per depth so haystack content is identical across depths.
        # Only the needle insertion position changes, isolating the depth variable.
        rng = random.Random(42)
        for idx in tqdm(indices, desc=f"S-RT @{ctx_len//1024}k depth={int(depth*100)}%"):
            sample   = needle_dataset[idx]
            needle   = sample.get("needle", "")
            question = sample.get("retrieval_question", "")
            expected = sample.get("gold_standard_answer", "")
            if not needle or not question or not expected:
                continue

            context = build_haystack_with_needle(
                llm, haystack_texts, needle, target_tokens, depth, rng)
            output  = llm.create_completion(
                prompt=_PROMPT.format(context=context, question=question),
                max_tokens=100, temperature=0.0,
                stop=["Question:", "Document:", "<|im_end|>"])

            raw_generated = output["choices"][0]["text"].strip()
            generated     = extract_structured_answer(raw_generated)

            lev_score  = levenshtein_soft_score(generated, expected)    # [0,1]
            score      = composite_retrieval_score(generated, expected)  # [0,1]
            is_correct = score >= 0.5

            r = {"depth_percent": int(depth * 100), "question": question,
                 "expected": expected, "predicted": generated,
                 "score": score, "levenshtein_score": lev_score,
                 "correct": is_correct}
            results_by_depth[depth].append(r)
            all_results.append(r)

    total    = len(all_results)
    correct  = sum(1 for r in all_results if r["correct"])
    accuracy = correct / total if total > 0 else 0.0   # [0,1]

    depth_breakdown = {}
    for d, dr in results_by_depth.items():
        dt = len(dr); dc = sum(1 for r in dr if r["correct"])
        depth_breakdown[f"{int(d * 100)}%"] = {
            "accuracy":       dc / dt if dt > 0 else 0.0,                          # [0,1]
            "avg_score":      sum(r["score"] for r in dr) / dt if dt > 0 else 0.0, # [0,1]
            "avg_lev_score":  sum(r["levenshtein_score"] for r in dr) / dt if dt > 0 else 0.0,
            "correct": dc, "total": dt,
        }

    return {"task": "S-RT", "task_description": "Single-Needle Retrieval",
            "context_length": ctx_len, "accuracy": accuracy,
            "correct": correct, "total": total,
            "depth_breakdown": depth_breakdown, "trials": all_results}


def evaluate_multi_retrieval(llm, needle_dataset, haystack_texts,
                              ctx_len, depth_percentages, num_samples,
                              needles_per_sample=5):
    """
    M-RT: embed needles_per_sample needles starting at each depth, ask each question.

    Scoring: composite_retrieval_score per needle (same as S-RT and M-RS).
    This eliminates the format-sensitivity bias of token-set F1, where a
    concise-but-correct answer like "Voyager of the Stars" would be penalized
    for missing the preamble words of the full reference sentence.

    correct = True if composite score >= 0.5 (consistent with S-RT/M-RS).
    accuracy = correct / total, in [0, 1].
    Legacy F1 is still stored per-needle for reference.
    """
    RESERVE = 400
    indices = list(range(len(needle_dataset)))
    random.seed(42); random.shuffle(indices)

    groups = [indices[i:i + needles_per_sample]
              for i in range(0, len(indices) - needles_per_sample + 1, needles_per_sample)]
    if num_samples is not None:
        groups = groups[:num_samples]

    results_by_depth = {d: [] for d in depth_percentages}
    all_results      = []
    target_tokens    = ctx_len - RESERVE

    for depth in depth_percentages:
        rng = random.Random(42)
        for group in tqdm(groups, desc=f"M-RT @{ctx_len//1024}k depth={int(depth*100)}%"):
            samples   = [needle_dataset[i] for i in group]
            needles   = [s.get("needle", "")               for s in samples]
            questions = [s.get("retrieval_question", "")   for s in samples]
            expecteds = [s.get("gold_standard_answer", "") for s in samples]
            if not all(needles) or not all(questions) or not all(expecteds):
                continue

            context = build_haystack_with_multiple_needles(
                llm, haystack_texts, needles, target_tokens, depth, rng)

            scores     = []
            corrects   = []
            needle_results = []
            for question, expected in zip(questions, expecteds):
                output    = llm.create_completion(
                    prompt=_PROMPT.format(context=context, question=question),
                    max_tokens=100, temperature=0.0,
                    stop=["Question:", "Document:", "<|im_end|>"])

                raw_generated = output["choices"][0]["text"].strip()
                generated     = extract_structured_answer(raw_generated)

                score      = composite_retrieval_score(generated, expected)  # [0,1]
                is_correct = score >= 0.5
                # Legacy F1 kept for reference / backward analysis
                _, _, f1_legacy = calculate_precision_recall_f1(generated, expected)

                scores.append(score)
                corrects.append(is_correct)
                needle_results.append({"question": question, "expected": expected,
                                       "predicted": generated,
                                       "score": score, "correct": is_correct,
                                       "legacy_f1": f1_legacy})

            nq    = len(needle_results)
            nc    = sum(1 for c in corrects if c)
            trial = {"depth_percent": int(depth * 100), "needle_count": nq,
                     "avg_score": sum(scores) / nq,
                     "accuracy":  nc / nq,
                     "correct": nc, "total": nq,
                     "needle_results": needle_results}
            results_by_depth[depth].append(trial)
            all_results.append(trial)

    total = len(all_results)
    if total == 0:
        return {"task": "M-RT", "task_description": "Multi-Needle Retrieval",
                "context_length": ctx_len, "accuracy": 0.0, "avg_score": 0.0,
                "correct": 0, "total": 0, "depth_breakdown": {}, "trials": []}

    all_correct = sum(r["correct"] for r in all_results)
    all_needles = sum(r["total"]   for r in all_results)
    accuracy    = all_correct / all_needles if all_needles > 0 else 0.0  # [0,1]

    depth_breakdown = {}
    for d, dr in results_by_depth.items():
        dt = len(dr)
        if dt == 0:
            continue
        d_correct = sum(r["correct"] for r in dr)
        d_total   = sum(r["total"]   for r in dr)
        depth_breakdown[f"{int(d * 100)}%"] = {
            "accuracy":  d_correct / d_total if d_total > 0 else 0.0,            # [0,1]
            "avg_score": sum(r["avg_score"] for r in dr) / dt if dt > 0 else 0.0, # [0,1]
            "correct": d_correct, "total": d_total,
        }

    return {"task": "M-RT", "task_description": "Multi-Needle Retrieval",
            "context_length": ctx_len, "accuracy": accuracy,
            "correct": all_correct, "total": all_needles,
            "depth_breakdown": depth_breakdown, "trials": all_results}


def evaluate_multi_reasoning(llm, reasoning_dataset, haystack_texts,
                              ctx_len, depth_percentages, num_samples):
    """
    M-RS: embed derivations as needles starting at each depth, ask reasoning question.

    Scoring: composite_retrieval_score (same as S-RT), applied for consistency.
    For single-word answers the exact-match path in levenshtein_soft_score fires
    first, so the composite score makes no difference in those cases.
    Both the composite score and the raw levenshtein score are stored per trial.

    accuracy = correct / total, in [0, 1].
    """
    RESERVE = 400
    indices = list(range(len(reasoning_dataset)))
    random.seed(42); random.shuffle(indices)
    if num_samples is not None:
        indices = indices[:num_samples]

    results_by_depth = {d: [] for d in depth_percentages}
    all_results      = []
    target_tokens    = ctx_len - RESERVE

    for depth in depth_percentages:
        rng = random.Random(42)
        for idx in tqdm(indices, desc=f"M-RS @{ctx_len//1024}k depth={int(depth*100)}%"):
            sample      = reasoning_dataset[idx]
            question    = sample.get("question", "")
            answer      = sample.get("answer", "")
            derivations = sample.get("derivations", [])
            if not question or not answer:
                continue
            if isinstance(derivations, str):
                try:    derivations = json.loads(derivations)
                except: derivations = [derivations]
            if not isinstance(derivations, list) or len(derivations) == 0:
                derivations = [answer]

            context   = build_haystack_with_multiple_needles(
                llm, haystack_texts, derivations, target_tokens, depth, rng)
            output    = llm.create_completion(
                prompt=_PROMPT.format(context=context, question=question),
                max_tokens=150, temperature=0.0,
                stop=["Question:", "Document:", "<|im_end|>"])

            raw_generated = output["choices"][0]["text"].strip()
            generated     = extract_structured_answer(raw_generated)

            lev_score  = levenshtein_soft_score(generated, answer)    # [0,1]
            score      = composite_retrieval_score(generated, answer)  # [0,1]
            is_correct = score >= 0.5

            r = {"depth_percent": int(depth * 100), "question": question,
                 "expected": answer, "predicted": generated,
                 "score": score, "levenshtein_score": lev_score,
                 "correct": is_correct,
                 "derivation_count": len(derivations)}
            results_by_depth[depth].append(r)
            all_results.append(r)

    total    = len(all_results)
    correct  = sum(1 for r in all_results if r["correct"])
    accuracy = correct / total if total > 0 else 0.0   # [0,1]

    depth_breakdown = {}
    for d, dr in results_by_depth.items():
        dt = len(dr); dc = sum(1 for r in dr if r["correct"])
        depth_breakdown[f"{int(d * 100)}%"] = {
            "accuracy":      dc / dt if dt > 0 else 0.0,                           # [0,1]
            "avg_score":     sum(r["score"] for r in dr) / dt if dt > 0 else 0.0,  # [0,1]
            "avg_lev_score": sum(r["levenshtein_score"] for r in dr) / dt if dt > 0 else 0.0,
            "correct": dc, "total": dt,
        }

    return {"task": "M-RS", "task_description": "Multi-Needle Reasoning",
            "context_length": ctx_len, "accuracy": accuracy,
            "correct": correct, "total": total,
            "depth_breakdown": depth_breakdown, "trials": all_results}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def evaluate_needlebench(model_path, context_lengths=None, depth_percentages=None,
                         num_samples=None, n_gpu_layers=-1, n_batch=256,
                         tasks=None, needles_per_sample=3):
    """
    Run full NeedleBench evaluation.

    Model loaded once at n_ctx = max(context_lengths) with flash_attn=True.
    All output metrics are in [0, 1].
    """
    if context_lengths   is None: context_lengths   = [4096, 8192]
    if depth_percentages is None: depth_percentages = DEFAULT_DEPTHS
    if tasks             is None: tasks             = list(TASK_TYPES)

    n_ctx = max(context_lengths)

    print("=" * 70)
    print("NEEDLEBENCH EVALUATION")
    print("=" * 70)
    print(f"Model:             {model_path}")
    print(f"Context lengths:   {context_lengths}")
    print(f"Depths:            {[f'{int(d*100)}%' for d in depth_percentages]}")
    print(f"Tasks:             {tasks}")
    print(f"Samples cap:       {num_samples if num_samples else 'all'}")
    print(f"n_ctx:             {n_ctx}")
    print(f"Scoring (S-RT/M-RS): composite_retrieval_score = max(levenshtein, coverage, substr)")
    print("=" * 70)

    print("\nLoading model...")
    try:
        llm = Llama(model_path=model_path, n_gpu_layers=n_gpu_layers,
                    n_batch=n_batch, n_ctx=n_ctx, flash_attn=True, verbose=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{model_path}': {e}")
    print("  Model loaded.")

    print("\nLoading datasets...")
    haystack_ds    = load_needlebench_subset("en_haystack_texts", split="test")
    haystack_texts = [s["text"] for s in haystack_ds if s.get("text")]
    needle_ds      = load_needlebench_subset(
        "retrieval_needles", split="test", language_filter="English")
    reasoning_ds   = (load_needlebench_subset("multi_needle_reasoning_needle", split="test")
                      if "M-RS" in tasks else None)

    output = {
        "model": model_path, "benchmark": "needlebench",
        "benchmark_source": "https://huggingface.co/datasets/opencompass/NeedleBench",
        "paper_reference":  "arXiv:2407.11963",
        "context_lengths_tested":   context_lengths,
        "depth_percentages_tested": [int(d * 100) for d in depth_percentages],
        "tasks_evaluated":          tasks,
        "needles_per_sample_mrt":   needles_per_sample,
        "scoring_note": (
            "All three tasks (S-RT, M-RT, M-RS) use composite_retrieval_score = "
            "max(levenshtein_soft_score, predicted_coverage_score, substr_score). "
            "M-RT also stores legacy_f1 per needle for reference. "
            "levenshtein_score is stored per trial for S-RT and M-RS."
        ),
        "tasks":            {},
        # NOTE: All composite scores in [0, 1]. Multiply by 100 for percentages.
        "composite_scores": {},
        "metadata": {"evaluation_time_seconds": None, "num_samples_limit": num_samples},
    }

    for task in tasks:
        print(f"\n{'=' * 60}\nTask: {task}\n{'=' * 60}")
        task_results = []

        for ctx_len in context_lengths:
            print(f"\n  ctx_len={ctx_len} | {len(depth_percentages)} depths × samples...")
            if task == "S-RT":
                r = evaluate_single_retrieval(
                    llm, needle_ds, haystack_texts, ctx_len, depth_percentages, num_samples)
            elif task == "M-RT":
                r = evaluate_multi_retrieval(
                    llm, needle_ds, haystack_texts, ctx_len, depth_percentages,
                    num_samples, needles_per_sample)
            elif task == "M-RS":
                r = evaluate_multi_reasoning(
                    llm, reasoning_ds, haystack_texts, ctx_len, depth_percentages, num_samples)
            else:
                continue
            task_results.append(r)
            _print_task_result(r)

        output["tasks"][task] = task_results

        # All tasks now use accuracy as the composite metric
        key  = "accuracy"
        vals = [r[key] for r in task_results if key in r]
        output["composite_scores"][task] = sum(vals) / len(vals) if vals else 0.0  # [0,1]

    s_rt = output["composite_scores"].get("S-RT", 0.0)
    m_rt = output["composite_scores"].get("M-RT", 0.0)
    m_rs = output["composite_scores"].get("M-RS", 0.0)
    output["composite_scores"]["Overall"] = 0.4 * s_rt + 0.3 * m_rt + 0.3 * m_rs  # [0,1]

    print(f"\n{'=' * 60}\nNEEDLEBENCH RESULTS SUMMARY\n{'=' * 60}")
    print(f"S-RT  accuracy : {s_rt * 100:.2f}%")
    print(f"M-RT  accuracy : {m_rt * 100:.2f}%")
    print(f"M-RS  accuracy : {m_rs * 100:.2f}%")
    print(f"Overall        : {output['composite_scores']['Overall'] * 100:.2f}%")
    print(f"Formula: 0.4·S-RT + 0.3·M-RT + 0.3·M-RS\n{'=' * 60}")

    return output


def _print_task_result(result):
    task = result.get("task", "?")
    ctx  = result.get("context_length", 0)
    bd   = result.get("depth_breakdown", {})
    print(f"\n  ── {task} @ {ctx // 1024}k ──")
    print(f"  Accuracy: {result.get('accuracy', 0) * 100:.1f}%")
    if bd:
        print(f"  {'Depth':<10} {'Score':>8}")
        for label, stats in sorted(bd.items(), key=lambda x: int(x[0].rstrip("%"))):
            score = stats.get("accuracy", 0) * 100
            print(f"  {label:<10} {score:>7.1f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    from datetime import datetime

    p = argparse.ArgumentParser(
        description="NeedleBench with depth sweep (requires official dataset)")
    p.add_argument("model_path")
    p.add_argument("--context-lengths",    type=int, nargs="+", default=[4096, 8192])
    p.add_argument("--depths",             type=int, nargs="+", default=[10, 30, 50, 70, 90],
                   metavar="DEPTH_PCT")
    p.add_argument("--num-samples",        type=int, default=None)
    p.add_argument("--n-gpu-layers",       type=int, default=-1)
    p.add_argument("--n-batch",            type=int, default=256)
    p.add_argument("--tasks",              type=str, nargs="+",
                   default=["S-RT", "M-RT", "M-RS"],
                   choices=["S-RT", "M-RT", "M-RS"])
    p.add_argument("--needles-per-sample", type=int, default=3)
    p.add_argument("--output",             type=str, default=None)
    args = p.parse_args()

    start = datetime.now()
    results = evaluate_needlebench(
        model_path=args.model_path,
        context_lengths=args.context_lengths,
        depth_percentages=[d / 100.0 for d in args.depths],
        num_samples=args.num_samples,
        n_gpu_layers=args.n_gpu_layers,
        n_batch=args.n_batch,
        tasks=args.tasks,
        needles_per_sample=args.needles_per_sample,
    )
    results["metadata"]["evaluation_time_seconds"] = (datetime.now() - start).total_seconds()

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {out}")

    return results


if __name__ == "__main__":
    main()