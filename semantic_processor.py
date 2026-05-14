import os
import re
import unicodedata
import warnings

import torch
from huggingface_hub.utils import logging as hf_logging
from rapidfuzz import fuzz
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers.utils import logging as transformers_logging


os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

warnings.filterwarnings("ignore", message="You are sending unauthenticated requests.*")
transformers_logging.set_verbosity_error()
hf_logging.set_verbosity_error()


MODEL_NAME = "vinai/phobert-base"
TOP_K = 50
ORIGINAL_KEEP_RANK = 3
ORIGINAL_KEEP_ANY_TOP = 20
PHONETIC_THRESHOLD = 75

_tokenizer = None
_model = None


def _ensure_model_loaded():
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return

    print(f"Loading semantic model: {MODEL_NAME}")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    _model.eval()
    print("Semantic model is ready.")


def syllable_tokenize(text):
    return re.findall(r"[\w']+|[^\w\s]", text, re.UNICODE)


def is_word_token(token):
    return token.replace("'", "").isalpha()


def is_word_candidate(token):
    return token.isalpha()


def strip_tone(word):
    normalized = unicodedata.normalize("NFD", word)
    return "".join(char for char in normalized if not unicodedata.combining(char)).lower()


def apply_case(candidate, original):
    return candidate.capitalize() if original[:1].isupper() else candidate


def is_probably_name(tokens, idx):
    return idx > 0 and tokens[idx][:1].isupper()


def is_phonetically_similar(candidate, original):
    candidate_base = strip_tone(candidate)
    original_base = strip_tone(original)

    if candidate_base == original_base:
        return True

    if len(candidate_base) != len(original_base):
        return False

    if candidate_base[:1] != original_base[:1]:
        return False

    if candidate_base[-1:] != original_base[-1:]:
        return False

    return fuzz.ratio(candidate_base, original_base) >= PHONETIC_THRESHOLD


def predict_masked(tokens, mask_idx, top_k=TOP_K):
    _ensure_model_loaded()

    masked_tokens = tokens.copy()
    masked_tokens[mask_idx] = _tokenizer.mask_token
    input_text = " ".join(masked_tokens)
    inputs = _tokenizer(input_text, return_tensors="pt")

    with torch.no_grad():
        outputs = _model(**inputs)

    mask_positions = (
        inputs["input_ids"][0] == _tokenizer.mask_token_id
    ).nonzero(as_tuple=False)
    if len(mask_positions) == 0:
        return []

    mask_position = mask_positions[0].item()
    logits = outputs.logits[0, mask_position]
    top_indices = torch.topk(logits, top_k).indices.tolist()

    candidates = []
    seen = set()
    for token_id in top_indices:
        candidate = _tokenizer.decode([token_id]).strip()
        key = candidate.lower()
        if not is_word_candidate(candidate) or key in seen:
            continue
        candidates.append(candidate)
        seen.add(key)

    return candidates


def correct_text(raw_text):
    if not raw_text:
        return {
            "original": raw_text,
            "corrected": raw_text,
            "changes": [],
            "has_changes": False,
        }

    tokens = syllable_tokenize(raw_text)
    result = tokens.copy()
    changes = []
    word_indices = [idx for idx, token in enumerate(tokens) if is_word_token(token)]

    for idx in word_indices:
        original = tokens[idx]
        if is_probably_name(tokens, idx):
            continue

        candidates = predict_masked(result, idx, top_k=TOP_K)
        candidate_words = [candidate.lower() for candidate in candidates]
        original_lower = original.lower()

        if original_lower in candidate_words[:ORIGINAL_KEEP_RANK]:
            continue

        if original_lower in candidate_words[:ORIGINAL_KEEP_ANY_TOP]:
            continue

        best_candidate = None
        for candidate in candidates:
            if candidate.lower() == original_lower:
                continue

            if is_phonetically_similar(candidate, original):
                best_candidate = candidate
                break

        if best_candidate:
            best_candidate = apply_case(best_candidate, original)
            result[idx] = best_candidate
            changes.append({"from": original, "to": best_candidate})

    corrected = " ".join(result)
    corrected = re.sub(r"\s([,.;:!?])", r"\1", corrected)

    return {
        "original": raw_text,
        "corrected": corrected,
        "changes": changes,
        "has_changes": len(changes) > 0,
    }


def correct_transcription(results, full_text=None):
    """Post-process faster-whisper output while preserving segment timestamps."""
    if not results:
        correction = correct_text(full_text or "")
        return [], correction["corrected"], correction["changes"]

    corrected_results = []
    corrected_text_parts = []
    all_changes = []

    for segment_index, segment in enumerate(results):
        original_text = segment.get("text", "")
        correction = correct_text(original_text)
        corrected_text = correction["corrected"]

        corrected_segment = {**segment, "text": corrected_text}
        corrected_results.append(corrected_segment)
        corrected_text_parts.append(corrected_text)

        for change in correction["changes"]:
            all_changes.append(
                {
                    **change,
                    "segment_index": segment_index,
                    "start": segment.get("start"),
                    "end": segment.get("end"),
                }
            )

    return corrected_results, " ".join(corrected_text_parts), all_changes
