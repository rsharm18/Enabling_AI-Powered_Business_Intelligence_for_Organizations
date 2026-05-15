import re
from typing import Dict, List, Any


def _is_bi_query( query: str) -> bool:
    bi_keywords = {
        "sales", "revenue", "profit", "product", "products",
        "region", "regional", "customer", "segment", "segmentation",
        "trend", "trends", "median", "average", "mean",
        "std", "standard", "deviation", "best", "highest",
        "lowest", "top", "bottom", "performance", "demographics",
    }
    tokens = set(re.findall(r"\b[a-z0-9]+\b", query.lower()))
    return bool(tokens.intersection(bi_keywords))


def _extract_query_terms( query: str) -> Dict[str, List[str]]:
    query_lower = query.lower()

    stop_words = {
        "tell", "me", "about", "which", "what", "who", "where", "when",
        "why", "how", "is", "are", "was", "were", "the", "a", "an",
        "for", "of", "to", "in", "on", "had", "has", "have", "show",
        "give", "please",
    }

    entity_phrases = re.findall(
        r"\b(?:widget|product|region|customer|segment)\s+[a-z0-9]+\b",
        query_lower,
    )

    tokens = re.findall(r"\b[a-z0-9]+\b", query_lower)
    useful_tokens = [
        token for token in tokens
        if token not in stop_words and len(token) > 1
    ]

    return {
        "phrases": list(set(entity_phrases)),
        "tokens": list(set(useful_tokens)),
    }


def calculate_keyword_boost( query: str, result: Dict[str, Any]) -> float:
    terms = _extract_query_terms(query)

    content = (
            result.get("chunk_text")
            or result.get("content")
            or result.get("document_content")
            or ""
    ).lower()

    metadata = result.get("metadata") or result.get("document_metadata") or {}
    metadata_text = " ".join(str(v) for v in metadata.values()).lower()

    boost = 0.0

    # Strong boost for exact entity phrase: "widget a", "product x", etc.
    for phrase in terms["phrases"]:
        if phrase in content:
            boost += 0.35
        elif phrase in metadata_text:
            boost += 0.20

    # Smaller boost for useful words like product, sales, region, widget.
    for token in terms["tokens"]:
        if token in content:
            boost += 0.05
        elif token in metadata_text:
            boost += 0.03

    # Strong BI boost: for BI questions, prefer csv_analysis_pickle chunks/docs.
    if _is_bi_query(query):
        source_type = str(metadata.get("source_type", "")).lower()
        doc_type = str(metadata.get("type", "")).lower()
        analysis_section = str(metadata.get("analysis_section", "")).lower()

        if (
                "csv_analysis" in source_type
                or "csv_analysis" in doc_type
                or "pickle" in source_type
                or analysis_section
        ):
            boost += 0.45

    return min(boost, 0.90)