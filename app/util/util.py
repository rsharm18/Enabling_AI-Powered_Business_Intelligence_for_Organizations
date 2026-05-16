from typing import List, Dict, Any


def format_context(
    results: List[Dict[str, Any]],
    max_total_chars: int = 6000
) -> str:
    """
    Format retrieved search results into concise context for the LLM.

    Includes:
    - source/document name
    - retrieval score
    - similarity
    - keyword boost
    """

    if not results:
        return "No relevant documents found."

    context_parts = []
    current_length = 0

    for i, result in enumerate(results, 1):
        result_type = result.get("type", "document")

        score = result.get("final_score", result.get("similarity", 0))
        similarity = result.get("similarity", 0)
        boost = result.get("keyword_boost", 0)

        if result_type == "document":
            content = result.get("content", "")
            metadata = result.get("metadata", {})
        else:
            content = result.get("chunk_text", "")
            metadata = result.get("document_metadata", {})

        source_type = metadata.get("source_type", "")
        analysis_section = metadata.get("analysis_section", "")
        doc_type = metadata.get("type", "")

        source_name = (
            metadata.get("source_name")
            or metadata.get("document_name")
            or metadata.get("file_name")
            or metadata.get("title")
            or analysis_section
            or source_type
            or "unknown_source"
        )

        block = (
            f"[Source {i}] "
            f"name={source_name}, "
            f"type={result_type}, "
            f"score={score:.3f}, "
            f"similarity={similarity:.3f}, "
            f"boost={boost:.3f}, "
            f"source_type={source_type}, "
            f"doc_type={doc_type}\n"
            f"{content}\n"
        )

        if current_length + len(block) > max_total_chars:
            remaining = max_total_chars - current_length

            if remaining > 500:
                context_parts.append(block[:remaining] + "...")

            break

        context_parts.append(block)
        current_length += len(block)

    return "\n\n".join(context_parts)