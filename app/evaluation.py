"""Evaluation utilities for the BI assistant."""

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import Config
from app.rag_analysis.langgraph_agent import ConversationalAgent, GROQ_AVAILABLE

logger = logging.getLogger(__name__)


DEFAULT_EVALUATION_CASES = [
    {
        "question": "Which product had the highest sales?",
        "reference": "The answer should identify the top product by total sales from the CSV analysis.",
        "expected_keywords": ["product", "sales"],
    },
    {
        "question": "Which region performed best for sales?",
        "reference": "The answer should identify the best region by total sales from the CSV analysis.",
        "expected_keywords": ["region", "sales"],
    },
    {
        "question": "What are the main benefits of AI in business intelligence?",
        "reference": "The answer should describe how AI supports BI through analysis, prediction, pattern detection, and decision support.",
        "expected_keywords": ["AI", "business intelligence", "data", "decision"],
    },
    {
        "question": "What statistical measures are available in the sales analysis?",
        "reference": "The answer should mention statistics such as median, standard deviation, mean, totals, or rankings.",
        "expected_keywords": ["median", "standard deviation", "mean", "sales"],
    },
]


@dataclass
class EvaluationConfig:
    """Configuration for an evaluation run."""

    eval_file: Optional[str] = None
    output_file: Optional[str] = None
    max_cases: Optional[int] = None


class AssistantEvaluator:
    """Run repeatable QA evaluation for the assistant."""

    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        self.agent = ConversationalAgent()
        self.qa_eval_chain_cls = self._load_qa_eval_chain()
        self.judge_llm = self._create_judge_llm()

    def _load_qa_eval_chain(self):
        """Load QAEvalChain if the installed LangChain distribution still provides it."""
        import importlib

        for module_name in (
            "langchain_classic.evaluation.qa",
            "langchain_classic.evaluation",
            "langchain.evaluation.qa",
            "langchain.evaluation",
        ):
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue

            qa_eval_chain = getattr(module, "QAEvalChain", None)
            if qa_eval_chain:
                logger.info(f"Using QAEvalChain from {module_name}")
                return qa_eval_chain

        logger.info("QAEvalChain is not available in the installed LangChain package")
        return None

    def _create_judge_llm(self):
        """Create a deterministic LLM judge when Groq is configured."""
        if not GROQ_AVAILABLE or not Config.GROQ_API_KEY:
            return None

        try:
            from langchain_groq import ChatGroq

            return ChatGroq(
                api_key=Config.GROQ_API_KEY,
                model=Config.GROQ_MODEL,
                temperature=0,
                max_tokens=300,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize evaluation judge LLM: {e}")
            return None

    def load_cases(self) -> List[Dict[str, Any]]:
        """Load evaluation cases from JSON or use built-in defaults."""
        path = Path(self.config.eval_file) if self.config.eval_file else Config.DATA_DIR / "evaluation_cases.json"

        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                cases = json.load(f)
        else:
            cases = DEFAULT_EVALUATION_CASES

        if self.config.max_cases:
            return cases[: self.config.max_cases]

        return cases

    def run(self) -> Dict[str, Any]:
        """Run evaluation and return a report dictionary."""
        cases = self.load_cases()
        results = []

        for index, case in enumerate(cases, 1):
            question = case["question"]
            logger.info(f"Evaluating case {index}/{len(cases)}: {question}")

            start = time.perf_counter()
            prediction = self.agent.chat(question, history=[])
            latency_seconds = time.perf_counter() - start

            grade = self.grade_answer(case, prediction)
            results.append({
                "question": question,
                "reference": case.get("reference", ""),
                "prediction": prediction,
                "latency_seconds": round(latency_seconds, 3),
                **grade,
            })

        passed = sum(1 for result in results if result.get("passed"))
        average_score = (
            sum(float(result.get("score", 0)) for result in results) / len(results)
            if results else 0.0
        )

        report = {
            "summary": {
                "total_cases": len(results),
                "passed": passed,
                "failed": len(results) - passed,
                "pass_rate": round(passed / len(results), 3) if results else 0.0,
                "average_score": round(average_score, 3),
                "grader": self._grader_name(),
                "report_path": str(self._output_path()),
            },
            "results": results,
        }

        self.save_report(report)
        return report

    def _grader_name(self) -> str:
        if self.qa_eval_chain_cls and self.judge_llm:
            return "QAEvalChain"
        if self.judge_llm:
            return "Groq LLM judge"
        return "keyword heuristic"

    def grade_answer(self, case: Dict[str, Any], prediction: str) -> Dict[str, Any]:
        """Grade a generated answer."""
        if self.qa_eval_chain_cls and self.judge_llm:
            qa_grade = self._grade_with_qa_eval_chain(case, prediction)
            if qa_grade:
                return qa_grade

        if self.judge_llm:
            llm_grade = self._grade_with_llm(case, prediction)
            if llm_grade:
                return llm_grade

        return self._grade_with_keywords(case, prediction)

    def _grade_with_qa_eval_chain(self, case: Dict[str, Any], prediction: str) -> Optional[Dict[str, Any]]:
        """Grade with QAEvalChain when the installed package exposes it."""
        try:
            chain = self.qa_eval_chain_cls.from_llm(self.judge_llm)
            graded = chain.evaluate(
                examples=[{
                    "query": case["question"],
                    "answer": case.get("reference", ""),
                }],
                predictions=[{
                    "query": case["question"],
                    "result": prediction,
                }],
            )
            raw = graded[0] if graded else {}
            value = str(raw.get("results", raw)).strip()
            passed = "CORRECT" in value.upper() or "GRADE: CORRECT" in value.upper()
            return {
                "score": 1.0 if passed else 0.0,
                "passed": passed,
                "reason": value,
            }
        except Exception as e:
            logger.warning(f"QAEvalChain grading failed, falling back: {e}")
            return None

    def _grade_with_llm(self, case: Dict[str, Any], prediction: str) -> Optional[Dict[str, Any]]:
        """Grade with a compact JSON-producing LLM judge."""
        prompt = f"""
You are grading a business intelligence assistant answer.

Question:
{case["question"]}

Reference answer:
{case.get("reference", "")}

Expected keywords:
{", ".join(case.get("expected_keywords", []))}

Assistant answer:
{prediction}

Return only JSON with:
score: number from 0 to 1
passed: boolean
reason: short string
"""
        try:
            response = self.judge_llm.invoke(prompt)
            content = response.content.strip()
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            payload = json.loads(match.group(0) if match else content)
            score = float(payload.get("score", 0))
            normalized_score = max(0.0, min(1.0, score))
            return {
                "score": normalized_score,
                "passed": normalized_score >= 0.7,
                "reason": str(payload.get("reason", "")),
            }
        except Exception as e:
            logger.warning(f"LLM judge grading failed, falling back: {e}")
            return None

    def _grade_with_keywords(self, case: Dict[str, Any], prediction: str) -> Dict[str, Any]:
        """Fallback grade based on expected keyword coverage and non-empty answer."""
        expected_keywords = [kw.lower() for kw in case.get("expected_keywords", [])]
        answer = prediction.lower()
        matched = [kw for kw in expected_keywords if kw in answer]

        if expected_keywords:
            score = len(matched) / len(expected_keywords)
        else:
            score = 1.0 if prediction.strip() else 0.0

        source_present = "source:" in answer
        adjusted_score = min(1.0, score + (0.1 if source_present else 0.0))

        return {
            "score": round(adjusted_score, 3),
            "passed": adjusted_score >= 0.7,
            "reason": f"Matched keywords: {matched}; source footer present: {source_present}",
        }

    def save_report(self, report: Dict[str, Any]) -> str:
        """Save the evaluation report to disk."""
        output_path = self._output_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Saved evaluation report to {output_path}")
        return str(output_path)

    def _output_path(self) -> Path:
        """Return the configured report output path."""
        return Path(self.config.output_file or Config.OUTPUT_DIR / "evaluation_report.json")


def run_evaluation(
    eval_file: Optional[str] = None,
    output_file: Optional[str] = None,
    max_cases: Optional[int] = None,
) -> Dict[str, Any]:
    """Run assistant evaluation and return the report."""
    evaluator = AssistantEvaluator(EvaluationConfig(
        eval_file=eval_file,
        output_file=output_file,
        max_cases=max_cases,
    ))
    return evaluator.run()
