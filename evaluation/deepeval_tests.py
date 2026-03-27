"""
evaluation/deepeval_tests.py

Unit-style RAG tests using DeepEval.

DeepEval lets you write pytest-style assertions for LLM outputs.
Run these in CI: pytest evaluation/deepeval_tests.py -v

Install: pip install deepeval pytest
"""

import pytest
import os
from dotenv import load_dotenv
load_dotenv()

try:
    from deepeval import assert_test
    from deepeval.metrics import (
        FaithfulnessMetric,
        AnswerRelevancyMetric,
        HallucinationMetric,
        ContextualRecallMetric,
    )
    from deepeval.test_case import LLMTestCase
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False


# ── Sample test cases (replace with your real eval set) ────────────────────────

SAMPLE_CASES = [
    {
        "input":             "What is the penalty for late delivery?",
        "actual_output":     "The penalty for late delivery is 2% of the contract value per week [SOURCE:abc123].",
        "expected_output":   "Late delivery incurs a 2% per week penalty capped at 10%.",
        "retrieval_context": [
            "10.2 In the event of late delivery, the Supplier shall pay a penalty of 2% of "
            "the total contract value per week of delay, up to a maximum of 10%."
        ],
    },
    {
        "input":             "When does the warranty expire?",
        "actual_output":     "The warranty expires 24 months after delivery [SOURCE:def456].",
        "expected_output":   "The warranty period is 24 months from the date of delivery.",
        "retrieval_context": [
            "6.1 The Supplier warrants all delivered goods for a period of 24 months "
            "from the date of acceptance by the Buyer."
        ],
    },
]


@pytest.mark.skipif(not DEEPEVAL_AVAILABLE, reason="deepeval not installed")
@pytest.mark.parametrize("case", SAMPLE_CASES)
def test_faithfulness(case):
    """Answer should not contain claims absent from the retrieved context."""
    test_case = LLMTestCase(
        input=case["input"],
        actual_output=case["actual_output"],
        retrieval_context=case["retrieval_context"],
    )
    metric = FaithfulnessMetric(threshold=0.7, model="gpt-4o")
    assert_test(test_case, [metric])


@pytest.mark.skipif(not DEEPEVAL_AVAILABLE, reason="deepeval not installed")
@pytest.mark.parametrize("case", SAMPLE_CASES)
def test_answer_relevancy(case):
    """Answer should directly address the question."""
    test_case = LLMTestCase(
        input=case["input"],
        actual_output=case["actual_output"],
    )
    metric = AnswerRelevancyMetric(threshold=0.7, model="gpt-4o")
    assert_test(test_case, [metric])


@pytest.mark.skipif(not DEEPEVAL_AVAILABLE, reason="deepeval not installed")
@pytest.mark.parametrize("case", SAMPLE_CASES)
def test_no_hallucination(case):
    """Hallucination score should be below 0.3."""
    test_case = LLMTestCase(
        input=case["input"],
        actual_output=case["actual_output"],
        context=case["retrieval_context"],
    )
    metric = HallucinationMetric(threshold=0.3, model="gpt-4o")
    assert_test(test_case, [metric])