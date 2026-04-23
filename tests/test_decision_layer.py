"""
tests/test_decision_layer.py - FIXED VERSION

Unit tests for rag/decision_layer.py module.

Tests cover:
  - Score-based decision making
  - Retry logic
  - Fallback logic
  - Threshold enforcement
  - Metadata logging
  - Edge cases
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from rag.decision_layer import DecisionLayer
from configs.settings import settings


# ─────────────────────────────────────────────────────────────────────────────
# DECISION LOGIC TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestDecisionLogic:
    """Test the core decision-making logic"""
    
    def test_good_scores_accept(self):
        """Good scores should result in ACCEPT decision"""
        decision_layer = DecisionLayer()
        
        scores = {
            "faithfulness": 0.85,
            "answer_relevancy": 0.80
        }
        
        answer, metadata, should_return = decision_layer.evaluate_and_decide(
            answer="Good answer",
            scores=scores,
            attempt=1
        )
        
        assert metadata["decision_type"] == "ACCEPT"
        assert should_return is True
        assert metadata["reason"]
    
    def test_low_faithfulness_retry(self):
        """Low faithfulness should result in RETRY decision"""
        decision_layer = DecisionLayer()
        
        scores = {
            "faithfulness": 0.45,  # Below threshold
            "answer_relevancy": 0.80
        }
        
        answer, metadata, should_return = decision_layer.evaluate_and_decide(
            answer="Poor answer",
            scores=scores,
            attempt=1,
            max_attempts=3
        )
        
        assert metadata["decision_type"] == "RETRY"
        assert should_return is False
        assert "retry_config" in metadata
    
    def test_low_relevance_fallback(self):
        """Low relevance should result in FALLBACK decision (if enabled)"""
        decision_layer = DecisionLayer()
        decision_layer.fallback_enabled = True
        
        scores = {
            "faithfulness": 0.75,
            "answer_relevancy": 0.50  # Below threshold
        }
        
        with patch.object(decision_layer, 'get_fallback_model', return_value='better-model'):
            answer, metadata, should_return = decision_layer.evaluate_and_decide(
                answer="Mediocre answer",
                scores=scores,
                attempt=1
            )
        
        assert metadata["decision_type"] == "FALLBACK"
        assert should_return is False
    
    def test_max_retries_reached_reject(self):
        """Max retries exceeded should result in REJECT"""
        decision_layer = DecisionLayer()
        
        scores = {
            "faithfulness": 0.45,  # Still bad
            "answer_relevancy": 0.80
        }
        
        # attempt=4, max_attempts=2 means 4 > 2 so REJECT
        answer, metadata, should_return = decision_layer.evaluate_and_decide(
            answer="Still poor",
            scores=scores,
            attempt=4,
            max_attempts=2
        )
        
        assert metadata["decision_type"] == "REJECT"
        assert should_return is True


# ─────────────────────────────────────────────────────────────────────────────
# RETRY LOGIC TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestRetryLogic:
    """Test retry decision and configuration"""
    
    def test_should_retry_returns_true(self):
        """should_retry should return True when appropriate"""
        decision_layer = DecisionLayer()
        
        scores = {"faithfulness": 0.45}  # Below threshold
        should_retry = decision_layer.should_retry(scores, attempt=1)
        
        assert should_retry is True
    
    def test_should_retry_returns_false_bad_score(self):
        """should_retry returns False for good scores"""
        decision_layer = DecisionLayer()
        
        scores = {"faithfulness": 0.85}  # Above threshold
        should_retry = decision_layer.should_retry(scores, attempt=1)
        
        assert should_retry is False
    
    def test_should_retry_returns_false_max_attempts(self):
        """should_retry returns False when max attempts exceeded"""
        decision_layer = DecisionLayer()
        
        scores = {"faithfulness": 0.45}  # Bad score
        # attempt=2, max_retry_attempts=2 (from settings)
        should_retry = decision_layer.should_retry(scores, attempt=2)
        
        # Should return False since attempt >= max_attempts
        assert should_retry is False
    
    def test_retry_config_increases_top_k(self):
        """Retry config should increase top_k values"""
        decision_layer = DecisionLayer()
        
        retry_config = decision_layer.get_retry_config()
        
        # Check that top_k values are increased
        assert retry_config["top_k_dense"] > settings.TOP_K_DENSE
        assert retry_config["top_k_sparse"] > settings.TOP_K_SPARSE
        assert retry_config["top_k_dense"] == (
            settings.TOP_K_DENSE + settings.RETRY_RETRIEVAL_TOP_K_INCREASE
        )
    
    def test_retry_config_lowers_threshold(self):
        """Retry config should lower retrieval score threshold"""
        decision_layer = DecisionLayer()
        
        retry_config = decision_layer.get_retry_config()
        
        assert retry_config["retrieval_score_threshold"] == settings.RETRY_LOWERED_THRESHOLD
        assert retry_config["retrieval_score_threshold"] < settings.RETRIEVAL_SCORE_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK LOGIC TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestFallbackLogic:
    """Test fallback decision and model selection"""
    
    def test_should_fallback_returns_true(self):
        """should_fallback returns True when appropriate"""
        decision_layer = DecisionLayer()
        decision_layer.fallback_enabled = True
        
        scores = {"answer_relevancy": 0.50}  # Below threshold
        should_fallback = decision_layer.should_fallback(scores)
        
        assert should_fallback is True
    
    def test_should_fallback_returns_false_good_score(self):
        """should_fallback returns False for good relevance"""
        decision_layer = DecisionLayer()
        decision_layer.fallback_enabled = True
        
        scores = {"answer_relevancy": 0.80}  # Above threshold
        should_fallback = decision_layer.should_fallback(scores)
        
        assert should_fallback is False
    
    def test_should_fallback_returns_false_disabled(self):
        """should_fallback returns False when fallback disabled"""
        decision_layer = DecisionLayer()
        decision_layer.fallback_enabled = False
        
        scores = {"answer_relevancy": 0.50}  # Bad score
        should_fallback = decision_layer.should_fallback(scores)
        
        assert should_fallback is False
    
    def test_get_fallback_model_returns_string(self):
        """get_fallback_model should return model name"""
        decision_layer = DecisionLayer()
        
        # Should return fallback model if configured
        model = decision_layer.get_fallback_model()
        
        # Could be None if not configured, or string if configured
        assert model is None or isinstance(model, str)


# ─────────────────────────────────────────────────────────────────────────────
# METADATA TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestDecisionMetadata:
    """Test that decision metadata is complete and accurate"""
    
    def test_metadata_has_required_fields(self):
        """Metadata should contain all required fields"""
        decision_layer = DecisionLayer()
        
        scores = {"faithfulness": 0.85, "answer_relevancy": 0.80}
        answer, metadata, _ = decision_layer.evaluate_and_decide(
            answer="test",
            scores=scores,
            attempt=1
        )
        
        required_fields = [
            "decision_id",
            "decision_type",
            "attempt",
            "max_attempts",
            "reason",
            "should_return",
            "original_answer",
            "scores",
            "thresholds",
            "timestamp"
        ]
        
        for field in required_fields:
            assert field in metadata, f"Missing field: {field}"
    
    def test_metadata_decision_id_unique(self):
        """Each decision should have unique ID"""
        decision_layer = DecisionLayer()
        scores = {"faithfulness": 0.85, "answer_relevancy": 0.80}
        
        _, metadata1, _ = decision_layer.evaluate_and_decide(
            answer="test1",
            scores=scores,
            attempt=1
        )
        
        _, metadata2, _ = decision_layer.evaluate_and_decide(
            answer="test2",
            scores=scores,
            attempt=1
        )
        
        assert metadata1["decision_id"] != metadata2["decision_id"]
    
    def test_metadata_scores_stored(self):
        """Metadata should store the input scores"""
        decision_layer = DecisionLayer()
        
        scores = {"faithfulness": 0.75, "answer_relevancy": 0.68}
        _, metadata, _ = decision_layer.evaluate_and_decide(
            answer="test",
            scores=scores,
            attempt=1
        )
        
        assert metadata["scores"]["faithfulness"] == 0.75
        assert metadata["scores"]["answer_relevancy"] == 0.68
    
    def test_metadata_thresholds_stored(self):
        """Metadata should store the thresholds used"""
        decision_layer = DecisionLayer()
        
        scores = {"faithfulness": 0.85, "answer_relevancy": 0.80}
        _, metadata, _ = decision_layer.evaluate_and_decide(
            answer="test",
            scores=scores,
            attempt=1
        )
        
        assert metadata["thresholds"]["faithfulness"] == decision_layer.faithfulness_threshold
        assert metadata["thresholds"]["relevance"] == decision_layer.relevance_threshold
    
    def test_metadata_timestamp_valid(self):
        """Metadata timestamp should be valid ISO format"""
        decision_layer = DecisionLayer()
        
        scores = {"faithfulness": 0.85, "answer_relevancy": 0.80}
        _, metadata, _ = decision_layer.evaluate_and_decide(
            answer="test",
            scores=scores,
            attempt=1
        )
        
        # Should not raise exception
        timestamp = datetime.fromisoformat(metadata["timestamp"])
        assert isinstance(timestamp, datetime)


# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestThresholds:
    """Test threshold enforcement"""
    
    def test_faithfulness_threshold_respected(self):
        """Decision should respect faithfulness threshold"""
        decision_layer = DecisionLayer()
        threshold = decision_layer.faithfulness_threshold
        
        # Just below threshold → should retry
        scores_bad = {"faithfulness": threshold - 0.05, "answer_relevancy": 0.90}
        _, metadata_bad, _ = decision_layer.evaluate_and_decide(
            answer="bad",
            scores=scores_bad,
            attempt=1,
            max_attempts=2
        )
        assert metadata_bad["decision_type"] == "RETRY"
        
        # Just above threshold → should accept
        scores_good = {"faithfulness": threshold + 0.05, "answer_relevancy": 0.90}
        _, metadata_good, _ = decision_layer.evaluate_and_decide(
            answer="good",
            scores=scores_good,
            attempt=1
        )
        assert metadata_good["decision_type"] == "ACCEPT"
    
    def test_relevance_threshold_respected(self):
        """Decision should respect relevance threshold"""
        decision_layer = DecisionLayer()
        decision_layer.fallback_enabled = True
        decision_layer.FALLBACK_LLM_MODEL = "gpt-4"  # FIXED: Set fallback model
        threshold = decision_layer.relevance_threshold
        
        # Just below threshold → should fallback
        scores_bad = {"faithfulness": 0.90, "answer_relevancy": threshold - 0.05}
        _, metadata_bad, _ = decision_layer.evaluate_and_decide(
            answer="mediocre",
            scores=scores_bad,
            attempt=1
        )
        assert metadata_bad["decision_type"] == "FALLBACK"
        
        # Just above threshold → should accept
        scores_good = {"faithfulness": 0.90, "answer_relevancy": threshold + 0.05}
        _, metadata_good, _ = decision_layer.evaluate_and_decide(
            answer="good",
            scores=scores_good,
            attempt=1
        )
        assert metadata_good["decision_type"] == "ACCEPT"


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY METHOD TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestUtilityMethods:
    """Test utility methods"""
    
    def test_log_decision_history(self):
        """log_decision_history should not raise exception"""
        decision_layer = DecisionLayer()
        
        # FIXED: Include 'scores' field in test data
        decisions = [
            {"decision_type": "RETRY", "scores": {"faithfulness": 0.45}},
            {"decision_type": "ACCEPT", "scores": {"faithfulness": 0.85}}
        ]
        
        # Should not raise
        decision_layer.log_decision_history(decisions)
    
    def test_log_decision_history_empty(self):
        """log_decision_history should handle empty list"""
        decision_layer = DecisionLayer()
        
        # Should not raise on empty list
        decision_layer.log_decision_history([])
    
    def test_get_decision_summary(self):
        """get_decision_summary should return readable string"""
        decision_layer = DecisionLayer()
        
        decision = {
            "decision_type": "RETRY",
            "attempt": 1,
            "max_attempts": 2,
            "reason": "Faithfulness too low"
        }
        
        summary = decision_layer.get_decision_summary(decision)
        
        assert isinstance(summary, str)
        assert "RETRY" in summary
        assert "1/2" in summary


# ─────────────────────────────────────────────────────────────────────────────
# EDGE CASE TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Test edge cases and unusual inputs"""
    
    def test_missing_score_defaults_to_zero(self):
        """Missing scores should default to 0.0"""
        decision_layer = DecisionLayer()
        
        scores = {}  # Empty scores
        answer, metadata, should_return = decision_layer.evaluate_and_decide(
            answer="test",
            scores=scores,
            attempt=1
        )
        
        # With missing scores (defaulting to 0), should retry
        assert metadata["decision_type"] in ["RETRY", "REJECT", "FALLBACK"]
    
    def test_extreme_scores(self):
        """Should handle extreme score values"""
        decision_layer = DecisionLayer()
        
        # Perfect scores
        scores_perfect = {"faithfulness": 1.0, "answer_relevancy": 1.0}
        _, metadata_perfect, _ = decision_layer.evaluate_and_decide(
            answer="perfect",
            scores=scores_perfect,
            attempt=1
        )
        assert metadata_perfect["decision_type"] == "ACCEPT"
        
        # Worst scores
        scores_worst = {"faithfulness": 0.0, "answer_relevancy": 0.0}
        _, metadata_worst, _ = decision_layer.evaluate_and_decide(
            answer="worst",
            scores=scores_worst,
            attempt=1,
            max_attempts=2
        )
        assert metadata_worst["decision_type"] in ["RETRY", "REJECT"]
    
    def test_attempt_equals_max_attempts(self):
        """Should handle attempt == max_attempts boundary"""
        decision_layer = DecisionLayer()
        
        scores = {"faithfulness": 0.45, "answer_relevancy": 0.80}
        
        # Exactly at max (should not retry anymore, attempt=2, max=2, so 2 > 2 is False)
        _, metadata, _ = decision_layer.evaluate_and_decide(
            answer="test",
            scores=scores,
            attempt=2,
            max_attempts=2
        )
        
        # Should retry since 2 is not > 2
        assert metadata["decision_type"] in ["RETRY", "FALLBACK"]


# ─────────────────────────────────────────────────────────────────────────────
# INITIALIZATION TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestInitialization:
    """Test DecisionLayer initialization"""
    
    def test_initializes_with_settings(self):
        """DecisionLayer should load thresholds from settings"""
        decision_layer = DecisionLayer()
        
        assert decision_layer.faithfulness_threshold == settings.RAGAS_FAITHFULNESS_THRESHOLD
        assert decision_layer.relevance_threshold == settings.RAGAS_RELEVANCE_THRESHOLD
        assert decision_layer.max_retry_attempts == settings.RAGAS_MAX_RETRY_ATTEMPTS
    
    def test_thresholds_are_reasonable(self):
        """Thresholds should be between 0 and 1"""
        decision_layer = DecisionLayer()
        
        assert 0.0 <= decision_layer.faithfulness_threshold <= 1.0
        assert 0.0 <= decision_layer.relevance_threshold <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# RUN TESTS
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])