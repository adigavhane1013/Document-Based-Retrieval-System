"""
tests/test_query_rewriter.py

Unit tests for rag/query_rewriter.py module.

Tests cover:
  - Ambiguity detection accuracy
  - Conditional rewriting logic
  - Rewriting output validation
  - Metadata logging completeness
  - Edge cases and error handling
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from rag.query_rewriter import (
    detect_ambiguity,
    should_rewrite,
    rewrite_query,
    rewrite_with_logging,
)
from configs.settings import settings


# ─────────────────────────────────────────────────────────────────────────────
# AMBIGUITY DETECTION TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestAmbiguityDetection:
    """Test cases for ambiguity score calculation"""
    
    def test_empty_query_maximum_ambiguity(self):
        """Empty query should have maximum ambiguity"""
        score = detect_ambiguity("")
        assert score == 1.0
        
    def test_whitespace_only_maximum_ambiguity(self):
        """Whitespace-only query should have maximum ambiguity"""
        score = detect_ambiguity("   ")
        assert score == 1.0
    
    def test_very_short_query_high_ambiguity(self):
        """Very short query (< 5 chars) should have high ambiguity"""
        score = detect_ambiguity("what")
        assert score > 0.5  # Should be ambiguous
    
    def test_vague_query_high_ambiguity(self):
        """Query with vague words should have higher ambiguity"""
        # "What about symptoms?" is vague
        score = detect_ambiguity("What about symptoms?")
        assert score > 0.6  # Should be quite ambiguous
    
    def test_specific_query_low_ambiguity(self):
        """Specific, detailed query should have low ambiguity"""
        score = detect_ambiguity("What are the symptoms of mild cognitive impairment?")
        assert score < 0.5  # Should be relatively clear
    
    def test_very_specific_query_minimal_ambiguity(self):
        """Very specific query should have minimal ambiguity"""
        score = detect_ambiguity("List the three types of cognitive impairment mentioned in the paper")
        assert score < 0.3  # Should be very clear
    
    def test_ambiguity_score_range(self):
        """Ambiguity score should always be between 0 and 1"""
        test_queries = [
            "",
            "a",
            "what",
            "What about it?",
            "List all cognitive domains",
            "Detailed explanation of how the reinforcement learning contextual bandit algorithm adapts difficulty"
        ]
        for query in test_queries:
            score = detect_ambiguity(query)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range for query: {query}"
    
    def test_question_starting_with_what_vague(self):
        """Simple 'what' questions are typically vague"""
        score = detect_ambiguity("What about it?")
        assert score > 0.5
    
    def test_question_with_specifics_clear(self):
        """Questions with specific terms are clearer"""
        score = detect_ambiguity("What treatment options exist for MCI?")
        assert score < 0.6


# ─────────────────────────────────────────────────────────────────────────────
# CONDITIONAL REWRITING TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestConditionalRewriting:
    """Test cases for should_rewrite() logic"""
    
    def test_vague_query_should_rewrite(self):
        """Vague queries should be flagged for rewriting"""
        # Using default threshold
        should = should_rewrite("What about symptoms?", threshold=0.6)
        assert should is True
    
    def test_specific_query_should_not_rewrite(self):
        """Specific queries should not be rewritten"""
        should = should_rewrite("What are the symptoms of MCI?", threshold=0.6)
        assert should is False
    
    def test_very_short_query_should_rewrite(self):
        """Very short queries (< MIN_LENGTH) should be rewritten"""
        short_query = "why"  # Only 3 chars, < QUERY_MIN_LENGTH (5)
        should = should_rewrite(short_query, threshold=0.9)  # High threshold shouldn't matter
        assert should is True  # Should still rewrite due to length
    
    def test_threshold_sensitivity(self):
        """should_rewrite() should respect threshold parameter"""
        query = "What about it?"
        
        # With high threshold, should not rewrite
        should_high = should_rewrite(query, threshold=0.9)
        assert should_high is False
        
        # With low threshold, should rewrite
        should_low = should_rewrite(query, threshold=0.3)
        assert should_low is True
    
    def test_empty_query_should_rewrite(self):
        """Empty queries should be flagged for rewriting"""
        should = should_rewrite("", threshold=0.5)
        assert should is True


# ─────────────────────────────────────────────────────────────────────────────
# QUERY REWRITING TESTS (WITH MOCKING)
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryRewriting:
    """Test cases for rewrite_query() function"""
    
    @patch('rag.query_rewriter.ChatOpenAI')
    def test_rewrite_query_valid_output(self, mock_llm_class):
        """rewrite_query() should return a non-empty string"""
        # Setup mock
        mock_response = Mock()
        mock_response.content = "What are the symptoms of mild cognitive impairment in early stages?"
        
        mock_llm = Mock()
        mock_llm.invoke.return_value = mock_response
        mock_llm_class.return_value = mock_llm
        
        # Test
        result = rewrite_query("What about symptoms?")
        
        # Assert
        assert isinstance(result, str)
        assert len(result) > 0
        assert "symptom" in result.lower()
    
    @patch('rag.query_rewriter.ChatOpenAI')
    def test_rewrite_query_maintains_intent(self, mock_llm_class):
        """Rewritten query should maintain original intent"""
        mock_response = Mock()
        mock_response.content = "How does the Epsilon-Greedy Contextual Bandit algorithm work in the MCI app?"
        
        mock_llm = Mock()
        mock_llm.invoke.return_value = mock_response
        mock_llm_class.return_value = mock_llm
        
        # Test
        result = rewrite_query("How does algorithm work?")
        
        # Assert
        assert "algorithm" in result.lower()
    
    @patch('rag.query_rewriter.ChatOpenAI')
    def test_rewrite_query_respects_length_limit(self, mock_llm_class):
        """Rewritten query should not exceed MAX_REWRITE_LENGTH"""
        # Mock with overly long response
        mock_response = Mock()
        mock_response.content = "a " * 200  # Very long output
        
        mock_llm = Mock()
        mock_llm.invoke.return_value = mock_response
        mock_llm_class.return_value = mock_llm
        
        # Test
        result = rewrite_query("short")
        
        # Assert
        assert len(result) <= settings.QUERY_MAX_REWRITE_LENGTH
    
    @patch('rag.query_rewriter.ChatOpenAI')
    def test_rewrite_query_fallback_on_llm_error(self, mock_llm_class):
        """rewrite_query() should return original query if LLM fails"""
        # Setup mock to raise exception
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("API error")
        mock_llm_class.return_value = mock_llm
        
        # Test
        original = "What about symptoms?"
        result = rewrite_query(original)
        
        # Assert - should return original query on error
        assert result == original
    
    @patch('rag.query_rewriter.ChatOpenAI')
    def test_rewrite_query_fallback_on_empty_response(self, mock_llm_class):
        """rewrite_query() should return original if LLM returns empty"""
        mock_response = Mock()
        mock_response.content = ""  # Empty response
        
        mock_llm = Mock()
        mock_llm.invoke.return_value = mock_response
        mock_llm_class.return_value = mock_llm
        
        # Test
        original = "What about it?"
        result = rewrite_query(original)
        
        # Assert
        assert result == original


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestRewriteWithLogging:
    """Test cases for rewrite_with_logging() - main entry point"""
    
    @patch('rag.query_rewriter.rewrite_query')
    @patch('rag.query_rewriter.should_rewrite')
    def test_returns_tuple_of_query_and_metadata(self, mock_should_rewrite, mock_rewrite):
        """Should return (query, metadata) tuple"""
        mock_should_rewrite.return_value = False
        
        result = rewrite_with_logging("test query")
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        query, metadata = result
        assert isinstance(query, str)
        assert isinstance(metadata, dict)
    
    @patch('rag.query_rewriter.rewrite_query')
    @patch('rag.query_rewriter.should_rewrite')
    def test_metadata_contains_required_fields(self, mock_should_rewrite, mock_rewrite):
        """Metadata should contain all required fields"""
        mock_should_rewrite.return_value = False
        
        _, metadata = rewrite_with_logging("test query")
        
        required_fields = [
            "rewrite_id",
            "original_query",
            "rewritten_query",
            "was_rewritten",
            "ambiguity_score",
            "timestamp",
            "elapsed_ms"
        ]
        
        for field in required_fields:
            assert field in metadata, f"Missing field: {field}"
    
    @patch('rag.query_rewriter.rewrite_query')
    @patch('rag.query_rewriter.should_rewrite')
    def test_metadata_rewrite_id_is_unique(self, mock_should_rewrite, mock_rewrite):
        """Each call should generate unique rewrite_id"""
        mock_should_rewrite.return_value = False
        
        _, metadata1 = rewrite_with_logging("query 1")
        _, metadata2 = rewrite_with_logging("query 2")
        
        assert metadata1["rewrite_id"] != metadata2["rewrite_id"]
    
    @patch('rag.query_rewriter.rewrite_query')
    @patch('rag.query_rewriter.should_rewrite')
    def test_was_rewritten_false_when_not_rewritten(self, mock_should_rewrite, mock_rewrite):
        """was_rewritten should be False if no rewriting occurred"""
        mock_should_rewrite.return_value = False
        
        query, metadata = rewrite_with_logging("specific query")
        
        assert metadata["was_rewritten"] is False
        assert metadata["rewritten_query"] == query
    
    @patch('rag.query_rewriter.rewrite_query')
    @patch('rag.query_rewriter.should_rewrite')
    def test_was_rewritten_true_when_rewritten(self, mock_should_rewrite, mock_rewrite):
        """was_rewritten should be True if rewriting occurred"""
        mock_should_rewrite.return_value = True
        mock_rewrite.return_value = "Rewritten query"
        
        _, metadata = rewrite_with_logging("vague query")
        
        assert metadata["was_rewritten"] is True
        assert metadata["original_query"] == "vague query"
        assert metadata["rewritten_query"] == "Rewritten query"
    
    @patch('rag.query_rewriter.rewrite_query')
    @patch('rag.query_rewriter.should_rewrite')
    def test_ambiguity_score_in_metadata(self, mock_should_rewrite, mock_rewrite):
        """Metadata should include ambiguity score"""
        mock_should_rewrite.return_value = False
        
        _, metadata = rewrite_with_logging("test")
        
        assert "ambiguity_score" in metadata
        assert isinstance(metadata["ambiguity_score"], float)
        assert 0.0 <= metadata["ambiguity_score"] <= 1.0
    
    @patch('rag.query_rewriter.rewrite_query')
    @patch('rag.query_rewriter.should_rewrite')
    def test_elapsed_time_recorded(self, mock_should_rewrite, mock_rewrite):
        """Metadata should include elapsed time"""
        mock_should_rewrite.return_value = False
        
        _, metadata = rewrite_with_logging("query")
        
        assert "elapsed_ms" in metadata
        assert isinstance(metadata["elapsed_ms"], float)
        assert metadata["elapsed_ms"] >= 0
    
    @patch('rag.query_rewriter.rewrite_query')
    @patch('rag.query_rewriter.should_rewrite')
    def test_timestamp_iso_format(self, mock_should_rewrite, mock_rewrite):
        """Timestamp should be in ISO format"""
        mock_should_rewrite.return_value = False
        
        _, metadata = rewrite_with_logging("query")
        
        # Should not raise exception
        timestamp = metadata["timestamp"]
        from datetime import datetime
        datetime.fromisoformat(timestamp)  # Will raise if invalid format


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:
    """Integration tests - actual behavior without mocking"""
    
    def test_full_pipeline_vague_query_disabled_rewriting(self):
        """Test full pipeline with rewriting disabled"""
        # Temporarily disable rewriting
        original_setting = settings.ENABLE_QUERY_REWRITING
        settings.ENABLE_QUERY_REWRITING = False
        
        try:
            query, metadata = rewrite_with_logging("What about symptoms?")
            
            # Should not rewrite even if it's vague
            assert query == "What about symptoms?"
            assert metadata["was_rewritten"] is False
        finally:
            settings.ENABLE_QUERY_REWRITING = original_setting
    
    def test_ambiguity_detection_consistency(self):
        """Same query should have same ambiguity score"""
        query = "What about the treatment options?"
        
        score1 = detect_ambiguity(query)
        score2 = detect_ambiguity(query)
        
        assert score1 == score2
    
    def test_should_rewrite_consistency(self):
        """Same query should have consistent rewriting decision"""
        query = "What about symptoms?"
        threshold = 0.6
        
        decision1 = should_rewrite(query, threshold)
        decision2 = should_rewrite(query, threshold)
        
        assert decision1 == decision2


# ─────────────────────────────────────────────────────────────────────────────
# EDGE CASES
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_unicode_query(self):
        """Should handle unicode characters"""
        query = "What about café and naïve?"
        score = detect_ambiguity(query)
        assert 0.0 <= score <= 1.0
    
    def test_very_long_query(self):
        """Should handle very long queries"""
        query = "a " * 500
        score = detect_ambiguity(query)
        assert 0.0 <= score <= 1.0
    
    def test_special_characters(self):
        """Should handle special characters"""
        query = "What??? About!!! Symptoms??? (!@#$%)"
        score = detect_ambiguity(query)
        assert 0.0 <= score <= 1.0
    
    def test_mixed_case(self):
        """Should handle mixed case"""
        query1 = "WHAT ABOUT SYMPTOMS?"
        query2 = "what about symptoms?"
        query3 = "What About Symptoms?"
        
        score1 = detect_ambiguity(query1)
        score2 = detect_ambiguity(query2)
        score3 = detect_ambiguity(query3)
        
        # Should be similar (case-insensitive)
        assert abs(score1 - score2) < 0.05
        assert abs(score2 - score3) < 0.05


# ─────────────────────────────────────────────────────────────────────────────
# RUN TESTS
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])