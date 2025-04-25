# Web Research Agent - Test Script

import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import json
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import agent components - adjust the import path if necessary
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from trial1 import (
        research_sub_query,
        decompose_query,
        tool_tracker,
        detect_query_type,
        validate_sub_query,
        truncate_text,
        add_citations_to_summary,
        process_large_content,
        extract_search_sources,
        should_use_tools,
        sanitize_llm_output,
        RateLimiter
    )
except ImportError as e:
    logger.error(f"Error importing agent components: {e}")
    logger.info("Make sure you're running this script from the correct directory")
    sys.exit(1)

class TestWebResearchAgent(unittest.TestCase):
    
    def setUp(self):
        """Reset tracker and other state before each test"""
        tool_tracker.reset()
    
    def test_query_type_detection(self):
        """Test query type detection functionality"""
        # Using patch to mock LLM calls for consistent testing
        with patch('trial1.llm.predict') as mock_predict:
            # Set up mocked returns for different query types
            mock_predict.side_effect = [
                "historical",  # For historical query
                "opinion",     # For opinion query
                "news",        # For news query
                "factual",     # For factual query
                "technical"    # For technical query
            ]
            
            # Test each query type
            self.assertEqual(detect_query_type("What happened in World War 2?"), "historical")
            self.assertEqual(detect_query_type("What is the best smartphone?"), "opinion")
            self.assertEqual(detect_query_type("Latest developments in AI"), "news")
            self.assertEqual(detect_query_type("How tall is the Eiffel Tower?"), "factual")
            self.assertEqual(detect_query_type("How does React's virtual DOM work?"), "technical")
            
            # Verify LLM was called the expected number of times
            self.assertEqual(mock_predict.call_count, 5)
    
    def test_query_decomposition(self):
        """Test query decomposition functionality"""
        with patch('trial1.llm.predict') as mock_predict:
            # Mock decomposition response from LLM
            mock_predict.return_value = """
            1. What are the environmental impacts of manufacturing electric vehicles?
            2. What are the environmental impacts of using electric vehicles vs. gas vehicles?
            3. What are the environmental impacts of disposing of electric vehicles?
            4. How do the overall lifecycle emissions compare between electric and gas vehicles?
            """
            
            complex_query = "What are the environmental impacts of electric vehicles compared to gas vehicles, considering manufacturing, usage, and disposal?"
            sub_queries = decompose_query(complex_query)
            
            # Verify we get the expected number of sub-questions
            self.assertEqual(len(sub_queries), 4)
            
            # Verify sub-queries match expected content
            self.assertIn("manufacturing", sub_queries[0].lower())
            self.assertIn("using", sub_queries[1].lower())
            self.assertIn("disposing", sub_queries[2].lower())
            self.assertIn("lifecycle", sub_queries[3].lower())
    
    def test_sub_query_validation(self):
        """Test sub-query validation and fixing"""
        # Test normal valid sub-query
        valid_query = "How do electric vehicles compare to gas vehicles?"
        self.assertEqual(validate_sub_query(valid_query), valid_query)
        
        # Test malformed sub-query with colon issues
        malformed_query = "EV Manufacturing:  environmental impact assessment"
        expected_fixed = "EV Manufacturing: environmental impact assessment"
        self.assertEqual(validate_sub_query(malformed_query), expected_fixed)
        
        # Test empty or very short sub-query
        self.assertEqual(validate_sub_query(""), "General information about this topic")
        self.assertEqual(validate_sub_query("EV"), "General information about this topic")
    
    def test_truncate_text(self):
        """Test text truncation functionality"""
        # Generate test text of known length
        test_text = "x" * 10000
        
        # Test default truncation
        truncated = truncate_text(test_text, max_tokens=1000)
        # Should be roughly 4000 chars (with safety margin)
        self.assertLess(len(truncated), 4000)
        self.assertIn("[content truncated", truncated)
        
        # Test truncation with smaller limit
        truncated_small = truncate_text(test_text, max_tokens=100)
        self.assertLess(len(truncated_small), 400)
        
        # Test with text smaller than limit
        short_text = "This is a short text."
        self.assertEqual(truncate_text(short_text, max_tokens=100), short_text)
    
    def test_citation_addition(self):
        """Test adding citations to summaries"""
        test_summary = "This is a test summary about electric vehicles."
        test_sources = [
            {"title": "EV Report 2023", "link": "https://example.com/ev-report", "snippet": "Comprehensive report on EVs"},
            {"title": "Gas vs Electric", "link": "https://example.com/comparison", "snippet": "Comparison of vehicle types"}
        ]
        
        # Test citation addition
        with_citations = add_citations_to_summary(test_summary, test_sources)
        
        # Verify summary is included
        self.assertIn(test_summary, with_citations)
        
        # Verify sources are included
        self.assertIn("EV Report 2023", with_citations)
        self.assertIn("https://example.com/ev-report", with_citations)
        self.assertIn("Gas vs Electric", with_citations)
        
        # Test with empty sources
        self.assertEqual(add_citations_to_summary(test_summary, []), test_summary)
    
    def test_source_extraction(self):
        """Test extracting sources from search results"""
        # Mock formatted search results
        search_results = """
        🔗 [EV Environmental Impact](https://example.com/ev-impact)
        Electric vehicles have lower lifetime emissions.
        
        🔗 [Climate Effects of Transportation](https://example.com/climate)
        Transportation accounts for 29% of greenhouse gas emissions.
        """
        
        sources = extract_search_sources(search_results)
        
        # Verify sources were extracted correctly
        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0]["title"], "EV Environmental Impact")
        self.assertEqual(sources[0]["link"], "https://example.com/ev-impact")
        self.assertEqual(sources[1]["title"], "Climate Effects of Transportation")
    
    def test_tool_usage_determination(self):
        """Test determination of whether to use tools"""
        with patch('trial1.llm.predict') as mock_predict:
            # Set up responses for should_use_tools
            mock_predict.side_effect = ["YES", "NO"]
            
            # Test cases that should and shouldn't use tools
            self.assertTrue(should_use_tools("What is the latest research on quantum computing?"))
            self.assertFalse(should_use_tools("What is 2+2?"))
    
    def test_rate_limiter(self):
        """Test rate limiter functionality"""
        # Create rate limiter allowing 2 calls per 1 second
        limiter = RateLimiter(max_calls=2, time_window=1)
        
        # First two calls should be allowed
        self.assertTrue(limiter.can_call())
        limiter.add_call()
        
        self.assertTrue(limiter.can_call())
        limiter.add_call()
        
        # Third call should be limited
        self.assertFalse(limiter.can_call())
        
        # Wait time should be positive
        self.assertGreater(limiter.wait_time(), 0)
    
    def test_sanitize_llm_output(self):
        """Test sanitizing LLM output"""
        # Test with code blocks
        test_output = """Here's some information:
        
        ```python
        def hello_world():
            print("Hello World")
        ```
        
        And some more text."""
        
        sanitized = sanitize_llm_output(test_output)
        
        # The code block markers should be removed but content preserved
        self.assertNotIn("```python", sanitized)
        self.assertNotIn("```", sanitized)
        self.assertIn("def hello_world():", sanitized)
        self.assertIn("And some more text.", sanitized)
    
    @patch('trial1.research_sub_query')
    def test_end_to_end_simple_query(self, mock_research):
        """Test end-to-end flow with a simple query"""
        # This is a higher-level test that would depend on your application structure
        # Here we're just mocking the research function
        
        mock_research.return_value = "Tokyo has a population of approximately 14 million in the city proper."
        
        # Run the process with a simple query
        result = research_sub_query("What is the population of Tokyo?")
        
        # Verify expected result
        self.assertIn("Tokyo", result)
        self.assertIn("14 million", result)
        
        # Verify research was called with the right query
        mock_research.assert_called_once()
    
    @patch('trial1.decompose_query')
    @patch('trial1.research_sub_query')
    def test_end_to_end_complex_query(self, mock_research, mock_decompose):
        """Test end-to-end flow with a complex query that gets decomposed"""
        # Set up our mocks
        mock_decompose.return_value = [
            "What are the features of the latest iPhone?",
            "What are the features of the latest Samsung Galaxy?"
        ]
        
        mock_research.side_effect = [
            "The latest iPhone features include A15 Bionic chip, improved cameras...",
            "The latest Samsung Galaxy features include Snapdragon processor, 108MP camera..."
        ]
        
        # In an actual test, you would call your main query handler here
        # For this example, we'll simulate the high-level flow
        
        query = "Compare the latest iPhone and Samsung Galaxy phones"
        
        # Simulate the main process flow
        if len(query.split()) > 5:  # Complex query
            sub_queries = decompose_query(query)
            results = []
            
            for sq in sub_queries:
                result = research_sub_query(sq)
                results.append(result)
            
            # In the actual code, you'd generate a summary here
            combined = "\n\n".join(results)
        else:
            combined = research_sub_query(query)
        
        # Verify the flow and results
        self.assertEqual(len(results), 2)
        self.assertIn("iPhone", combined)
        self.assertIn("Samsung", combined)
        
        # Verify our mocks were called correctly
        mock_decompose.assert_called_once_with(query)
        self.assertEqual(mock_research.call_count, 2)

def run_tests():
    """Run all tests"""
    logger.info("Starting Web Research Agent tests...")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    logger.info("Tests completed.")

if __name__ == "__main__":
    logger.info("Web Research Agent Test Suite")
    run_tests()