#  Web Research Agent Test Cases

This document contains test cases for the Web Research Agent, demonstrating its capabilities across different query types and scenarios. Each test case includes the input query, expected behavior, and validation criteria.

##  Table of Contents

1. [Simple Factual Queries](#simple-factual-queries)
2. [Multi-Part Complex Queries](#multi-part-complex-queries)
3. [News and Current Events](#news-and-current-events)
4. [Technical and Specialized Queries](#technical-and-specialized-queries)
5. [Opinion and Review Queries](#opinion-and-review-queries)
6. [Edge Cases and Error Handling](#edge-cases-and-error-handling)
7. [Running the Tests](#running-the-tests)

## 🔍 Simple Factual Queries

### Test Case 1: Basic Factual Question

**Input Query:**
```
What is the population of Tokyo in 2023?
```

**Expected Behavior:**
- Agent should not decompose this simple query
- Should trigger Web Search Tool
- May trigger Web Scraper on a relevant page 
- Should return a concise answer with the population figure
- Answer should include citation(s)

**Validation Criteria:**
- ✅ Returns a specific number for Tokyo's population
- ✅ Cites at least one credible source
- ✅ Completes within 20 seconds
- ✅ Tool usage shows search and possibly scraper

### Test Case 2: Historical Fact

**Input Query:**
```
When was the Eiffel Tower built and who designed it?
```

**Expected Behavior:**
- Agent may decompose into two related questions 
- Should identify query as historical type
- Should use the Web Search Tool with historical optimization
- May use the Web Scraper on an authoritative page
- Should return specific dates and the name of the designer

**Validation Criteria:**
- ✅ Returns the correct year (1887-1889)
- ✅ Identifies Gustave Eiffel as the designer
- ✅ Provides at least one relevant citation
- ✅ Query is detected as "historical" type

##  Multi-Part Complex Queries

### Test Case 3: Multi-aspect Research Question

**Input Query:**
```
What are the environmental impacts of electric vehicles compared to gas vehicles, considering manufacturing, usage, and disposal?
```

**Expected Behavior:**
- Agent should decompose this into at least 3 sub-questions
- Should address manufacturing, usage, and disposal aspects
- Should use multiple tools for comprehensive research
- Final summary should integrate insights from sub-questions
- Should include citations from diverse sources

**Validation Criteria:**
- ✅ Decomposes into 3+ related sub-questions
- ✅ Each aspect (manufacturing, usage, disposal) is addressed
- ✅ Provides comparative analysis
- ✅ Final summary integrates findings across sub-questions
- ✅ Multiple citations from varied sources

### Test Case 4: Cross-Domain Complex Query

**Input Query:**
```
How has machine learning impacted healthcare diagnostics, what are the ethical concerns, and what future developments are expected?
```

**Expected Behavior:**
- Agent should decompose into multiple sub-questions
- Should address technical, ethical, and future-looking aspects
- Should use Web Search, possibly News Tool for recent developments
- Final summary should synthesize across domains
- Should include citations from technical and ethical sources

**Validation Criteria:**
- ✅ Covers technical aspects of ML in healthcare
- ✅ Addresses ethical concerns
- ✅ Discusses future developments
- ✅ Final synthesis connects these different domains
- ✅ Sources include both technical and ethical perspectives

##  News and Current Events

### Test Case 5: Recent News Query

**Input Query:**
```
What are the latest developments in quantum computing?
```

**Expected Behavior:**
- Agent should identify this as a news-type query
- Should trigger the News Aggregator tool
- Should use search with news site optimization
- Should return recent developments with dates
- Should prioritize recent sources

**Validation Criteria:**
- ✅ Query is detected as "news" type
- ✅ News Aggregator tool is used
- ✅ Results include developments from recent months
- ✅ Sources include news and technology publications
- ✅ Information is presented chronologically or by importance

### Test Case 6: Evolving Situation Query

**Input Query:**
```
What is the current state of renewable energy adoption worldwide?
```

**Expected Behavior:**
- Agent should identify need for current information
- Should use News Aggregator for recent developments
- Should use Web Search for baseline statistics
- May use Scraper for detailed reports
- Final answer should indicate recency of information

**Validation Criteria:**
- ✅ Includes recent statistics and developments
- ✅ Distinguishes between established trends and recent changes
- ✅ Cites authoritative sources for statistics
- ✅ Uses news sources for recent developments
- ✅ Notes the time-sensitive nature of the information

## Technical and Specialized Queries

### Test Case 7: Programming/Technical Question

**Input Query:**
```
What are the key differences between React and Angular for front-end development?
```

**Expected Behavior:**
- Agent should identify as technical query
- Should use search with technical site optimization
- May use scraper on documentation or comparison pages
- Should structure response to highlight framework differences
- Should cite developer resources and documentation

**Validation Criteria:**
- ✅ Query is detected as "technical" type
- ✅ Response organizes differences into clear categories
- ✅ Technical accuracy of framework comparison
- ✅ Citations include developer resources
- ✅ Balanced presentation of both frameworks

### Test Case 8: Scientific Query

**Input Query:**
```
How do mRNA vaccines work and how do they differ from traditional vaccines?
```

**Expected Behavior:**
- Agent should decompose into mechanism and comparison
- Should use search and potentially scraper
- Should prioritize authoritative scientific sources
- Should explain complex concepts clearly
- Should provide balanced information from medical sources

**Validation Criteria:**
- ✅ Explains mRNA mechanism accurately
- ✅ Compares with traditional vaccine approaches
- ✅ Citations include medical/scientific sources
- ✅ Explains technical concepts in accessible language
- ✅ Information is scientifically accurate

##  Opinion and Review Queries

### Test Case 9: Product Comparison

**Input Query:**
```
What are the pros and cons of the latest iPhone compared to Samsung Galaxy phones?
```

**Expected Behavior:**
- Agent should identify as opinion query
- Should search with review/opinion optimization
- Should balance positive and negative aspects
- Should use recent sources for latest models
- Final summary should be balanced across brands

**Validation Criteria:**
- ✅ Query is detected as "opinion" type
- ✅ Lists pros and cons for both phones
- ✅ Uses comparative structure
- ✅ Cites multiple review sources
- ✅ Maintains neutrality between brands

### Test Case 10: Subjective Topic

**Input Query:**
```
What are considered the best science fiction novels of the past decade?
```

**Expected Behavior:**
- Agent should identify as opinion query
- Should include multiple perspectives
- Should reference lists, awards, and critic reviews
- Final answer should acknowledge subjectivity
- Should cite literary sources and reviews

**Validation Criteria:**
- ✅ Acknowledges subjective nature of "best"
- ✅ References multiple novels and authors
- ✅ Cites award information or critic reviews
- ✅ Provides context for recommendations
- ✅ Presents diverse perspectives

## ⚠️ Edge Cases and Error Handling

### Test Case 11: Potentially Overloading Query

**Input Query:**
```
Write a comprehensive analysis of global economic trends since 1900, including all major recessions, their causes, effects, and policy responses.
```

**Expected Behavior:**
- Agent should recognize excessive scope
- Should focus on key recessions rather than all events
- Should manage token limits appropriately
- Should decompose into manageable sub-questions
- Final summary should acknowledge the breadth of the topic

**Validation Criteria:**
- ✅ Avoids 413 errors through proper truncation
- ✅ Decomposes into manageable chunks
- ✅ Provides organized, structured response
- ✅ Handles large content appropriately
- ✅ Maintains coherence despite scope

### Test Case 12: Unreliable Information Domain

**Input Query:**
```
What are the most effective weight loss supplements?
```

**Expected Behavior:**
- Agent should prioritize medical and scientific sources
- Should distinguish between evidence-based and marketing claims
- Should note scientific consensus where applicable
- Should maintain appropriate caution with health claims
- Should cite authoritative health sources

**Validation Criteria:**
- ✅ Prioritizes scientific evidence
- ✅ Distinguishes between proven and unproven claims
- ✅ Cites medical/health authority sources
- ✅ Notes limitations in research where appropriate
- ✅ Maintains appropriate caution on health topics

## 🧪 Running the Tests

You can test these scenarios using the `test_agent.py` script, which automates running these queries through the agent and validates the results.

### Manual Testing

For manual testing:

1. Launch the Streamlit app:
```bash
streamlit run trial1.py
```

2. Enter each test query in the interface
3. Validate the results against the criteria
4. Check the debug panel for tool usage

### Automated Testing

The `test_agent.py` script provides automated testing:

```python
# test_agent.py

import unittest
from unittest.mock import patch
import os
import json
from dotenv import load_dotenv

# Import agent components
from trial1 import (
    research_sub_query,
    decompose_query, 
    tool_tracker,
    detect_query_type
)

class TestWebResearchAgent(unittest.TestCase):
    
    def setUp(self):
        # Reset tool tracker between tests
        tool_tracker.reset()
    
    def test_query_type_detection(self):
        # Test query type detection
        self.assertEqual(detect_query_type("What happened in World War 2?"), "historical")
        self.assertEqual(detect_query_type("What is the best smartphone?"), "opinion")
        self.assertEqual(detect_query_type("Latest developments in AI"), "news")
        
    def test_query_decomposition(self):
        # Test query decomposition
        complex_query = "What are the environmental impacts of electric vehicles compared to gas vehicles, considering manufacturing, usage, and disposal?"
        sub_queries = decompose_query(complex_query)
        
        # Verify we get at least 3 sub-questions
        self.assertGreaterEqual(len(sub_queries), 3)
        
        # Verify sub-queries are relevant
        keywords = ["environmental", "electric", "vehicle", "manufacturing", "usage", "disposal"]
        for sq in sub_queries:
            found = False
            for kw in keywords:
                if kw.lower() in sq.lower():
                    found = True
            self.assertTrue(found, f"Sub-query '{sq}' doesn't match expected content")
    
    def test_simple_factual_query(self):
        # Test a simple factual query
        result = research_sub_query("What is the population of Tokyo?")
        
        # Verify result contains a population figure
        self.assertTrue(any(char.isdigit() for char in result), "Result should contain population figures")
        
        # Verify search tool was used
        self.assertGreater(tool_tracker.usage["search"], 0, "Search tool should be used")
    
    # Additional tests for other scenarios...

if __name__ == '__main__':
    # Load environment variables
    load_dotenv()
    
    # Run tests
    unittest.main()
```

Run the automated tests with:
```bash
python test_agent.py
```

### Test Results Documentation

Document test results in a format like:

```
Test Case: Complex Query Decomposition
Input: "What are the environmental impacts of electric vehicles compared to gas vehicles, considering manufacturing, usage, and disposal?"
Result: PASS
- Decomposed into 4 sub-questions
- All aspects covered (manufacturing, usage, disposal, comparison)
- Final synthesis integrated findings
- Multiple relevant citations provided
```

This comprehensive testing approach ensures the agent handles a wide range of query types and edge cases effectively.