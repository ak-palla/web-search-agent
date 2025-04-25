# Web Research Agent: Error Handling & Problem Resolution

This document outlines the comprehensive approach to handling errors and resolving problems in the Web Research Agent.

## 1. Network and API Issues

### Rate Limiting and Throttling
```python
class RateLimiter:
    def __init__(self, max_calls=5, time_window=60):
        self.max_calls = max_calls  # Maximum calls allowed in time window
        self.time_window = time_window  # Time window in seconds
        self.calls = []  # List of timestamps of calls
```

- **Detection**: The agent monitors API call frequency
- **Response**: When approaching rate limits:
  1. Calculates necessary wait time
  2. Pauses execution: `time.sleep(wait_time)`
  3. Continues after the waiting period
- **Prevention**: Implements exponential backoff for retries

### HTTP Errors (429, 413, etc.)
```python
for attempt in range(max_retries):
    try:
        result = agent_executor.invoke({"input": limited_query})
        break
    except Exception as e:
        if ("429" in str(e) or "413" in str(e)) and attempt < max_retries - 1:
            # If it's a 413 Payload Too Large error, reduce the query size
            if "413" in str(e):
                limited_query = truncate_text(limited_query, max_tokens=int(len(limited_query) * 0.5 / 4))
            
            retry_delay *= 2  # Exponential backoff
            time.sleep(retry_delay)
        else:
            raise e
```

- **413 (Payload Too Large)**: Automatically reduces content size by 50% and retries
- **429 (Too Many Requests)**: Implements exponential backoff between retries
- **Other HTTP errors**: Logs the error and provides a fallback response

## 2. Content Issues

### Excessively Large Content
```python
def truncate_text(text, max_tokens=4000):
    # More accurate token estimation: ~4 chars per token for English text
    char_limit = max_tokens * 4
    
    # Add a safety margin for API calls to prevent 413 errors
    if max_tokens > 1000:
        char_limit = int(char_limit * 0.9)  # 10% safety margin
```

- **Detection**: Checks content length against token limits
- **Response**: Truncates with a safety margin and adds "[content truncated due to length]" notice
- **Alternative Processing**: For large web pages, splits and processes in chunks:
  ```python
  def process_large_content(content, max_chunk_size=3000):
      if len(content) < max_chunk_size:
          return content
          
      chunks = text_splitter.split_text(content)
      # Process each chunk and combine results
  ```

### Inaccessible Websites
```python
def web_scraper(tool_input: str) -> str:
    try:
        # First attempt with Firecrawl
        result = firecrawl.scrape_url(url=url)
        # ...
    except Exception as firecrawl_error:
        logger.warning(f"Firecrawl failed: {firecrawl_error}. Falling back to requests + BeautifulSoup")
        # Fallback to requests + BeautifulSoup
```

- **Primary Method**: Uses Firecrawl for web scraping
- **Fallback Mechanism**: If Firecrawl fails, automatically switches to requests + BeautifulSoup
- **Permission Check**: Respects robots.txt with `is_scraping_allowed(url)` function
- **Graceful Degradation**: If all scraping methods fail, returns error message and continues research process

## 3. Information Quality Issues

### Contradictory Information
```python
def contradiction_resolver(content1: str, content2: str) -> str:
    prompt = f"""Compare the following two pieces of content and identify any contradictions:
    
    Content 1: {content1}
    
    Content 2: {content2}
    
    Are there any contradictions? If so, resolve them by:
    1. Identifying the specific contradicting claims
    2. Comparing the reliability of the sources
    3. Providing a balanced conclusion that acknowledges the contradiction
    """
```

- **Detection**: When collecting information from multiple sources, identifies potential contradictions
- **Resolution**: Uses the LLM to analyze conflicting information and:
  1. Identifies specific contradictions
  2. Evaluates source reliability
  3. Provides balanced conclusions
- **Presentation**: Acknowledges contradictions in the final response, presenting multiple perspectives

### Insufficient or Low-Quality Results
```python
# If no tools were used, try once more with explicit instruction
if should_use_tools(sub_query) and all(v == 0 for v in tool_tracker.usage.values()):
    logger.warning("No tools were used. Retrying with explicit instruction.")
    try:
        # Force tool usage
        forced_query = f"Use the Web Search Tool to find updated information about: {truncate_text(sub_query, max_tokens=1000)}"
        result = agent_executor.invoke({"input": forced_query})
        response = result.get("output", result)
    except Exception as e:
        logger.error(f"Error in forced tool usage: {e}")
```

- **Detection**: Monitors tool usage and detects when no tools were used
- **Response**: Forces explicit tool usage with direct instructions
- **Fallback**: If information is still insufficient, notes limitations in the response
- **Alternative Sources**: Tries different search optimizations (e.g., adds "site:" directives)

## 4. LLM and Processing Issues

### LLM Output Parsing Errors
```python
def sanitize_llm_output(raw_output):
    """Remove markdown code blocks from LLM output to prevent parsing errors"""
    if isinstance(raw_output, str):
        # Remove triple backtick code blocks
        cleaned = re.sub(r'```(?:json|python)?\s*(.*?)```', r'\1', raw_output, flags=re.DOTALL)
        return cleaned
    return raw_output
```

- **Prevention**: Sanitizes LLM outputs to remove problematic formatting
- **Error Handling**: Uses regex patterns to clean outputs
- **Agent Framework**: Leverages LangChain's `handle_parsing_errors=True` parameter

### Query Decomposition Failures
```python
def decompose_query(query: str) -> list:
    # ... decomposition logic ...
    
    if not sub_questions:
        # Fallback: try to extract any sentences that end with question marks
        import re
        question_pattern = r'[^.!?]*\?'
        sub_questions = re.findall(question_pattern, response)
        sub_questions = [q.strip() for q in sub_questions if q.strip()]
    
    # If still no sub-questions, return original query
    return sub_questions if sub_questions else [query]
```

- **Detection**: Checks if decomposition failed to generate sub-questions
- **Fallback 1**: Attempts to extract questions using regex patterns
- **Fallback 2**: If no sub-questions can be extracted, proceeds with the original query

## 5. Integration and User Experience

### Progress Tracking and Feedback
```python
# Create progress bar
progress_bar = st.progress(0)
progress_text = st.empty()

# Update during research
for i, sub_q in enumerate(sub_queries, start=1):
    progress_text.text(f"Researching sub-question {i}/{len(sub_queries)}: {validated_sub_q}")
    # ... research process ...
    progress_bar.progress(i / len(sub_queries))
```

- **Transparency**: Shows clear progress indicators during research
- **Visibility**: Displays which sub-question is currently being researched
- **Completion Indication**: Provides success message upon completion

### Debug Information
```python
with st.expander("📑 Debug Information"):
    st.subheader("Tool Usage")
    st.json(tool_tracker.usage)
    
    st.subheader("Sources Collected")
    source_list = tool_tracker.get_sources()
    # ... source display ...
```

- **Tool Usage Tracking**: Shows which tools were used and how many times
- **Source Tracking**: Displays all sources collected during research
- **System Information**: Provides technical details about the agent configuration
