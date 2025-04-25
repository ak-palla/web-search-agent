#  Web Research Agent: Technical Documentation

This document provides a comprehensive explanation of how the Web Research Agent works, including its architecture, components, and information flow.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Key Components](#key-components)
3. [Information Flow](#information-flow)
4. [Core Functions](#core-functions)
5. [Tool Implementation](#tool-implementation)
6. [Agent Execution](#agent-execution)
7. [Query Processing](#query-processing)
8. [Error Handling](#error-handling)
9. [Rate Limiting](#rate-limiting)
10. [User Interface](#user-interface)
11. [Configuration Options](#configuration-options)
12. [References and Citations](#references-and-citations)

## Architecture Overview

The Web Research Agent is built on a modular architecture that integrates several key components:

1. **Agent Framework**: LangChain's agent framework for orchestrating tools
2. **LLM Backend**: Groq's Llama3 model for reasoning and text generation
3. **Tool Ecosystem**: Search, scraping, analysis, and news aggregation tools
4. **Memory System**: Conversation memory for contextual awareness
5. **UI Layer**: Streamlit interface for user interaction
6. **Tracking System**: Source tracking for citations and references

This architecture allows the agent to break down complex queries, delegate to specialized tools, and synthesize information from multiple sources.

```
┌─────────────────────────────────────────────────────────────┐
│                        Streamlit UI                         │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                    Query Processing                         │
│  ┌─────────────────┐    ┌─────────────────┐                 │
│  │ Query Analysis  │───►│Query Decomposer │                 │
│  └─────────────────┘    └─────────────────┘                 │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                     Agent Executor                          │
│  ┌─────────────┐  ┌────────────┐  ┌──────────────────┐      │
│  │    LLM      │◄─┤   Memory   │  │ Tool Selection   │      │
│  └─────┬───────┘  └────────────┘  └────────┬─────────┘      │
│        │                                   │                 │
│        └───────────────┬──────────────────┘                 │
└────────────────────────┬──────────────────────────────────┬─┘
                         │                                  │
┌───────────────────────▼─────────────────┐  ┌─────────────▼──────────────┐
│             Tool Ecosystem              │  │    Results Processing       │
│  ┌─────────┐ ┌────────┐ ┌────────────┐  │  │ ┌────────────┐ ┌─────────┐ │
│  │  Search │ │Scraper │ │  Analyzer  │  │  │ │ Synthesis  │ │Citations│ │
│  └─────────┘ └────────┘ └────────────┘  │  │ └────────────┘ └─────────┘ │
└───────────────────────────────────────┬─┘  └──────────────────────────┬─┘
                                        │                               │
                                        └───────────────┬───────────────┘
                                                       │
                                      ┌───────────────▼───────────────┐
                                      │     Final Response            │
                                      └───────────────────────────────┘
```

##  Key Components

### 1. Tool Usage Tracker
The `ToolUsageTracker` class manages:
- Tool usage statistics
- Source collection for citations
- Result tracking across sub-queries

```python
class ToolUsageTracker:
    def __init__(self):
        self.usage = {"search": 0, "scraper": 0, "analyzer": 0, "news": 0}
        self.sources = []  # Store sources for citation
    
    def track(self, tool_name):
        # Track tool usage
        
    def add_source(self, source):
        # Add source for citation
        
    def get_sources(self):
        # Return collected sources
```

### 2. Rate Limiter
The `RateLimiter` class prevents API overload:
- Tracks API call timestamps
- Enforces maximum call frequency
- Calculates wait times when rate limits are approached

```python
class RateLimiter:
    def __init__(self, max_calls=5, time_window=60):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    def can_call(self):
        # Check if call is allowed
        
    def wait_time(self):
        # Calculate wait time before next allowed call
```

### 3. Tools
The agent has four main tools:
- **Web Search Tool**: DuckDuckGo with semantic re-ranking
- **Web Scraper**: Firecrawl with BeautifulSoup fallback
- **Content Analyzer**: Text summarization and analysis
- **News Aggregator**: Specialized news search

Each tool is wrapped as a LangChain Tool object:
```python
search = Tool(
    name="Web Search Tool",
    func=lambda tool_input: duckduckgo_multi_page_search(tool_input),
    description="Performs a detailed web search with semantic re-ranking..."
)
```

### 4. Agent
The agent uses LangChain's `CHAT_CONVERSATIONAL_REACT_DESCRIPTION` agent type:
- Takes tools as input
- Uses the LLM for reasoning
- Has memory for conversation context
- Uses a custom prompt that encourages tool usage

##  Information Flow

1. **User Query Intake**:
   - User enters query through Streamlit UI
   - Query is analyzed to determine complexity

2. **Query Decomposition** (for complex queries):
   - LLM breaks the query into 3-7 sub-questions
   - Each sub-question is validated and normalized

3. **Tool Selection and Execution**:
   - For each sub-question, the agent decides which tools to use
   - Tools are executed with rate limiting and error handling
   - Results are collected and sources are tracked

4. **Result Synthesis**:
   - Sub-question results are combined
   - LLM generates a final summary
   - Citations are added to the result

5. **Response Presentation**:
   - Results are displayed in the Streamlit UI
   - Chat history is updated
   - Debug information is available

## 🛠️ Core Functions

### Query Type Detection
```python
def detect_query_type(query: str) -> str:
    """Detects query type (historical, opinion, news, factual, technical)"""
```
This function analyzes the query to optimize search strategies.

### Query Decomposition
```python
def decompose_query(query: str) -> list:
    """Breaks complex queries into sub-questions"""
```
Used for multi-faceted research questions, improving answer comprehensiveness.

### Sub-Query Validation
```python
def validate_sub_query(sub_query):
    """Validates and fixes formatting issues in sub-questions"""
```
Ensures proper formatting and prevents malformed sub-queries.

### Citation Management
```python
def add_citations_to_summary(summary, sources):
    """Adds source citations to the final summary"""
```
Improves credibility by linking information to sources.

## 🔧 Tool Implementation

### Web Search
The search tool combines:
1. Query type optimization (adds site qualifiers based on query type)
2. Multi-page search for broader coverage
3. Semantic re-ranking using embeddings for relevance
4. Source tracking for citations

Key implementation details:
```python
def duckduckgo_multi_page_search(tool_input: str, pages: int = 3) -> str:
    # 1. Detect query type and optimize search query
    qtype = detect_query_type(query)
    if qtype == "news":
        query += " site:news.google.com OR site:reuters.com OR site:bloomberg.com"
    
    # 2. Search multiple pages
    for i in range(pages):
        search_results = ddg.run(tool_input=query)
        
    # 3. Semantic re-ranking
    query_embedding = embeddings.embed_query(query)
    for r in results:
        content = f"{title} {snippet}"
        result_embedding = embeddings.embed_query(content)
        score = cosine_similarity(query_embedding, result_embedding)
        
    # 4. Track sources
    for _, result in top_results[:5]:
        tool_tracker.add_source(result)
```

### Web Scraper
The scraper includes:
1. Robots.txt compliance checking
2. Primary scraper using Firecrawl
3. Fallback to requests + BeautifulSoup
4. Table extraction for structured data
5. Large content processing

Key implementation:
```python
def web_scraper(tool_input: str) -> str:
    # 1. Check robots.txt
    if not is_scraping_allowed(url):
        return f"Scraping disallowed by robots.txt for {url}"
        
    # 2. Try primary scraper
    try:
        result = firecrawl.scrape_url(url=url)
        # ...
    except Exception:
        # 3. Fallback scraper
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        # ...
        
    # 4. Extract tables
    tables = extract_tables_from_html(html)
    
    # 5. Process large content
    if content and len(content) > 5000:
        content = process_large_content(content)
```

### Content Analyzer
Processes raw content into digestible summaries:
```python
def content_analyzer(tool_input: str) -> str:
    # Truncate input to prevent token limit errors
    truncated_input = truncate_text(tool_input, max_tokens=3000)
    
    # Generate summary
    prompt = f"""Analyze and summarize the following content in 3-5 key points:
    
    {truncated_input}
    
    Summary:"""
    
    summary = llm.predict(prompt)
```

### News Aggregator
Enhanced news search with:
1. Key term extraction for better relevance
2. Multi-source search focused on news domains
3. Source tracking for citations

```python
def news_search(tool_input: str) -> str:
    # Extract key terms
    key_terms_prompt = f"""Extract 3-5 key technical terms from this query..."""
    key_terms = llm.predict(key_terms_prompt).strip()
    
    # Construct enhanced query
    query = f"latest research {tool_input} {key_terms} site:news.google.com OR..."
```

## 🤖 Agent Execution

The agent execution process is managed by:

1. **Tool Selection Strategy**:
```python
def should_use_tools(query: str) -> bool:
    """Determines if external tools should be used"""
```

2. **Research Sub-Query Function**:
```python
def research_sub_query(sub_query: str) -> str:
    """Researches a single sub-query with tools"""
    # Validate sub-query
    sub_query = validate_sub_query(sub_query)
    
    # Modify for tool usage
    if should_use_tools(sub_query):
        research_query = f"Research thoroughly using available tools: {sub_query}"
        
    # Rate limiting
    wait_time = api_limiter.wait_time()
    if wait_time > 0:
        time.sleep(wait_time)
        
    # Execute with retry
    for attempt in range(max_retries):
        try:
            result = agent_executor.invoke({"input": limited_query})
            # ...
        except Exception as e:
            if "413" in str(e):
                # Reduce content size for 413 errors
                limited_query = truncate_text(limited_query, max_tokens=int(len(limited_query) * 0.5 / 4))
```

3. **Final Synthesis**:
```python
summary_prompt = (
    "You are a research assistant. Below are answers to several sub-questions..."
    + "\n\n".join(processed_responses)
    + "\n\nPlease synthesize the key insights..."
)

final_summary = llm.predict(summary_prompt)
final_summary_with_citations = add_citations_to_summary(final_summary, tool_tracker.get_sources())
```

## 🧠 Query Processing

Query processing involves:

1. **Complexity Analysis**: Determining if a query is complex enough to decompose
```python
if use_decomposition and len(query.split()) > 5:
    sub_queries = decompose_query(query)
```

2. **Decomposition**: Breaking complex queries into sub-questions
```python
def decompose_query(query: str) -> list:
    system_prompt = (
        "You are a research assistant specialized in breaking down complex questions..."
    )
    prompt = f"{system_prompt}\n\nResearch Question:\n{limited_query}\n\nDecomposed Sub-Questions:"
    response = llm.predict(prompt)
    
    # Extract sub-questions
    sub_questions = []
    for line in response.strip().split('\n'):
        # ... extraction logic ...
```

3. **Validation**: Ensuring each sub-question is properly formatted
```python
def validate_sub_query(sub_query):
    if not sub_query or len(sub_query.strip()) < 10:
        return "General information about this topic"
    
    if ":" in sub_query:
        parts = sub_query.split(":", 1)
        if len(parts) > 1:
            return parts[0].strip() + ": " + parts[1].strip()
```

## ⚠️ Error Handling

The agent implements several error handling strategies:

1. **Token Limit Management**: To prevent 413 Payload Too Large errors
```python
def truncate_text(text, max_tokens=4000):
    # Add safety margin for API calls
    if max_tokens > 1000:
        char_limit = int(char_limit * 0.9)  # 10% safety margin
```

2. **Exponential Backoff**: For rate limit errors
```python
if ("429" in str(e) or "413" in str(e)) and attempt < max_retries - 1:
    retry_delay *= 2  # Exponential backoff
    logger.warning(f"API error ({e}), retrying in {retry_delay} seconds...")
    time.sleep(retry_delay)
```

3. **Graceful Degradation**: Fallback mechanisms when primary methods fail
```python
# Fallback to requests + BeautifulSoup if Firecrawl fails
try:
    result = firecrawl.scrape_url(url=url)
except Exception as firecrawl_error:
    # Fallback scraping with requests + BeautifulSoup
```

4. **Tool Usage Verification**: Forces tool usage if none were used
```python
if should_use_tools(sub_query) and all(v == 0 for v in tool_tracker.usage.values()):
    # Force tool usage
    forced_query = f"Use the Web Search Tool to find updated information about: {truncate_text(sub_query, max_tokens=1000)}"
```

## Rate Limiting

Rate limiting is implemented through the `RateLimiter` class:

```python
class RateLimiter:
    def __init__(self, max_calls=5, time_window=60):
        self.max_calls = max_calls  # Max calls allowed in time window
        self.time_window = time_window  # Time window in seconds
        self.calls = []  # List of timestamps of calls
    
    def can_call(self):
        now = time.time()
        # Remove calls outside time window
        self.calls = [call for call in self.calls if call > now - self.time_window]
        # If we haven't reached max calls, allow call
        return len(self.calls) < self.max_calls
    
    def add_call(self):
        self.calls.append(time.time())
    
    def wait_time(self):
        if self.can_call():
            return 0
        # Return time until oldest call expires
        return self.calls[0] + self.time_window - time.time()
```

Usage in the agent:
```python
# Check API rate limits
wait_time = api_limiter.wait_time()
if wait_time > 0:
    logger.info(f"Rate limit approaching, waiting {wait_time:.2f} seconds")
    time.sleep(wait_time)

# Add call to rate limiter
api_limiter.add_call()
```

##  User Interface

The Streamlit UI includes:

1. **Main Query Input**:
```python
query = st.text_input("", placeholder="e.g., What are the latest AI breakthroughs in healthcare?")
```

2. **Settings Panel**:
```python
with st.sidebar:
    st.header("⚙️ Settings")
    use_decomposition = st.checkbox("Decompose complex queries", value=True)
    search_depth = st.slider("Search depth", min_value=1, max_value=5, value=3)
```

3. **Research Progress Display**:
```python
progress_bar = st.progress(0)
progress_text = st.empty()

for i, sub_q in enumerate(sub_queries, start=1):
    progress_text.text(f"Researching sub-question {i}/{len(sub_queries)}: {validated_sub_q}")
    # ...
    progress_bar.progress(i / len(sub_queries))
```

4. **Chat History**:
```python
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("💬 Research History")
    for user_msg, bot_msg in st.session_state.chat_history:
        st.markdown(f"""
        <div style='margin-bottom: 1rem;'>
            <div style='background-color: #DCF8C6; padding: 0.7rem; border-radius: 10px;'>
                <strong>🧍‍♂️ You:</strong><br>{user_msg}
            </div>
            <div style='background-color: #E6E6FA; padding: 0.7rem; border-radius: 10px; margin-top: 5px;'>
                <strong>🤖 Agent:</strong><br>{bot_msg}
            </div>
        </div>
        """, unsafe_allow_html=True)
```

5. **Debug Information**:
```python
with st.expander("📑 Debug Information"):
    st.subheader("Tool Usage")
    st.json(tool_tracker.usage)
    
    st.subheader("Sources Collected")
    source_list = tool_tracker.get_sources()
    if source_list:
        for i, source in enumerate(source_list, 1):
            st.markdown(f"{i}. [{source.get('title', 'Untitled')}]({source.get('link', '#')})")
```

## ⚙️ Configuration Options

The agent can be configured through several parameters:

1. **Query Decomposition**: Enable/disable breaking complex queries into sub-questions
```python
use_decomposition = st.checkbox("Decompose complex queries", value=True)
```

2. **Search Depth**: Number of search pages to analyze
```python
search_depth = st.slider("Search depth", min_value=1, max_value=5, value=3)
```

3. **Rate Limiting**: Adjust API call frequency
```python
api_limiter = RateLimiter(max_calls=3, time_window=60)
```

4. **Token Limits**: Adjust truncation thresholds for different contexts
```python
truncated_input = truncate_text(tool_input, max_tokens=3000)
```

5. **Agent Prompt**: Customize the agent's behavior
```python
custom_prompt = """You are an advanced web research agent that helps users find accurate information..."""
```

##  References and Citations

The citation system works by:

1. **Source Collection**: Each tool adds sources to a central tracker
```python
def add_source(self, source):
    """Add a source for later citation"""
    if source and isinstance(source, dict):
        if source not in self.sources:
            self.sources.append(source)
```

2. **Source Extraction**: Search results are parsed to extract citation information
```python
def extract_search_sources(search_results):
    """Extract source information from search results for citation"""
    sources = []
    
    if isinstance(search_results, str):
        # Try to parse links and titles from formatted search results
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        matches = re.findall(link_pattern, search_results)
        
        for title, link in matches:
            sources.append({
                'title': title,
                'link': link,
                'snippet': ''
            })
```

3. **Citation Addition**: Citations are added to the final summary
```python
def add_citations_to_summary(summary, sources):
    """Add source citations to the final summary"""
    if not sources:
        return summary
        
    # Add citations section
    citation_text = "\n\n**Sources:**\n"
    for i, source in enumerate(sources, 1):
        if source.get('title') and source.get('link'):
            citation_text += f"{i}. [{source['title']}]({source['link']})\n"
    
    return summary + citation_text
```

This comprehensive citation system ensures that information is properly attributed and verifiable.
