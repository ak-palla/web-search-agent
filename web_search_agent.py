# Web Research Agent - Improved Version

import os
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import LangChainTracer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from firecrawl import FirecrawlApp
import urllib.robotparser
import requests
from urllib.parse import urlparse
import numpy as np
from bs4 import BeautifulSoup
import logging
import time
import re
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optional tokenizer
try:
    from nltk.tokenize import sent_tokenize
except:
    sent_tokenize = lambda text: text.split('.')

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Web Research Agent"

# Initialize LLM and Embeddings
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
embeddings = OpenAIEmbeddings()

# --- Helper Functions ---
def sanitize_llm_output(raw_output):
    """Remove markdown code blocks from LLM output to prevent parsing errors"""
    if isinstance(raw_output, str):
        # Remove triple backtick code blocks
        cleaned = re.sub(r'```(?:json|python)?\s*(.*?)```', r'\1', raw_output, flags=re.DOTALL)
        return cleaned
    return raw_output

def truncate_text(text, max_tokens=4000):
    """Truncate text to prevent exceeding token limits with improved estimation"""
    if not text:
        return ""
    
    # More accurate token estimation: ~4 chars per token for English text
    char_limit = max_tokens * 4
    
    # Add a safety margin for API calls to prevent 413 errors
    if max_tokens > 1000:
        char_limit = int(char_limit * 0.9)  # 10% safety margin
        
    if len(text) > char_limit:
        logger.warning(f"Text truncated from {len(text)} to {char_limit} characters")
        return text[:char_limit] + "... [content truncated due to length]"
    return text

def process_large_content(content, max_chunk_size=3000):
    """Process large content by splitting and summarizing in chunks"""
    if len(content) < max_chunk_size:
        return content
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_text(content)
    logger.info(f"Split large content into {len(chunks)} chunks")
    
    summaries = []
    for i, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {i+1}/{len(chunks)}")
        try:
            # We can't use content_analyzer directly here because of circular reference
            # So we do a simplified version
            prompt = f"""Summarize the following content in 3-5 key points:
            
            {chunk}
            
            Summary:"""
            
            chunk_summary = llm.predict(prompt)
            summaries.append(chunk_summary)
            # Rate limiting between chunk processing
            time.sleep(1)
        except Exception as e:
            logger.error(f"Error processing chunk {i+1}: {e}")
            summaries.append(f"[Processing error for this section: {e}]")
    
    combined = "\n\n".join(summaries)
    return combined

def validate_sub_query(sub_query):
    """Validate and fix formatting issues in sub-queries"""
    # Check if sub-query appears to be truncated or malformed
    if not sub_query or len(sub_query.strip()) < 10:
        return "General information about this topic"
    
    # Fix issues with colons in sub-query formatting
    if ":" in sub_query:
        parts = sub_query.split(":", 1)
        if len(parts) > 1:
            return parts[0].strip() + ": " + parts[1].strip()
    
    return sub_query.strip()

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

# --- Tool Usage Tracker ---
class ToolUsageTracker:
    def __init__(self):
        self.usage = {"search": 0, "scraper": 0, "analyzer": 0, "news": 0}
        self.sources = []  # Store sources for citation
    
    def track(self, tool_name):
        if tool_name in self.usage:
            self.usage[tool_name] += 1
            logger.info(f"Tool used: {tool_name}. Total usage: {self.usage[tool_name]}")
        return True
    
    def add_source(self, source):
        """Add a source for later citation"""
        if source and isinstance(source, dict):
            if source not in self.sources:
                self.sources.append(source)
                logger.info(f"Added source: {source.get('title', 'Untitled')}")
    
    def get_report(self):
        return self.usage
    
    def get_sources(self):
        return self.sources
    
    def reset(self):
        self.usage = {"search": 0, "scraper": 0, "analyzer": 0, "news": 0}
        self.sources = []

# Initialize a single global instance
tool_tracker = ToolUsageTracker()

# --- Query Type Detection ---
def detect_query_type(query: str) -> str:
    """Detect the type of query to optimize search strategy."""
    prompt = f"""Analyze this query and determine its primary type from the following options:
    - historical: About past events, trends, or historical information
    - opinion: Seeking reviews, perspectives, or subjective information
    - news: About recent events or current information
    - factual: Seeking objective, verifiable information
    - technical: Seeking detailed technical knowledge or specifications
    
    Query: {query}
    
    Respond with ONLY ONE of the type names (lowercase, no punctuation).
    """
    try:
        # Limit input to prevent token issues
        limited_query = truncate_text(query, max_tokens=1000)
        result = llm.predict(prompt).strip().lower()
        logger.info(f"Query type detected: {result}")
        return result
    except Exception as e:
        logger.error(f"Error detecting query type: {e}")
        # Default fallback
        lowered = query.lower()
        if "history" in lowered or "in the past" in lowered:
            return "historical"
        elif "opinion" in lowered or "review" in lowered:
            return "opinion"
        elif "news" in lowered or "latest" in lowered:
            return "news"
        return "factual"

# --- Extract Search Results ---
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
                'snippet': ''  # We don't have snippets in this format
            })
    
    return sources

# --- Cosine Similarity ---
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# --- DuckDuckGo Search with tool_input fix ---
def duckduckgo_multi_page_search(tool_input: str, pages: int = 3) -> str:
    """Perform a multi-page search with semantic re-ranking."""
    try:
        tool_tracker.track("search")
        query = tool_input
        qtype = detect_query_type(query)
        
        # Optimize search based on query type
        if qtype == "news":
            query += " site:news.google.com OR site:reuters.com OR site:bloomberg.com"
        elif qtype == "historical":
            query += " before:2022"
        elif qtype == "opinion":
            query += " review OR opinion OR perspective"
        elif qtype == "technical":
            query += " site:github.com OR site:stackoverflow.com OR site:medium.com"

        logger.info(f"Searching for: {query}")
        ddg = DuckDuckGoSearchResults()
        results = []
        
        for i in range(pages):
            logger.info(f"Searching page {i+1}")
            search_results = ddg.run(tool_input=query)
            
            # Handle different return types
            if isinstance(search_results, list):
                results.extend(search_results)
            elif isinstance(search_results, str) and search_results.strip():
                # Try to parse as JSON string if it's not already a list
                try:
                    parsed = json.loads(search_results)
                    if isinstance(parsed, list):
                        results.extend(parsed)
                    else:
                        results.append({"title": "Search Result", "link": "", "snippet": search_results})
                except:
                    results.append({"title": "Search Result", "link": "", "snippet": search_results})

        if not results:
            logger.warning("No search results found")
            return "No results found."

        # Semantic re-ranking
        query_embedding = embeddings.embed_query(query)
        scored_results = []
        
        for r in results:
            # Handle potential missing keys
            title = r.get('title', '')
            snippet = r.get('snippet', '')
            link = r.get('link', '')
            
            content = f"{title} {snippet}"
            result_embedding = embeddings.embed_query(content)
            score = cosine_similarity(query_embedding, result_embedding)
            scored_results.append((score, {"title": title, "link": link, "snippet": snippet}))

        top_results = sorted(scored_results, key=lambda x: x[0], reverse=True)
        logger.info(f"Found {len(top_results)} results, ranked by relevance")

        # Add sources to tracker for citation
        for _, result in top_results[:5]:
            tool_tracker.add_source(result)

        formatted = "\n\n".join([f"🔗 [{r['title']}]({r['link']})\n{r['snippet']}" for _, r in top_results[:5]])
        return formatted
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        return f"Search failed: {e}"

search = Tool(
    name="Web Search Tool",
    func=lambda tool_input: duckduckgo_multi_page_search(tool_input),
    description="Performs a detailed web search with semantic re-ranking. Use for finding current information, facts, news, and context."
)

# --- Firecrawl Web Scraper (FIXED) ---
firecrawl = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

def is_scraping_allowed(url: str) -> bool:
    """Check if scraping is allowed by robots.txt."""
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch("*", url)
    except:
        return True

def extract_tables_from_html(html: str) -> str:
    """Extract tables from HTML content."""
    soup = BeautifulSoup(html, 'html.parser')
    tables = soup.find_all('table')
    if not tables:
        return "No tables found."

    extracted = []
    for table in tables:
        rows = table.find_all('tr')
        for row in rows:
            cols = row.find_all(['td', 'th'])
            row_text = " | ".join(col.get_text(strip=True) for col in cols)
            extracted.append(row_text)
        extracted.append("\n")
    return "\n".join(extracted)

def extract_source_from_url(url: str) -> dict:
    """Extract source information from a URL for citation"""
    try:
        parsed = urlparse(url)
        title = parsed.netloc
        if parsed.path and parsed.path != '/':
            path_parts = parsed.path.split('/')
            # Use the last meaningful part of the path as title
            for part in reversed(path_parts):
                if part and len(part) > 3:
                    # Convert hyphens/underscores to spaces and capitalize
                    cleaned = part.replace('-', ' ').replace('_', ' ').capitalize()
                    title = f"{parsed.netloc}: {cleaned}"
                    break
        
        return {
            'title': title,
            'link': url,
            'snippet': f"Content from {parsed.netloc}"
        }
    except:
        return {'title': url, 'link': url, 'snippet': ''}

def web_scraper(tool_input: str) -> str:
    """Scrape content from a webpage while respecting robots.txt."""
    try:
        tool_tracker.track("scraper")
        url = tool_input
        
        if not url.startswith(('http://', 'https://')):
            return "Please provide a valid URL starting with http:// or https://"
            
        if not is_scraping_allowed(url):
            logger.warning(f"Scraping disallowed by robots.txt for {url}")
            return f"Scraping disallowed by robots.txt for {url}"

        logger.info(f"Scraping URL: {url}")
        
        try:
            # First attempt with Firecrawl
            result = firecrawl.scrape_url(url=url)
            content = result.get("content", "")
            html = result.get("html", "")
            title = result.get("title", "")
            
            # Add source for citation
            source = {
                'title': title or extract_source_from_url(url)['title'],
                'link': url,
                'snippet': content[:150] + "..." if content and len(content) > 150 else ""
            }
            tool_tracker.add_source(source)
            
        except Exception as firecrawl_error:
            logger.warning(f"Firecrawl failed: {firecrawl_error}. Falling back to requests + BeautifulSoup")
            # Fallback to requests + BeautifulSoup if Firecrawl fails
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Extract title for citation
                    title_tag = soup.find('title')
                    title = title_tag.get_text() if title_tag else extract_source_from_url(url)['title']
                    
                    # Add source for citation
                    source = {
                        'title': title,
                        'link': url,
                        'snippet': ""
                    }
                    tool_tracker.add_source(source)
                    
                    # Remove unwanted elements
                    for element in soup(['script', 'style', 'nav', 'footer', 'iframe']):
                        element.decompose()
                    
                    # Extract main content
                    main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content') or soup.body
                    
                    if main_content:
                        content = main_content.get_text(separator='\n', strip=True)
                    else:
                        content = soup.get_text(separator='\n', strip=True)
                    
                    # Clean up content
                    content = '\n'.join(line.strip() for line in content.split('\n') if line.strip())
                    html = str(soup)
                else:
                    return f"Failed to scrape URL: HTTP status code {response.status_code}"
            except Exception as requests_error:
                logger.error(f"Fallback scraping also failed: {requests_error}")
                return f"Scraping failed: {requests_error}"
        
        tables = extract_tables_from_html(html)

        if not content:
            logger.warning("No content found in scraping result")
            
        # Process content if it's too large
        if content and len(content) > 5000:
            content = process_large_content(content)
            
        combined = f"[Source]({url})\n\n{content}\n\n**Extracted Tables:**\n{tables}" if content else "No content found."
        logger.info(f"Successfully scraped URL: {url}")
        return combined
    except Exception as e:
        logger.error(f"Scraping failed for {url}: {e}")
        return f"Scraping failed for {url}: {e}"

scraper_tool = Tool(
    name="Web Scraper",
    func=web_scraper,
    description="Extracts detailed content from a specific webpage. Input should be a URL. Use this to get the full content of an article or webpage after finding it with the Web Search Tool."
)

# --- Content Analyzer ---
def content_analyzer(tool_input: str) -> str:
    """Analyze and summarize content."""
    try:
        tool_tracker.track("analyzer")
        
        if len(tool_input) < 100:
            return "Input text too short for meaningful analysis."
        
        # Truncate input to prevent token limit errors
        truncated_input = truncate_text(tool_input, max_tokens=3000)
        
        prompt = f"""Analyze and summarize the following content in 3-5 key points:
        
        {truncated_input}
        
        Summary:"""
        
        summary = llm.predict(prompt)
        logger.info("Content analysis completed successfully")
        return summary
    except Exception as e:
        logger.error(f"Content analysis failed: {e}")
        return f"Analysis failed: {e}"

analyzer_tool = Tool(
    name="Content Analyzer",
    func=content_analyzer,
    description="Analyzes raw content and returns a summary with key points. Use this after scraping to distill important information."
)

# --- News Aggregator (IMPROVED) ---
def news_search(tool_input: str) -> str:
    """Enhanced news search with better keyword extraction."""
    try:
        tool_tracker.track("news")
        
        # Extract key terms for better search
        key_terms_prompt = f"""Extract 3-5 key technical terms from this query that would be useful for finding specific news articles:
        
        Query: {tool_input}
        
        Return ONLY the terms separated by OR, with no additional text or explanation."""
        
        # Limit input size for key terms extraction
        limited_input = truncate_text(tool_input, max_tokens=1000)
        
        # Get key terms to enhance the search
        try:
            key_terms = llm.predict(key_terms_prompt).strip()
            # Clean up the response if it contains extra text
            if "OR" in key_terms:
                # Keep only the terms with OR separators
                terms_parts = key_terms.split("OR")
                key_terms = " OR ".join([term.strip() for term in terms_parts if term.strip()])
            logger.info(f"Extracted key terms: {key_terms}")
        except Exception as e:
            logger.warning(f"Error extracting key terms: {e}")
            key_terms = ""
        
        # Construct enhanced query with domain-specific sites
        query = f"latest research {tool_input} {key_terms} site:news.google.com OR site:reuters.com OR site:theverge.com OR site:techcrunch.com OR site:nature.com"
        
        logger.info(f"Enhanced news search for: {query}")
        results = duckduckgo_multi_page_search(query, pages=2)
        
        # Extract and add sources for citation
        sources = extract_search_sources(results)
        for source in sources:
            tool_tracker.add_source(source)
            
        return results
    except Exception as e:
        logger.error(f"News search failed: {e}")
        return f"News search failed: {e}"

news_tool = Tool(
    name="News Aggregator",
    func=news_search,
    description="Searches for the latest news on a topic from reliable sources. Use this for current events and recent developments."
)

# --- Contradiction Resolver ---
def contradiction_resolver(content1: str, content2: str) -> str:
    """Identify and resolve contradictions between content."""
    try:
        # Truncate inputs to prevent token issues
        content1 = truncate_text(content1, max_tokens=2000)
        content2 = truncate_text(content2, max_tokens=2000)
        
        prompt = f"""Compare the following two pieces of content and identify any contradictions:
        
        Content 1: {content1}
        
        Content 2: {content2}
        
        Are there any contradictions? If so, resolve them by:
        1. Identifying the specific contradicting claims
        2. Comparing the reliability of the sources
        3. Providing a balanced conclusion that acknowledges the contradiction
        
        Analysis:"""
        
        return llm.predict(prompt)
    except Exception as e:
        logger.error(f"Contradiction resolution failed: {e}")
        return f"Contradiction analysis failed: {e}"

# --- Query Decomposer ---
def decompose_query(query: str) -> list:
    """Break down a complex query into simpler sub-questions."""
    try:
        logger.info(f"Decomposing query: {query}")
        system_prompt = (
            "You are a research assistant specialized in breaking down complex questions. "
            "Given a user's research question, return 3-7 specific sub-questions that will help "
            "comprehensively answer the original question. Format as a numbered list."
            "Each sub-question should cover a distinct aspect of the main question."
            "The sub-questions should be specific enough that they can be directly researched."
        )
        
        # Limit input size
        limited_query = truncate_text(query, max_tokens=2000)
        prompt = f"{system_prompt}\n\nResearch Question:\n{limited_query}\n\nDecomposed Sub-Questions:"
        
        response = llm.predict(prompt)
        
        # Extract sub-questions using multiple patterns
        sub_questions = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            # Match patterns like "1. Question", "1) Question", etc.
            if (line[0].isdigit() or 
                (len(line) > 2 and line[0].isdigit() and line[1] in ['.', ')', ':'])):
                # Remove the number prefix and clean up
                cleaned = line.split('.', 1)[-1].split(')', 1)[-1].strip()
                if cleaned:
                    sub_questions.append(cleaned)
        
        if not sub_questions:
            # Fallback: try to extract any sentences that end with question marks
            import re
            question_pattern = r'[^.!?]*\?'
            sub_questions = re.findall(question_pattern, response)
            sub_questions = [q.strip() for q in sub_questions if q.strip()]
        
        # Validate each sub-question
        sub_questions = [validate_sub_query(q) for q in sub_questions]
        
        logger.info(f"Decomposed into {len(sub_questions)} sub-questions")
        return sub_questions if sub_questions else [query]
    except Exception as e:
        logger.error(f"Query decomposition failed: {e}")
        return [query]

# --- Tool Selection Strategy ---
def should_use_tools(query: str) -> bool:
    """Determine if external tools should be used to answer the query."""
    prompt = f"""
    Analyze this query: "{truncate_text(query, max_tokens=1000)}"
    
    Would this query benefit from using external tools like web search or content analysis?
    Consider factors like:
    - Is this about current events or recent developments?
    - Does this require factual data rather than general knowledge?
    - Is this about specific products, companies, or people?
    - Is this too complex for general knowledge?
    - Would recent web data provide better answers?
    
    Answer YES or NO only.
    """
    try:
        response = llm.predict(prompt).strip().upper()
        return "YES" in response
    except:
        # Default to using tools
        return True

# --- Rate Limiter for API Calls ---
class RateLimiter:
    def __init__(self, max_calls=5, time_window=60):
        self.max_calls = max_calls  # Max calls allowed in time window
        self.time_window = time_window  # Time window in seconds
        self.calls = []  # List of timestamps of calls
    
    def can_call(self):
        """Check if we can make a call"""
        now = time.time()
        # Remove calls outside time window
        self.calls = [call for call in self.calls if call > now - self.time_window]
        # If we haven't reached max calls, allow call
        return len(self.calls) < self.max_calls
    
    def add_call(self):
        """Add a call to the history"""
        self.calls.append(time.time())
    
    def wait_time(self):
        """Calculate time to wait before next call is allowed"""
        if self.can_call():
            return 0
        # Return time until oldest call expires
        return self.calls[0] + self.time_window - time.time()

# More conservative rate limiting
api_limiter = RateLimiter(max_calls=3, time_window=60)

# --- Agent Setup ---
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
tracer = LangChainTracer()
tools = [search, scraper_tool, analyzer_tool, news_tool]

# Custom prompt template that encourages tool usage
custom_prompt = """You are an advanced web research agent that helps users find accurate information. You have access to these tools:

{tools}

To use a tool, use the following format:
```
Thought: I need to find information about X
Action: tool_name
Action Input: input for the tool
```

After you use a tool, I'll show you the result:
```
Observation: tool result
```

Always use tools for factual information rather than relying on what you already know. This ensures that your answers are up-to-date and accurate.

Start by thinking about which tools would help answer the user's question. Don't skip the tool-using step for factual or current information.

Begin!

User Question: {input}
{agent_scratchpad}
"""

agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    callbacks=[tracer],
    agent_kwargs={
        "prefix": custom_prompt
    },
    handle_parsing_errors=True  # Enable handling of parsing errors
)

# --- Modified research process with tool usage and rate limiting ---
def research_sub_query(sub_query: str) -> str:
    """Research a single sub-query, encouraging tool usage."""
    logger.info(f"Researching sub-query: {sub_query}")
    
    # Validate sub-query
    sub_query = validate_sub_query(sub_query)
    
    # Modify the query to encourage tool usage
    if should_use_tools(sub_query):
        # Add a prompt that encourages tool usage
        research_query = f"Research thoroughly using available tools: {sub_query}"
    else:
        research_query = sub_query
    
    # Limit query size
    limited_query = truncate_text(research_query, max_tokens=3500)
        
    start_time = time.time()
    try:
        # Check API rate limits
        wait_time = api_limiter.wait_time()
        if wait_time > 0:
            logger.info(f"Rate limit approaching, waiting {wait_time:.2f} seconds")
            time.sleep(wait_time)
        
        # Add call to rate limiter
        api_limiter.add_call()
        
        # Execute agent with exponential backoff for 429 errors
        max_retries = 3
        retry_delay = 2  # Initial delay in seconds
        
        for attempt in range(max_retries):
            try:
                result = agent_executor.invoke({"input": limited_query})
                response = result.get("output", result)
                
                # Sanitize response if needed
                if isinstance(response, str) and "```" in response:
                    response = sanitize_llm_output(response)
                    
                break
            except Exception as e:
                if ("429" in str(e) or "413" in str(e)) and attempt < max_retries - 1:
                    # If it's a 413 Payload Too Large error, reduce the query size
                    if "413" in str(e):
                        logger.warning("413 Payload too large error detected, reducing query size")
                        # Reduce content by 50%
                        limited_query = truncate_text(limited_query, max_tokens=int(len(limited_query) * 0.5 / 4))
                    
                    retry_delay *= 2  # Exponential backoff
                    logger.warning(f"API error ({e}), retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise e
    except Exception as e:
        logger.error(f"Error researching sub-query: {e}")
        response = f"Error researching: {e}"
    
    logger.info(f"Sub-query completed in {time.time() - start_time:.2f} seconds")
    
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
    
    return response

# --- Streamlit UI ---
st.set_page_config(page_title="Web Research Agent", layout="wide")
st.title("🌐 Advanced Web Research Agent")

st.markdown("""
<div style="padding: 1rem; background-color: #f0f2f6; border-radius: 10px;">
    <strong>👋 Welcome to your personal Web Research Agent!</strong><br>
    Ask complex questions and get comprehensive answers with cited sources. This agent searches the web, scrapes content, analyzes information, and synthesizes findings to give you reliable insights.
</div>
""", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    use_decomposition = st.checkbox("Decompose complex queries", value=True, 
                                   help="Break down complex questions into multiple sub-questions")
    search_depth = st.slider("Search depth", min_value=1, max_value=5, value=3, 
                            help="Number of search pages to analyze")
    st.markdown("---")
    

# Main query input
st.markdown("### 🔎 Ask a Research Question")
query = st.text_input("", placeholder="e.g., What are the latest AI breakthroughs in healthcare?", key="query_input")

if query:
    with st.spinner("🧠 Researching your question..."):
        # Reset tool usage tracker for this query
        tool_tracker.reset()  # Use the reset method instead of creating a new instance
        
        # Determine if query should be decomposed
        if use_decomposition and len(query.split()) > 5:  # Only decompose if query is complex
            sub_queries = decompose_query(query)
            
            # Create progress bar
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            # Research each sub-query
            combined_response = ""
            sub_responses = []
            
            for i, sub_q in enumerate(sub_queries, start=1):
                # Ensure sub-query is properly formatted
                validated_sub_q = validate_sub_query(sub_q)
                progress_text.text(f"Researching sub-question {i}/{len(sub_queries)}: {validated_sub_q}")
                result = research_sub_query(validated_sub_q)
                sub_responses.append(result)
                combined_response += f"### 🔹 Sub-Query {i}: {validated_sub_q}\n{result}\n\n"
                progress_bar.progress(i / len(sub_queries))
            
            # Final summary generation
            progress_text.text("Generating final summary...")
            
            # Collect all sub-responses and truncate/process if needed
            processed_responses = []
            for i, resp in enumerate(sub_responses):
                truncated_resp = truncate_text(resp, max_tokens=1500)
                processed_responses.append(f"{i+1}. {truncated_resp}")
            
            summary_prompt = (
                "You are a research assistant. Below are answers to several sub-questions related to a user's main query.\n\n"
                f"Main Query: {query}\n\n"
                + "\n\n".join(processed_responses)
                + "\n\nPlease synthesize the key insights into a concise summary that directly answers the main query."
                + "\n\nInclude only the most important information and cite specific facts where possible."
            )
            
            final_summary = llm.predict(summary_prompt)
            
            # Add citations to the final summary
            final_summary_with_citations = add_citations_to_summary(final_summary, tool_tracker.get_sources())
            
            combined_response += f"---\n\n### 🧾 Final Summary\n{final_summary_with_citations}"
            st.session_state.chat_history.append((query, combined_response))
            
            # Clear progress indicators
            progress_bar.empty()
            progress_text.empty()
            
            # Report on tool usage
            st.success(f"✅ Research completed! Used search {tool_tracker.usage['search']} times and scraper {tool_tracker.usage['scraper']} times.")
        else:
            # For simple queries, don't decompose
            result = research_sub_query(query)
            
            # Add citations to the result
            result_with_citations = add_citations_to_summary(result, tool_tracker.get_sources())
            
            st.session_state.chat_history.append((query, result_with_citations))
            st.success("✅ Research completed!")

# Display chat history
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

# Debug section
with st.expander("📑 Debug Information"):
    st.subheader("Tool Usage")
    st.json(tool_tracker.usage)
    
    st.subheader("Sources Collected")
    source_list = tool_tracker.get_sources()
    if source_list:
        for i, source in enumerate(source_list, 1):
            st.markdown(f"{i}. [{source.get('title', 'Untitled')}]({source.get('link', '#')})")
    else:
        st.text("No sources collected yet")
    
    st.subheader("System Information")
    st.code(f"""
    LLM: Groq Llama3-8b-8192
    Search Depth: {search_depth if 'search_depth' in locals() else 3}
    Query Decomposition: {'Enabled' if 'use_decomposition' in locals() and use_decomposition else 'Disabled'}
    """)