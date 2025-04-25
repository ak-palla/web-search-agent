ADVANCED WEB RESEARCH AGENT
============================

A powerful AI-powered web research assistant built with LangChain, Streamlit, and Groq LLM that 
helps users find comprehensive answers to complex questions by searching, scraping, and analyzing 
web content.

FEATURES
--------

- Intelligent Query Decomposition: Breaks down complex questions into manageable sub-questions
- Multi-Tool Ecosystem: Integrates search, scraping, content analysis, and news aggregation
- Semantic Re-Ranking: Uses embeddings to rank search results by relevance
- Citation System: Provides sources for all information gathered
- Adaptive Strategy: Optimizes search based on query type (news, technical, opinion, etc.)
- Robustness: Multiple fallback mechanisms and error handling
- Rate Limiting: Prevents API overuse and handles 429/413 errors
- User-Friendly Interface: Clean Streamlit UI with progress tracking and debug information

INSTALLATION
------------

1. Clone this repository:
   git clone https://github.com/ak-palla/web-research-agent.git
   cd web-research-agent

2. Install the required dependencies:
   pip install -r requirements.txt

3. Set up your environment variables by creating a .env file with the following:
   GROQ_API_KEY=your_groq_api_key
   OPENAI_API_KEY=your_openai_api_key
   FIRECRAWL_API_KEY=your_firecrawl_api_key
   LANGCHAIN_API_KEY=your_langchain_api_key

USAGE
-----

Run the Streamlit app:
   streamlit run web_search_agent.py

The web interface will open in your browser, where you can:
1. Enter your research question in the text input
2. Configure settings in the sidebar
3. View the detailed research results with sources
4. Access debug information if needed

REQUIREMENTS
-----------

- Python 3.8+
- API keys for Groq, OpenAI (for embeddings), Firecrawl, and LangChain
- See requirements.txt for all Python package dependencies

HOW IT WORKS
-----------

1. Query Analysis: The agent first determines the type of query and if it needs decomposition
2. Query Decomposition: Complex queries are broken down into targeted sub-questions
3. Tool Selection: For each sub-question, the agent chooses the most appropriate tools
4. Research Execution: The agent researches each sub-question with web search, scraping, and analysis
5. Synthesis: Results from all sub-questions are combined into a comprehensive final summary
6. Citation: Sources are tracked and cited in the final response

AGENT TOOLS
-----------

- Web Search Tool: Uses DuckDuckGo with semantic re-ranking to find relevant information
- Web Scraper: Extracts detailed content from webpages using Firecrawl or BeautifulSoup
- Content Analyzer: Summarizes and analyzes raw content
- News Aggregator: Finds the latest news on a topic from reliable sources

CUSTOMIZATION
------------

You can customize the agent's behavior by:
- Adjusting the rate limits in the RateLimiter class
- Modifying the search depth and query decomposition settings
- Adding new tools to the tools list
- Customizing the agent prompt in the custom_prompt variable

DOCUMENTATION
------------

See DOCUMENTATION.md for comprehensive technical details about how the agent works.

TEST CASES
---------

Refer to TEST_CASES.md for a set of test scenarios to validate the agent's capabilities.

CONTRIBUTING
-----------

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (git checkout -b feature/amazing-feature)
3. Commit your changes (git commit -m 'Add some amazing feature')
4. Push to the branch (git push origin feature/amazing-feature)
5. Open a Pull Request

LICENSE
------

This project is licensed under the MIT License - see the LICENSE file for details.

ACKNOWLEDGEMENTS
--------------

- LangChain for the agent framework
- Streamlit for the web interface
- Groq for the LLM API
- Firecrawl for web scraping capabilities
- DuckDuckGo for search functionality
