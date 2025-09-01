import asyncio
import logging
import json
import os
import sys
import time
from typing import Dict, Any, List, Optional
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
import weakref

# Crawl4AI imports
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("crawl8")

# FastAPI app
app = FastAPI(title="Crawl8 Service", version="1.0.0")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models (same as before)
class Article(BaseModel):
    url: str

class DataItem(BaseModel):
    json: Dict[str, str]

class BatchInput(BaseModel):
    class Config:
        extra = "allow"

class Source(BaseModel):
    url: str
    id: Optional[int] = None
    title: Optional[str] = None

class MultipleCrawlRequest(BaseModel):
    sources: List[Source]
    rules: Dict[str, str]

# URL content cache with TTL
class URLContentCache:
    def __init__(self, ttl_seconds: int = 3600):
        self._cache = {}
        self._timestamps = {}
        self.ttl_seconds = ttl_seconds
    
    def get(self, url: str) -> Optional[Dict[str, Any]]:
        if url in self._cache:
            if time.time() - self._timestamps[url] < self.ttl_seconds:
                logger.info(f"Cache HIT for URL: {url}")
                return self._cache[url]
            else:
                # Cache expired
                del self._cache[url]
                del self._timestamps[url]
        return None
    
    def set(self, url: str, content: Dict[str, Any]):
        self._cache[url] = content
        self._timestamps[url] = time.time()
        logger.info(f"Cache SET for URL: {url}")
    
    def clear(self):
        self._cache.clear()
        self._timestamps.clear()

# Global cache instance
url_cache = URLContentCache(ttl_seconds=3600)  # 1 hour TTL

# Middleware for logging (same as before)
@app.middleware("http")
async def log_crawl_requests(request: Request, call_next):
    if request.url.path in ["/crawl-batch"]:
        body = await request.body()
        logger.info(f"[{request.url.path} Request Body] {body.decode('utf-8') if body else 'No body'}")
        response = await call_next(request)
        resp_body = b""
        async for chunk in response.body_iterator:
            resp_body += chunk
        logger.info(f"[{request.url.path} Response Body] {resp_body.decode('utf-8') if resp_body else 'No response'}")
        return Response(
            content=resp_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )
    else:
        return await call_next(request)

# Utility functions (same as before)
def _parse_json_lax(raw: str):
    """Try to parse JSON from a possibly noisy response containing nested JSON."""
    if not raw or not isinstance(raw, str):
        return None
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass
    return None

# Enhanced OllamaCrawler with caching and parallel processing
class OllamaCrawler:
    def __init__(self, ollama_url: str = "http://13.203.188.153:11434"):
        self.ollama_url = ollama_url
        self.model = "gpt-oss:20b"  # Updated model name
        self.crawler = AsyncWebCrawler()
        self.http_client = httpx.AsyncClient(timeout=6000.0)
        # Semaphore to limit concurrent crawls (adjust based on your server capacity)
        self.crawl_semaphore = asyncio.Semaphore(10)
        # Semaphore to limit concurrent Ollama requests
        self.ollama_semaphore = asyncio.Semaphore(25)
    
    async def get_url_content(self, url: str) -> Dict[str, Any]:
        """Get URL content with caching to avoid duplicate crawls"""
        # Check cache first
        cached_content = url_cache.get(url)
        if cached_content:
            return cached_content
        
        async with self.crawl_semaphore:
            # Double-check cache in case another request crawled it while waiting
            cached_content = url_cache.get(url)
            if cached_content:
                return cached_content
            
            try:
                prune_filter = PruningContentFilter(
                    threshold=0.7,
                    threshold_type="fixed",
                    min_word_threshold=20
                )
                
                md_generator = DefaultMarkdownGenerator(
                    content_filter=prune_filter,
                    options={
                        "ignore_links": True,
                        "ignore_images": True,
                        "body_only": True
                    }
                )
                
                run_config = CrawlerRunConfig(
                    # excluded_tags=['header', 'footer', 'nav', 'aside', 'script', 'style', 'form'],
                    excluded_tags=[],
                    word_count_threshold=20,
                    exclude_external_links=True,
                    remove_overlay_elements=True,
                    process_iframes=False,
                    markdown_generator=md_generator,
                    cache_mode=CacheMode.ENABLED
                )
                
                logger.info(f"Crawling URL: {url}")
                result = await self.crawler.arun(url=url, config=run_config)
                
                if not result.success:
                    crawl_result = {
                        "success": False,
                        "error_message": result.error_message or "Crawl failed",
                        "status_code": result.status_code,
                        "raw_text": None
                    }
                else:
                    raw_text = ""
                    if result.markdown and result.markdown.fit_markdown:
                        raw_text = result.markdown.fit_markdown
                    elif result.markdown and result.markdown.raw_markdown:
                        raw_text = result.markdown.raw_markdown
                    else:
                        raw_text = result.cleaned_html or ""
                    
                    crawl_result = {
                        "success": True,
                        "raw_text": raw_text,
                        "status_code": result.status_code,
                        "error_message": None
                    }
                
                # Cache the result
                url_cache.set(url, crawl_result)
                return crawl_result
                
            except Exception as e:
                logger.error(f"Error crawling {url}: {str(e)}")
                error_result = {
                    "success": False,
                    "error_message": f"Crawling error: {str(e)}",
                    "status_code": None,
                    "raw_text": None
                }
                # Cache error results for a shorter time to allow retries
                url_cache.set(url, error_result)
                return error_result

    async def extract_with_ollama(self, content: str, query: str, url: str) -> Dict[str, Any]:
        """Extract information using Ollama with semaphore limiting"""
        async with self.ollama_semaphore:
            try:
                prompt = f"""
You are an expert information extraction assistant. 
Analyze the given web content and return only the direct answer to the query.

QUERY: {query}

WEB CONTENT:
{content}

STRICT INSTRUCTIONS:

1. Your entire response must be ONLY valid JSON — no natural language outside the JSON.
2. Do not repeat or rephrase the query in the output.
3. Matching rules:
   - City must match exactly (case-insensitive).
   - Doctor's name may match with variations (case-insensitive, or common abbreviations).
   - Specialty may match with synonyms or abbreviations (e.g., "Gastroenterology" ≈ "Gastroenterologist").
4. If the doctor's name, city, AND specialty do not all match under these rules, return "Not specified" for every field.
5. If the answer is missing, return "Not specified".
6. Always include all five top-level fields exactly as shown.
7. "extracted_details" must contain exactly one key-value pair. Use the most relevant key and value from the content.
8. Keep all values short and text-only (no markdown, no extra formatting).
9. Confidence must be one of: "high", "medium", "low".

OUTPUT FORMAT (no deviations, no extra text):
{{
  "query_answer": "Direct extracted answer or 'Not specified'",
  "extracted_details": {{
    "key1": "value1"
  }},
  "confidence": "high | medium | low",
  "source_context": "Brief description of where the info was located",
  "additional_notes": "Other relevant insights or clarifications"
}}
"""


                ollama_payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.8,
                        "max_tokens": 30000,
                        "overlap_ratio": 0.2
                    }
                }
                
                logger.info(f"Sending extraction request to Ollama for query: {query}")
                
                response = await self.http_client.post(
                    f"{self.ollama_url}/api/generate",
                    json=ollama_payload,
                    headers={"Content-Type": "application/json"}
                )
                
                response.raise_for_status()
                response_data = response.json()
                
                ollama_response_text = response_data.get("response", "")
                extracted_info = _parse_json_lax(ollama_response_text)
                
                if extracted_info is None:
                    logger.warning("JSON parsing failed, using fallback response")
                    extracted_info = {
                        "query_answer": ollama_response_text,
                        "extracted_details": {},
                        "confidence": "low",
                        "source_context": "Raw AI response",
                        "additional_notes": "JSON parsing failed, returning raw response"
                    }
                
                return extracted_info
                
            except httpx.HTTPError as e:
                logger.error(f"HTTP error with Ollama: {str(e)}")
                return {
                    "query_answer": "Error in AI extraction",
                    "extracted_details": {},
                    "confidence": "low",
                    "source_context": "Error occurred",
                    "additional_notes": f"Ollama HTTP error: {str(e)}",
                    "error": str(e)
                }
            except Exception as e:
                logger.error(f"Error with Ollama extraction: {str(e)}")
                return {
                    "query_answer": "Error in AI extraction",
                    "extracted_details": {},
                    "confidence": "low",
                    "source_context": "Error occurred",
                    "additional_notes": f"Ollama API error: {str(e)}",
                    "error": str(e)
                }

    async def process_field_parallel(self, field_name: str, query: str, sources: List[Source]) -> Dict[str, Any]:
        """Process a single field across all sources in parallel"""
        logger.info(f"Processing field: {field_name} with query: {query}")
        
        field_result = {
            "query": query,
            "found": False,
            "result": None,
            "source_used": None,
            "attempts": []
        }

        # Create tasks to get content from all URLs in parallel
        url_tasks = []
        for source in sources:
            url_tasks.append(self.get_url_content(source.url))
        
        # Wait for all URL content to be fetched
        url_contents = await asyncio.gather(*url_tasks, return_exceptions=True)
        
        # Process each source for this field
        for idx, source in enumerate(sources):
            source_id = source.id if source.id is not None else idx + 1
            source_title = source.title if source.title else f"Source {source_id}"
            
            url_content = url_contents[idx]
            
            # Handle exceptions from URL fetching
            if isinstance(url_content, Exception):
                attempt = {
                    "source_id": source_id,
                    "source_url": source.url,
                    "source_title": source_title,
                    "success": False,
                    "query_answer": "",
                    "confidence": "low",
                    "error": str(url_content)
                }
                field_result["attempts"].append(attempt)
                continue
            
            if not url_content.get("success", False):
                attempt = {
                    "source_id": source_id,
                    "source_url": source.url,
                    "source_title": source_title,
                    "success": False,
                    "query_answer": "",
                    "confidence": "low",
                    "error": url_content.get("error_message", "Unknown crawl error")
                }
                field_result["attempts"].append(attempt)
                continue
            
            # Extract information for this specific query
            raw_text = url_content.get("raw_text", "")
            if not raw_text:
                attempt = {
                    "source_id": source_id,
                    "source_url": source.url,
                    "source_title": source_title,
                    "success": False,
                    "query_answer": "No content found",
                    "confidence": "low"
                }
                field_result["attempts"].append(attempt)
                continue
            
            try:
                extracted_data = await self.extract_with_ollama(raw_text, query, source.url)
                query_answer = extracted_data.get("query_answer", "")
                confidence = extracted_data.get("confidence", "low")
                
                attempt = {
                    "source_id": source_id,
                    "source_url": source.url,
                    "source_title": source_title,
                    "success": True,
                    "query_answer": query_answer,
                    "confidence": confidence
                }
                field_result["attempts"].append(attempt)
                
                # Check if we got a valid answer
                if self.is_valid_answer(query_answer):
                    logger.info(f"Found valid answer for {field_name}: {query_answer}")
                    field_result["found"] = True
                    field_result["result"] = {
                        "success": True,
                        "url": source.url,
                        "query": query,
                        "extracted_data": extracted_data,
                        "raw_text": raw_text[:2000] if raw_text else None,
                        "status_code": url_content.get("status_code")
                    }
                    field_result["source_used"] = {
                        "id": source_id,
                        "title": source_title,
                        "url": source.url
                    }
                    break  # Found valid data, stop trying other sources
                else:
                    logger.info(f"Invalid answer for {field_name} from source {source_id}: {query_answer}")
                    
            except Exception as e:
                logger.error(f"Error processing source {source_id} for field {field_name}: {str(e)}")
                attempt = {
                    "source_id": source_id,
                    "source_url": source.url,
                    "source_title": source_title,
                    "success": False,
                    "query_answer": "",
                    "confidence": "low",
                    "error": str(e)
                }
                field_result["attempts"].append(attempt)
        
        if not field_result["found"]:
            logger.warning(f"No valid data found for field {field_name} after trying all sources")
        
        return field_result

    def is_valid_answer(self, query_answer: str) -> bool:
        if not query_answer:
            return False
        
        invalid_responses = [
            "not found", "n/a", "na", "not available", "not mentioned",
            "information not found", "error in ai extraction", "not specified"
        ]
        
        return query_answer.lower().strip() not in invalid_responses

    async def close(self):
        """Close the HTTP client"""
        await self.http_client.aclose()

# Initialize crawler with Ollama
ollama_crawler = OllamaCrawler(ollama_url="http://13.203.188.153:1145")

# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "ok", "service": "crawl8"}

# Clear cache endpoint
@app.post("/clear-cache")
async def clear_cache():
    url_cache.clear()
    return {"status": "cache cleared"}

# Main batch processing endpoint with parallel processing
@app.post("/crawl-batch")
async def crawl_batch(request: Request):
    """
    Batch processing endpoint with parallel processing and caching
    """
    try:
        # Get raw JSON data (same parsing logic as before)
        data = await request.json()
        logger.info(f"Received batch request with data: {data}")
        
        rules_dict = {}
        articles = []
        
        for key, value in data.items():
            if key == "articles":
                articles = [Article(**article) for article in value]
            elif key.isdigit() or key.isdecimal():
                if isinstance(value, dict) and "json" in value:
                    rules_dict.update(value["json"])
            else:
                if isinstance(value, dict) and "json" in value:
                    rules_dict.update(value["json"])
        
        if not rules_dict:
            return {"success": False, "error": "No valid rules found in the request"}
        
        if not articles:
            return {"success": False, "error": "No articles found in the request"}
        
        # Convert articles to sources format
        sources = []
        for idx, article in enumerate(articles):
            source = Source(
                url=article.url,
                id=idx + 1,
                title=f"Article {idx + 1}"
            )
            sources.append(source)
        
        crawl_request = MultipleCrawlRequest(sources=sources, rules=rules_dict)
        
        # **KEY IMPROVEMENT: Process all fields in parallel**
        logger.info(f"Starting parallel processing of {len(rules_dict)} fields across {len(sources)} sources")
        
        # Create tasks for all fields to be processed in parallel
        field_tasks = []
        for field_name, query in crawl_request.rules.items():
            task = ollama_crawler.process_field_parallel(field_name, query, crawl_request.sources)
            field_tasks.append((field_name, task))
        
        # Execute all field processing tasks in parallel
        start_time = time.time()
        field_results = await asyncio.gather(*[task for _, task in field_tasks], return_exceptions=True)
        end_time = time.time()
        
        # Build results dictionary
        results = {}
        for idx, (field_name, _) in enumerate(field_tasks):
            field_result = field_results[idx]
            if isinstance(field_result, Exception):
                logger.error(f"Error processing field {field_name}: {str(field_result)}")
                results[field_name] = {
                    "query": crawl_request.rules[field_name],
                    "found": False,
                    "result": None,
                    "source_used": None,
                    "attempts": [],
                    "error": str(field_result)
                }
            else:
                results[field_name] = field_result
        
        # Create summary
        summary = {
            "total_fields": len(crawl_request.rules),
            "fields_found": sum(1 for result in results.values() if result.get("found", False)),
            "fields_not_found": sum(1 for result in results.values() if not result.get("found", False)),
            "total_sources": len(crawl_request.sources),
            "total_attempts": sum(len(result.get("attempts", [])) for result in results.values()),
            "processing_time_seconds": round(end_time - start_time, 2),
            "parallel_processing": True
        }
        
        logger.info(f"Completed parallel processing in {summary['processing_time_seconds']} seconds")
        
        return {
            "success": True,
            "summary": summary,
            "results": results,
            "request_info": {
                "sources_count": len(crawl_request.sources),
                "rules_count": len(crawl_request.rules),
                "cache_enabled": True
            }
        }
        
    except Exception as e:
        logger.error(f"Error in /crawl-batch: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    await ollama_crawler.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8233)
