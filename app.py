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

# Pydantic models for the exact input format
class Article(BaseModel):
    url: str

class DataItem(BaseModel):
    json: Dict[str, str]

class BatchInput(BaseModel):
    # This allows any number of numbered keys plus articles
    class Config:
        extra = "allow"  # Allow additional fields not defined in the model

# Internal processing models
class Source(BaseModel):
    url: str
    id: Optional[int] = None
    title: Optional[str] = None

class MultipleCrawlRequest(BaseModel):
    sources: List[Source]
    rules: Dict[str, str]

# Middleware for logging
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

# Utility functions
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

# OllamaCrawler class (replaces GeminiCrawler)
class OllamaCrawler:
    def __init__(self, ollama_url: str = "http://13.203.188.153:11434"):
        self.ollama_url = ollama_url
        self.model = "llama3.1:8b"
        self.crawler = AsyncWebCrawler()
        self.http_client = httpx.AsyncClient(timeout=6000.0)

    async def crawl_and_extract(self, url: str, query: str) -> Dict[str, Any]:
        try:
            prune_filter = PruningContentFilter(
                threshold=0.5,
                threshold_type="fixed",
                min_word_threshold=10
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
                excluded_tags=['header', 'footer', 'nav', 'aside', 'script', 'style', 'form'],
                word_count_threshold=10,
                exclude_external_links=True,
                remove_overlay_elements=True,
                process_iframes=False,
                markdown_generator=md_generator,
                cache_mode=CacheMode.ENABLED
            )
            
            result = await self.crawler.arun(url=url, config=run_config)
            
            if not result.success:
                return {
                    "success": False,
                    "error_message": result.error_message or "Crawl failed",
                    "status_code": result.status_code
                }
            
            raw_text = ""
            if result.markdown and result.markdown.fit_markdown:
                raw_text = result.markdown.fit_markdown
            elif result.markdown and result.markdown.raw_markdown:
                raw_text = result.markdown.raw_markdown
            else:
                raw_text = result.cleaned_html or ""
            
            extracted_data = await self.extract_with_ollama(raw_text, query, url)
            
            return {
                "success": True,
                "url": url,
                "query": query,
                "extracted_data": extracted_data,
                "raw_text": raw_text[:2000] if raw_text else None,
                "status_code": result.status_code
            }
            
        except Exception as e:
            logger.error(f"Error crawling {url}: {str(e)}")
            return {
                "success": False,
                "error_message": f"Crawling error: {str(e)}",
                "status_code": None
            }

    async def extract_with_ollama(self, content: str, query: str, url: str) -> Dict[str, Any]:
        try:
            prompt = f"""You are an expert information extraction assistant. Analyze the following web content and extract information based on the specific query.

QUERY: {query}

WEB CONTENT:
{content}

INSTRUCTIONS:
1. Focus specifically on the query: "{query}"
2. Extract only relevant information that directly answers the query
3. Structure your response as JSON with clear key-value pairs
4. If the information is not found, clearly state "Not found" for that field
5. Include confidence level (high/medium/low) for each extracted piece of information
6. Provide source context (which part of the content the info came from)

RESPONSE FORMAT (JSON):
{{
    "query_answer": "Direct answer to the query",
    "extracted_details": {{
        "key1": "value1",
        "key2": "value2"
    }},
    "confidence": "high/medium/low",
    "source_context": "Brief description of where info was found",
    "additional_notes": "Any additional relevant information"
}}

Return ONLY the JSON response, no additional text."""
            
            # Make request to Ollama API
            ollama_payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            
            logger.info(f"Sending request to Ollama at {self.ollama_url}/api/generate")
            
            response = await self.http_client.post(
                f"{self.ollama_url}/api/generate",
                json=ollama_payload,
                headers={"Content-Type": "application/json"}
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            # Extract the response text from Ollama's response format
            ollama_response_text = response_data.get("response", "")
            logger.info(f"Raw Ollama response: {ollama_response_text}")
            
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
ollama_crawler = OllamaCrawler(ollama_url="http://13.203.188.153:11434")

# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "ok", "service": "crawl8"}

# Main batch processing endpoint for exact input format
@app.post("/crawl-batch")
async def crawl_batch(request: Request):
    """
    Batch processing endpoint for exact input format:
    {
        "0": {
            "json": {
                "field_name": "query text"
            }
        },
        "1": {
            "json": {
                "field_name_2": "query text 2"
            }
        },
        "articles": [
            {
                "url": "https://example.com"
            }
        ]
    }
    """
    try:
        # Get raw JSON data
        data = await request.json()
        logger.info(f"Received batch request with data: {data}")
        
        # Extract rules from numbered keys (0, 1, 2, etc.)
        rules_dict = {}
        articles = []
        
        for key, value in data.items():
            if key == "articles":
                # Handle articles array
                articles = [Article(**article) for article in value]
            elif key.isdigit() or key.isdecimal():
                # Handle numbered data items (0, 1, 2, etc.)
                if isinstance(value, dict) and "json" in value:
                    rules_dict.update(value["json"])
            else:
                # Handle other potential keys that might contain json data
                if isinstance(value, dict) and "json" in value:
                    rules_dict.update(value["json"])
        
        if not rules_dict:
            return {
                "success": False,
                "error": "No valid rules found in the request"
            }
        
        if not articles:
            return {
                "success": False,
                "error": "No articles found in the request"
            }
        
        # Convert articles to sources format
        sources = []
        for idx, article in enumerate(articles):
            source = Source(
                url=article.url,
                id=idx + 1,
                title=f"Article {idx + 1}"
            )
            sources.append(source)
        
        # Create the crawl request with converted data
        crawl_request = MultipleCrawlRequest(
            sources=sources,
            rules=rules_dict
        )
        
        results = {}
        
        # Process each rule (query)
        for field_name, query in crawl_request.rules.items():
            logger.info(f"Processing field: {field_name}")
            results[field_name] = {
                "query": query,
                "found": False,
                "result": None,
                "source_used": None,
                "attempts": []
            }
            
            # Try each source URL until we get valid data
            for idx, source in enumerate(crawl_request.sources):
                source_id = source.id if source.id is not None else idx + 1
                source_title = source.title if source.title else f"Source {source_id}"
                
                logger.info(f"Trying source {source_id}: {source.url} for field {field_name}")
                
                try:
                    crawl_result = await ollama_crawler.crawl_and_extract(source.url, query)
                    
                    # Record the attempt
                    attempt = {
                        "source_id": source_id,
                        "source_url": source.url,
                        "source_title": source_title,
                        "success": crawl_result.get("success", False),
                        "query_answer": crawl_result.get("extracted_data", {}).get("query_answer", ""),
                        "confidence": crawl_result.get("extracted_data", {}).get("confidence", "low")
                    }
                    results[field_name]["attempts"].append(attempt)
                    
                    if crawl_result.get("success", False):
                        query_answer = crawl_result.get("extracted_data", {}).get("query_answer", "")
                        
                        # Check if we got a valid answer
                        if ollama_crawler.is_valid_answer(query_answer):
                            logger.info(f"Found valid answer for {field_name}: {query_answer}")
                            results[field_name]["found"] = True
                            results[field_name]["result"] = crawl_result
                            results[field_name]["source_used"] = {
                                "id": source_id,
                                "title": source_title,
                                "url": source.url
                            }
                            break  # Found valid data, move to next field
                        else:
                            logger.info(f"Invalid answer for {field_name} from source {source_id}: {query_answer}")
                    else:
                        logger.warning(f"Crawl failed for {field_name} from source {source_id}: {crawl_result.get('error_message', 'Unknown error')}")
                        
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
                    results[field_name]["attempts"].append(attempt)
            
            if not results[field_name]["found"]:
                logger.warning(f"No valid data found for field {field_name} after trying all sources")
        
        # Create summary
        summary = {
            "total_fields": len(crawl_request.rules),
            "fields_found": sum(1 for result in results.values() if result["found"]),
            "fields_not_found": sum(1 for result in results.values() if not result["found"]),
            "total_sources": len(crawl_request.sources),
            "total_attempts": sum(len(result["attempts"]) for result in results.values())
        }
        
        return {
            "success": True,
            "summary": summary,
            "results": results,
            "request_info": {
                "sources_count": len(crawl_request.sources),
                "rules_count": len(crawl_request.rules)
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
