import asyncio
import logging
import json
import os
import sys
import time
from typing import Dict, Any, List, Optional, Set
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
import weakref
from urllib.parse import urlparse
import re

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

# Pydantic models
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
url_cache = URLContentCache(ttl_seconds=3600)

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

def extract_doctor_info_from_query(query: str) -> Dict[str, str]:
    """Extract doctor name, specialty, and city from query string - FIXED VERSION"""
    doctor_info = {"name": "", "specialty": "", "city": ""}
    
    # Updated patterns to handle unquoted values
    # Pattern: "For doctor NAME specialty SPECIALTY, city CITY"
    
    # Extract name (after "doctor" until "specialty")
    name_pattern = r'doctor\s+([^,]+?)\s+specialty'
    name_match = re.search(name_pattern, query, re.IGNORECASE)
    if name_match:
        doctor_info["name"] = name_match.group(1).strip()
    
    # Extract specialty (after "specialty" until ", city")
    specialty_pattern = r'specialty\s+([^,]+?),\s*city'
    specialty_match = re.search(specialty_pattern, query, re.IGNORECASE)
    if specialty_match:
        doctor_info["specialty"] = specialty_match.group(1).strip()
    
    # Extract city (after "city" and before comma or end)
    city_pattern = r'city\s+([^,]+?)(?:,|$)'
    city_match = re.search(city_pattern, query, re.IGNORECASE)
    if city_match:
        doctor_info["city"] = city_match.group(1).strip()
    
    return doctor_info

def get_domain_from_url(url: str) -> str:
    """Extract domain from URL"""
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except:
        return url

def deduplicate_sources_by_domain(sources: List[Source]) -> List[Source]:
    """Remove duplicate domains, keeping only the first occurrence"""
    seen_domains = set()
    unique_sources = []
    
    for source in sources:
        domain = get_domain_from_url(source.url)
        if domain not in seen_domains:
            seen_domains.add(domain)
            unique_sources.append(source)
            logger.info(f"Keeping URL from domain: {domain} - {source.url}")
        else:
            logger.info(f"Skipping duplicate domain: {domain} - {source.url}")
    
    return unique_sources

# Enhanced OllamaCrawler with LLM-based URL validation and FLEXIBLE location matching
class OllamaCrawler:
    def __init__(self, ollama_url: str = "http://13.203.188.153:11434"):
        self.ollama_url = ollama_url
        self.model = "qwen2.5:14b"
        self.crawler = AsyncWebCrawler()
        self.http_client = httpx.AsyncClient(timeout=6000.0)
        self.crawl_semaphore = asyncio.Semaphore(10)
        self.ollama_semaphore = asyncio.Semaphore(25)
    
    async def llm_verify_content_relevance(self, content: str, doctor_info: Dict[str, str]) -> Dict[str, Any]:
        """Use LLM to verify if content is relevant to target doctor with FLEXIBLE location matching"""
        async with self.ollama_semaphore:
            try:
                doctor_name = doctor_info.get("name", "")
                specialty = doctor_info.get("specialty", "")
                city = doctor_info.get("city", "")
                
                prompt = f"""
You are an expert assistant evaluating medical webpage content for relevance to a specific doctor.

TARGET DOCTOR DETAILS:
- Name: {doctor_name}
- Specialty: {specialty}
- Target City: {city}

WEBPAGE CONTENT TO EVALUATE:
{content}

EVALUATION GUIDELINES:
1. PRIMARY CRITERIA: Focus on doctor's name and specialty matching
2. FLEXIBLE LOCATION MATCHING: Doctors often practice in multiple cities or have information mentioning different locations
3. ACCEPT if the doctor's name and specialty match, even if the city is different
4. Consider name variations: Dr. {doctor_name}, {doctor_name.title()}, etc.
5. Consider specialty synonyms: General Physician = General Practitioner = GP = Family Medicine
6. PRIORITIZE doctor identity over exact geographic location
7. Geographic proximity or related cities should be considered relevant

IMPORTANT: A doctor practicing in a nearby city or having multiple practice locations is still the SAME doctor.

Return ONLY JSON with this exact format:
{{
  "is_relevant": true/false,
  "confidence": "high|medium|low",
  "reason": "Brief explanation focusing on doctor identity match rather than location",
  "name_found": true/false,
  "specialty_found": true/false,
  "city_found": true/false,
  "location_note": "Any notes about location differences if applicable"
}}
"""
                
                ollama_payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                    }
                }
                
                logger.info(f"LLM verifying content relevance for doctor {doctor_name} with flexible location matching")
                
                response = await self.http_client.post(
                    f"{self.ollama_url}/api/generate",
                    json=ollama_payload,
                    headers={"Content-Type": "application/json"}
                )
                
                response.raise_for_status()
                response_data = response.json()
                
                ollama_response_text = response_data.get("response", "")
                relevance_info = _parse_json_lax(ollama_response_text)
                
                if relevance_info is None:
                    logger.warning("LLM relevance verification failed, assuming not relevant")
                    relevance_info = {
                        "is_relevant": False,
                        "confidence": "low",
                        "reason": "Failed to parse LLM response",
                        "name_found": False,
                        "specialty_found": False,
                        "city_found": False,
                        "location_note": "Error in verification"
                    }
                
                return relevance_info
                
            except Exception as e:
                logger.error(f"Error in LLM relevance verification: {str(e)}")
                return {
                    "is_relevant": False,
                    "confidence": "low",
                    "reason": f"Error in verification: {str(e)}",
                    "name_found": False,
                    "specialty_found": False,
                    "city_found": False,
                    "location_note": "Verification failed",
                    "error": str(e)
                }
    
    async def get_url_content(self, url: str) -> Dict[str, Any]:
        """Get URL content with caching"""
        cached_content = url_cache.get(url)
        if cached_content:
            return cached_content
        
        async with self.crawl_semaphore:
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
                url_cache.set(url, error_result)
                return error_result

    async def extract_multiple_fields_from_content(self, content: str, fields_dict: Dict[str, str], url: str, doctor_info: Dict[str, str]) -> Dict[str, Any]:
        """Extract MULTIPLE fields from content in a SINGLE call with FLEXIBLE doctor matching"""
        async with self.ollama_semaphore:
            try:
                doctor_name = doctor_info.get("name", "")
                specialty = doctor_info.get("specialty", "")
                city = doctor_info.get("city", "")
                
                # Create fields list for the prompt
                fields_list = []
                for field_name, query in fields_dict.items():
                    fields_list.append(f'"{field_name}": "{query}"')
                
                fields_json_string = "{\n" + ",\n".join(fields_list) + "\n}"

                prompt = f"""
You are an expert medical information extraction assistant.
Extract information for ALL the specified fields from the given web content about a specific doctor.

TARGET DOCTOR: {doctor_name}
SPECIALTY: {specialty}
TARGET CITY: {city}

EXTRACTION FIELDS:
{fields_json_string}

WEB CONTENT:
{content}

STRICT INSTRUCTIONS:
1. Your entire response must be ONLY valid JSON — no natural language outside the JSON.
2. Look specifically for information about "{doctor_name}" who specializes in "{specialty}".
3. Extract data for ALL fields provided above in a single response.
4. FLEXIBLE MATCHING RULES:
   - Accept name variations: Dr. {doctor_name}, {doctor_name.title()}, etc.
   - Accept specialty synonyms: General Physician = General Practitioner = GP = Family Medicine
   - IMPORTANT: Be flexible with city/location - doctors often practice in multiple cities
   - If the doctor's name and specialty match, consider it relevant even if location differs
5. For each field, if you find relevant information, extract it precisely
6. If information for a field is not found, return "Not specified" for that field.
7. Set doctor_matched to true if you're confident this is information about the target doctor (prioritize name+specialty over location)
8. Always include all fields requested, even if some are "Not specified".

OUTPUT FORMAT (no deviations, no extra text):
{{
  "extracted_fields": {{
    "field1_name": {{
      "query_answer": "Direct extracted answer or 'Not specified'",
      "confidence": "high | medium | low",
      "source_context": "Brief description of where the info was located",
      "doctor_matched": true/false
    }},
    "field2_name": {{
      "query_answer": "Direct extracted answer or 'Not specified'",
      "confidence": "high | medium | low",
      "source_context": "Brief description of where the info was located",
      "doctor_matched": true/false
    }}
  }},
  "overall_confidence": "high | medium | low"
}}
"""
                
                ollama_payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                    }
                }
                
                logger.info(f"Extracting {len(fields_dict)} fields in single call for doctor {doctor_name} with flexible matching")
                
                response = await self.http_client.post(
                    f"{self.ollama_url}/api/generate",
                    json=ollama_payload,
                    headers={"Content-Type": "application/json"}
                )
                
                response.raise_for_status()
                response_data = response.json()
                
                ollama_response_text = response_data.get("response", "")
                extracted_info = _parse_json_lax(ollama_response_text)
                
                if extracted_info is None or "extracted_fields" not in extracted_info:
                    logger.warning("Multi-field JSON parsing failed, using fallback response")
                    # Create fallback structure
                    fallback_fields = {}
                    for field_name in fields_dict.keys():
                        fallback_fields[field_name] = {
                            "query_answer": "Error in AI extraction",
                            "confidence": "low",
                            "source_context": "JSON parsing failed",
                            "doctor_matched": False
                        }
                    
                    extracted_info = {
                        "extracted_fields": fallback_fields,
                        "overall_confidence": "low"
                    }
                
                return extracted_info
                
            except Exception as e:
                logger.error(f"Error with multi-field Ollama extraction: {str(e)}")
                # Create error structure for all fields
                error_fields = {}
                for field_name in fields_dict.keys():
                    error_fields[field_name] = {
                        "query_answer": "Error in AI extraction",
                        "confidence": "low",
                        "source_context": "Error occurred",
                        "doctor_matched": False,
                        "error": str(e)
                    }
                
                return {
                    "extracted_fields": error_fields,
                    "overall_confidence": "low",
                    "error": str(e)
                }

    async def process_sources_optimized(self, sources: List[Source], rules: Dict[str, str]) -> Dict[str, Any]:
        """OPTIMIZED: LLM-based URL validation with FLEXIBLE location matching"""
        logger.info(f"Starting LLM-based processing with flexible location matching of {len(rules)} fields across {len(sources)} sources")
        
        # Extract doctor info from the first query
        first_query = next(iter(rules.values()))
        doctor_info = extract_doctor_info_from_query(first_query)
        logger.info(f"Extracted doctor info: {doctor_info}")
        
        # Remove duplicate domains first
        unique_sources = deduplicate_sources_by_domain(sources)
        
        # Filter relevant URLs using LLM verification with flexible location matching
        relevant_sources = []
        logger.info("Using LLM to filter URLs with flexible location matching...")
        
        for source in unique_sources:
            url_content = await self.get_url_content(source.url)
            if url_content.get("success", False):
                raw_text = url_content.get("raw_text", "")
                if raw_text:
                    # Use LLM to verify relevance with flexible location matching
                    relevance_result = await self.llm_verify_content_relevance(raw_text, doctor_info)
                    if relevance_result.get("is_relevant", False):
                        relevant_sources.append(source)
                        location_note = relevance_result.get("location_note", "")
                        logger.info(f"URL {source.url} is RELEVANT - LLM reason: {relevance_result.get('reason', 'No reason provided')} | Location: {location_note}")
                    else:
                        logger.info(f"URL {source.url} is NOT RELEVANT - LLM reason: {relevance_result.get('reason', 'No reason provided')}")
                else:
                    logger.info(f"URL {source.url} - No content found")
            else:
                logger.info(f"URL {source.url} FAILED to crawl - {url_content.get('error_message', 'Unknown error')}")
        
        if not relevant_sources:
            logger.warning("No relevant URLs found for the target doctor")
            results = {}
            for field_name in rules.keys():
                results[field_name] = {
                    "query": rules[field_name],
                    "found": False,
                    "result": None,
                    "source_used": None,
                    "attempts": []
                }
            return results
        
        logger.info(f"Processing {len(relevant_sources)} relevant URLs out of {len(sources)} total URLs")
        
        results = {}
        remaining_fields = dict(rules)  # Copy of rules for tracking remaining fields
        
        # Initialize results structure
        for field_name in rules.keys():
            results[field_name] = {
                "query": rules[field_name],
                "found": False,
                "result": None,
                "source_used": None,
                "attempts": []
            }
        
        # Process each source, trying to extract ALL remaining fields at once
        for idx, source in enumerate(relevant_sources):
            if not remaining_fields:
                logger.info("All fields found, stopping processing")
                break
            
            source_id = source.id if source.id is not None else idx + 1
            source_title = source.title if source.title else f"Source {source_id}"
            
            logger.info(f"Processing {len(remaining_fields)} remaining fields from source {source_id} ({source.url})")
            
            # Get URL content
            url_content = await self.get_url_content(source.url)
            
            if not url_content.get("success", False):
                # Add failed attempts for all remaining fields
                for field_name in remaining_fields.keys():
                    attempt = {
                        "source_id": source_id,
                        "source_url": source.url,
                        "source_title": source_title,
                        "success": False,
                        "query_answer": "",
                        "confidence": "low",
                        "error": url_content.get("error_message", "Unknown crawl error")
                    }
                    results[field_name]["attempts"].append(attempt)
                continue
            
            raw_text = url_content.get("raw_text", "")
            if not raw_text:
                # Add failed attempts for all remaining fields
                for field_name in remaining_fields.keys():
                    attempt = {
                        "source_id": source_id,
                        "source_url": source.url,
                        "source_title": source_title,
                        "success": False,
                        "query_answer": "No content found",
                        "confidence": "low"
                    }
                    results[field_name]["attempts"].append(attempt)
                continue
            
            try:
                # Extract ALL remaining fields from this source in ONE call
                multi_field_result = await self.extract_multiple_fields_from_content(
                    raw_text, remaining_fields, source.url, doctor_info
                )
                
                extracted_fields = multi_field_result.get("extracted_fields", {})
                
                # Process results for each field
                fields_found_in_this_source = []
                
                for field_name in list(remaining_fields.keys()):  # Convert to list to allow modification
                    field_data = extracted_fields.get(field_name, {})
                    query_answer = field_data.get("query_answer", "")
                    confidence = field_data.get("confidence", "low")
                    source_context = field_data.get("source_context", "")
                    doctor_matched = field_data.get("doctor_matched", False)
                    
                    attempt = {
                        "source_id": source_id,
                        "source_url": source.url,
                        "source_title": source_title,
                        "success": True,
                        "query_answer": query_answer,
                        "confidence": confidence,
                        "source_context": source_context,
                        "doctor_matched": doctor_matched
                    }
                    results[field_name]["attempts"].append(attempt)
                    
                    # ACCEPT VALID ANSWERS: Prioritize doctor_matched=true but accept valid answers
                    if self.is_valid_answer(query_answer):
                        # If we don't have an answer yet, OR if this has better doctor matching
                        should_accept = (
                            not results[field_name]["found"] or  # No answer yet
                            (doctor_matched and not self.get_current_doctor_matched_status(results[field_name])) or  # Better match
                            (doctor_matched and results[field_name]["found"])  # Exact match replacing any match
                        )
                        
                        if should_accept:
                            logger.info(f"Accepting answer for {field_name} from source {source_id}: {query_answer} (doctor_matched: {doctor_matched})")
                            results[field_name]["found"] = True
                            results[field_name]["result"] = {
                                "success": True,
                                "url": source.url,
                                "query": rules[field_name],
                                "extracted_data": {
                                    "query_answer": query_answer,
                                    "extracted_details": {field_name: query_answer},
                                    "confidence": confidence,
                                    "source_context": source_context,
                                    "additional_notes": ""
                                },
                                "raw_text": raw_text[:2000] if raw_text else None,
                                "status_code": url_content.get("status_code")
                            }
                            results[field_name]["source_used"] = {
                                "id": source_id,
                                "title": source_title,
                                "url": source.url
                            }
                            
                            # If we have a perfect match (doctor_matched=true), remove from remaining fields
                            if doctor_matched:
                                fields_found_in_this_source.append(field_name)
                                if field_name in remaining_fields:
                                    del remaining_fields[field_name]
                
                if fields_found_in_this_source:
                    logger.info(f"Source {source_id} provided data for {len(fields_found_in_this_source)} fields: {fields_found_in_this_source}")
                else:
                    logger.info(f"Source {source_id} provided data but no perfect doctor matches")
                    
            except Exception as e:
                logger.error(f"Error processing source {source_id}: {str(e)}")
                # Add failed attempts for all remaining fields
                for field_name in remaining_fields.keys():
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
        
        return results
    
    def get_current_doctor_matched_status(self, field_result: Dict) -> bool:
        """Check if current result has doctor_matched=true"""
        if field_result.get("result") and field_result["result"].get("extracted_data"):
            # This is a simplified check - in practice you'd store this info
            return False  # Assume false for simplicity, real implementation would track this
        return False
    
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

# Initialize crawler
ollama_crawler = OllamaCrawler(ollama_url="http://13.203.188.153:11434")

# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "ok", "service": "crawl8"}

# Clear cache endpoint
@app.post("/clear-cache")
async def clear_cache():
    url_cache.clear()
    return {"status": "cache cleared"}

# Main batch processing endpoint - FLEXIBLE LOCATION MATCHING VERSION
@app.post("/crawl-batch")
async def crawl_batch(request: Request):
    """
    LLM-based batch processing endpoint with FLEXIBLE location matching
    MAINTAINS ORIGINAL INPUT/OUTPUT FORMAT
    """
    try:
        # Parse request data - EXACT SAME AS ORIGINAL
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
        
        # Convert articles to sources format - SAME AS ORIGINAL
        sources = []
        for idx, article in enumerate(articles):
            source = Source(
                url=article.url,
                id=idx + 1,
                title=f"Article {idx + 1}"
            )
            sources.append(source)
        
        # Process with LLM-based algorithm with flexible location matching
        start_time = time.time()
        results = await ollama_crawler.process_sources_optimized(sources, rules_dict)
        end_time = time.time()
        
        # Create summary - SAME FORMAT AS ORIGINAL
        summary = {
            "total_fields": len(rules_dict),
            "fields_found": sum(1 for result in results.values() if result.get("found", False)),
            "fields_not_found": sum(1 for result in results.values() if not result.get("found", False)),
            "total_sources": len(sources),
            "total_attempts": sum(len(result.get("attempts", [])) for result in results.values()),
            "processing_time_seconds": round(end_time - start_time, 2),
            "optimization_enabled": True,
            "standardization_applied": False
        }
        
        logger.info(f"Completed LLM-based processing with flexible location matching in {summary['processing_time_seconds']} seconds")
        
        # EXACT SAME RESPONSE FORMAT AS ORIGINAL
        return {
            "success": True,
            "summary": summary,
            "results": results,
            "request_info": {
                "sources_count": len(sources),
                "rules_count": len(rules_dict),
                "cache_enabled": True,
                "optimization_strategy": "llm_flexible_location_validation_and_extraction"
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
    uvicorn.run(app, host="0.0.0.0", port=8235)
