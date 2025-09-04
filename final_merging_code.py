# pip install flask crawl4ai google-genai
from flask import Flask, request, jsonify
import asyncio
import os
import json
from typing import List, Optional, Dict, Any
from google import genai
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter

app = Flask(__name__)

# ---- Helper Functions ----
def clean_query_string(query: Optional[str]) -> str:
    """Remove --gemini from query string if present"""
    if not query:
        return ""
    return query.replace("--gemini", "").strip()

def safe_json_parse(text: str) -> dict:
    """Safely parse JSON response handling various formats"""
    try:
        if not text or not text.strip():
            return {}
        
        data = json.loads(text.strip())
        
        if isinstance(data, list):
            return {"doctor_info": data[0] if data else {}}
        
        if not isinstance(data, dict):
            return {"doctor_info": str(data)}
        
        return data
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return {}
    except Exception as e:
        print(f"Unexpected parsing error: {e}")
        return {}

def build_clean_extraction_prompt(text_content: str) -> str:
    return f"""
Extract doctor information from this text and return a clean JSON structure.

TEXT:
{text_content}

Return ONLY a JSON object with this exact structure:
{{
  "personal_info": {{
    "name": "",
    "experience": "",
    "age": ""
  }},
  "professional_info": {{
    "speciality": "",
    "focus_area": "",
    "languages": []
  }},
  "education": {{
    "degrees": [],
    "certifications": ""
  }},
  "licenses": [],
  "practice_locations": [],
  "contact_info": {{
    "phone": "",
    "email": ""
  }},
  "verification_info": {{
    "source": "",
    "verified": true
  }}
}}

STRICT RULES:
- Do NOT include contact info (phone/email) from websites such as Practo, HexaHealth, hospital portals, or official directories.
- Only include phone/email if explicitly stated as the doctor’s personal/professional contact in the text.
- If not available, set as "NA".
- Always return valid JSON only (no extra text, comments, or explanations).
"""


async def crawl_single_url(url: str) -> str:
    """Crawl a single URL and return cleaned text"""
    try:
        config = CrawlerRunConfig(
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=PruningContentFilter()
            )
        )
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url, config=config)
            if getattr(result, "success", False):
                markdown = getattr(result, "markdown", "")
                return getattr(markdown, "raw_markdown", str(markdown) if markdown else "")
        return ""
    except Exception as e:
        print(f"Error crawling {url}: {str(e)}")
        return ""

def extract_with_gemini_clean(text_content: str, temperature: float = 0.1) -> dict:
    """Extract data with clean structure"""
    try:
        client = genai.Client(api_key="AIzaSyCi8c6ikevd0oA6JC3OrbidXCdAlDmcxYQ")
        prompt = build_clean_extraction_prompt(text_content)
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "temperature": temperature,
            },
        )
        
        response_text = getattr(response, 'text', str(response))
        parsed_data = safe_json_parse(response_text)
        
        return parsed_data if parsed_data else {}
        
    except Exception as e:
        print(f"Error with Gemini extraction: {str(e)}")
        return {}

def llm_strict_speciality_merge(all_profiles: List[dict], query_string: str, temperature: float = 0.1) -> dict:
    """STRICT speciality and comprehensive Indian name matching - only exact matches accepted"""
    
    profiles_text = "DOCTOR PROFILES TO ANALYZE:\n\n"
    
    for i, result in enumerate(all_profiles, 1):
        source_url = result.get('verification_info', {}).get('source', f'Source_{i}')
        profiles_text += f"PROFILE {i} - SOURCE: {source_url}\n"
        profiles_text += json.dumps(result, indent=2)
        profiles_text += "\n\n"

    # Use string concatenation instead of f-string to avoid brace escaping issues
    merge_prompt = """
You are an expert AI medical data analyst specializing in Indian doctor profiles with deep understanding of Indian naming conventions.

SEARCH QUERY: """ + query_string + """

""" + profiles_text + """

COMPREHENSIVE INDIAN NAME MATCHING RULES:

🇮🇳 **INDIAN NAMING CONVENTIONS AWARENESS:**

**Regional Patterns:**
- **North Indian**: First Name + Middle Name + Surname (e.g., "Rahul Kumar Sharma")
- **South Indian**: Village/Initial + Father's Name + Given Name + Caste (e.g., "M. Suresh Kumar Iyer")
- **Tamil**: Often "R. Ramesh" = "Ramesh Raman" (patronymic system)
- **Sikh**: Always includes Singh (male) or Kaur (female) (e.g., "Harpreet Singh Bedi")
- **Marathi**: Surname often comes first in documents (e.g., "Sharma, Rajesh Kumar")

**Name Normalization Steps (MANDATORY):**
1. Remove ALL honorifics: Dr., Prof., Mr., Mrs., Ms., Shri, Smt., etc.
2. Handle punctuation: Remove dots, normalize apostrophes, handle hyphens
3. Normalize spacing: Remove extra spaces, trim whitespace
4. Convert to lowercase for comparison
5. Handle comma-separated formats: "Sharma, Rajesh" → "Rajesh Sharma"
6. Recognize initials: "R.K. Sharma" vs "Rajesh Kumar Sharma"

**Token-Based Matching Rules:**

✅ **ACCEPT These Name Variations:**

1. **Exact Token Match (Different Orders)**
   - "Amar Singh" = "Singh Amar" ✅ (common in Maharashtra documents)
   - "Rajesh Kumar Patel" = "Patel Rajesh Kumar" ✅
   - "Ramesh Gupta" = "Gupta Ramesh" ✅

2. **Initial Expansion/Contraction**
   - "R.K. Sharma" = "Rajesh Kumar Sharma" ✅ (initials match)
   - "A. Singh" = "Amar Singh" ✅ (single initial expansion)
   - "S. Ramesh" = "Subramanian Ramesh" ✅ (South Indian patronymic)

3. **Standard Abbreviations**
   - "Ram Kumar" = "Ramkumar" ✅ (spacing variations)
   - "Raj Kumar" = "Rajkumar" ✅
   - "D'Souza" = "DSouza" ✅ (punctuation normalization)

4. **Sikh Name Variations**
   - "Harpreet Singh Bedi" = "Harpreet Kaur Bedi" ❌ (different gender markers)
   - "Harpreet Singh" = "Singh Harpreet" ✅ (order variation)

5. **Patronymic Variations (South Indian)**
   - "R. Ramesh" where R = Father's initial ✅
   - "Ramesh Raman" = "R. Ramesh" ✅ (if R matches Raman)

6. **Middle Name Flexibility**
   - "Rajesh Sharma" = "Rajesh Kumar Sharma" ✅ (missing middle name)
   - "A.K. Patel" = "Anil Patel" ✅ (one initial missing)

❌ **STRICT REJECTIONS:**

1. **Different Surnames**
   - "Rajesh Sharma" ≠ "Rajesh Verma" ❌ (different surnames)
   - "Singh" ≠ "Sing" ❌ (spelling difference)
  

2. **Different Given Names**
   - "Amar Singh" ≠ "Amaresh Singh" ❌ (different first names)
   - "Raj" ≠ "Rajesh" ❌ (not proven abbreviation)
   - "Krishna" ≠ "Krishnan" ❌ (different names)

3. **Incompatible Token Sets**
   - "Ramesh Kumar" ≠ "Kumar Suresh" ❌ (different given names)
   - "A.B. Patel" ≠ "Anil Sharma" ❌ (surname mismatch)

4. **Gender Marker Conflicts (Sikh)**
   - "Singh" (male) ≠ "Kaur" (female) ❌
   - Must match gender implications

5. **Incomplete Matches**
   - "Rajesh" ≠ "Rajesh Kumar Sharma" ❌ (too incomplete)
   - "R. Kumar" ≠ "Rajesh Patel" ❌ (surname missing)

**Advanced Matching Logic:**

🔍 **Name Comparison Algorithm:**
1. Normalize both names using steps above
2. Tokenize into word lists
3. Check if token sets are compatible:
   - All non-initial tokens must match exactly
   - Initials must be expandable to matching tokens
   - Handle surname-first formats automatically
4. Verify no conflicting tokens exist
5. Accept only if confidence is HIGH

**Cultural Context Awareness:**
- Women's names may include husband's name after marriage
- Caste names (Sharma, Patel, Iyer) are typically surnames
- Regional suffixes (Kumar, Singh, Kaur) have specific meanings
- Transliteration variations are common but must be handled carefully

**CRITICAL TEST CASES:**
- "Dr. Amaresh Singh" vs "Singh Amar" = ❌ REJECT (different first names)
- "Dr. Amaresh Singh" vs "Singh Amaresh" = ✅ ACCEPT (same tokens, reordered)
- "R.K. Sharma" vs "Rajesh Kumar Sharma" = ✅ ACCEPT (initials match)
- "Dr. Raj Patel" vs "Rajesh Patel" = ❌ REJECT (Raj ≠ Rajesh without proof)

🔒 **SPECIALITY MATCHING - ULTRA STRICT:**
- General Physician = General Practice = General Practitioner = GP = Physician = Family Medicine 
- Paediatrics = Pediatrics = Child Specialist = Paediatrician ONLY  
- Cardiology = Cardiologist = Heart Specialist = Cardiac Medicine ONLY
- Gastroenterology = Gastroenterologist = GI Specialist ONLY
- NO cross-speciality matching allowed
- NO flexible interpretation

🏥 **LOCATION MATCHING:**
- Handle city aliases: Bangalore = Bengaluru, Mumbai = Bombay, Chennai = Madras
- State consistency required
- Geographic region should be logical

📋 **OUTPUT REQUIREMENTS:**
- Every profile MUST be categorized as accepted OR rejected
- Include detailed "reason_for_rejection" for ALL rejected profiles
- Accepted profiles must pass BOTH name AND speciality checks
- Total profiles = accepted + rejected count

**CRITICAL EXAMPLES FOR YOUR REFERENCE:**
1. Query: "Dr. Amaresh Singh General Physician" 
   - "Singh Amar, General Physician" = ❌ REJECT ("Amar" ≠ "Amaresh")
   - "Singh Amaresh, General Practice" = ✅ ACCEPT (same tokens + valid speciality)
   
2. Query: "Dr. R.K. Patel Cardiology"
   - "Rajesh Kumar Patel, Cardiologist" = ✅ ACCEPT (initials match + valid speciality)
   - "Raj Patel, Cardiology" = ❌ REJECT ("R" doesn't clearly = "Raj")

Return EXACTLY this JSON structure:
{
    "accepted_profiles": [
        // ONLY profiles passing ALL strict criteria
    ],
    "rejected_profiles": [
        {
            "profile": {...},
            "reason_for_rejection": "detailed_reason_here"
        }
    ],
    "merged_profile": {
        "Cleaned_Query": \"""" + query_string + """\",
        "Primary_Email": "",
        "Primary_Phone": "",
        "Degree_1": "",
        "Degree_2": "",
        "Degree_3": "",
        "Degree_4": "",
        "Degree_5": "",
        "License_1_Issue_Year": "",
        "License_1_Number": "",
        "License_1_Type": "",
        "License_1_Body": "",
        "License_2_Number": "",
        "License_2_Type": "",
        "License_2_Body": "",
        "Experience_Years": "",
        "First_Name": "",
        "Full_Name": "",
        "Last_Name": "",
        "Focus_Area": "",
        "Primary_Speciality": "",
        "Secondary_Speciality": "",
        "Languages_Spoken_1": "",
        "Data_Source": "",
        "Practice_Address": "",
        "Practice_City": "",
        "Practice_Pincode": "",
        "Practice_State": "",
        "Practice_Country": "",
        "Practice_Latitude": "",
        "Practice_Longitude": "",
        "Practice_Plus_Code": "",
        "Practice_HCO_Name": "",
        "Practice_HCO_Speciality": "",
        "Practice_HCO_Type": "",
        "Practice_Phone": "",
        "Practice_Consultation_Fee": "",
        "Practice_Timing": "",
        "Practice_Website": ""
    },
    "accepted_count": 0,
    "rejected_count": 0,
    "total_profiles": 0
}

FINAL INSTRUCTIONS:
- Apply MAXIMUM strictness for name matching using Indian conventions
- Apply ULTRA strictness for speciality matching  
- Provide detailed rejection reasons
- Better to reject uncertain matches than accept false positives
- Focus on precision over recall

Return ONLY JSON, no explanatory text.
"""

    try:
        client = genai.Client(api_key="AIzaSyCi8c6ikevd0oA6JC3OrbidXCdAlDmcxYQ")
        
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=merge_prompt,
            config={
                "response_mime_type": "application/json",
                "temperature": temperature,
            },
        )
        
        response_text = getattr(response, 'text', str(response))
        result = safe_json_parse(response_text)
        
        # Ensure required structure
        if 'accepted_profiles' not in result:
            result['accepted_profiles'] = []
        if 'rejected_profiles' not in result:
            result['rejected_profiles'] = []
        if 'merged_profile' not in result:
            result['merged_profile'] = {}
        
        # Add counts and validation
        result['accepted_count'] = len(result['accepted_profiles'])
        result['rejected_count'] = len(result['rejected_profiles'])
        result['total_profiles'] = len(all_profiles)
        
        print(f"🔒 COMPREHENSIVE INDIAN NAME MATCHING: {result['accepted_count']} accepted, {result['rejected_count']} rejected")
        
        return result
        
    except Exception as e:
        print(f"LLM processing failed: {e}")
        return {
            "accepted_profiles": [],
            "rejected_profiles": [{"profile": profile, "reason_for_rejection": "processing_error"} for profile in all_profiles],
            "merged_profile": {"Cleaned_Query": query_string},
            "accepted_count": 0,
            "rejected_count": len(all_profiles),
            "total_profiles": len(all_profiles)
        }



# def llm_strict_speciality_merge(all_profiles: List[dict], query_string: str, temperature: float = 0.1) -> dict:
#     """STRICT speciality matching - only exact matches accepted"""
    
#     profiles_text = "DOCTOR PROFILES TO ANALYZE:\n\n"
    
#     for i, result in enumerate(all_profiles, 1):
#         source_url = result.get('verification_info', {}).get('source', f'Source_{i}')
#         profiles_text += f"PROFILE {i} - SOURCE: {source_url}\n"
#         profiles_text += json.dumps(result, indent=2)
#         profiles_text += "\n\n"

# #     merge_prompt = f"""
# # You are an expert medical data analyst. Analyze these doctor profiles with STRICT speciality matching for the query: {query_string}

# # {profiles_text}

# # CRITICAL STRICT MATCHING RULES:

# # 🔒 **SPECIALITY MATCHING - MUST BE EXACT:**
# # - If query specifies "General Physician" → ONLY accept profiles with exactly "General Physician" or "General Practice" or "Family Medicine"
# # - If query specifies "Paediatrics" → ONLY accept profiles with exactly "Paediatrics" or "Pediatrics" 
# # - If query specifies "Cardiology" → ONLY accept profiles with exactly "Cardiology" or "Cardiologist"
# # - DO NOT accept cross-speciality matches (e.g., Paediatrics for General Physician query)
# # - DO NOT be flexible with specialities - be STRICT and EXACT

# # 📝 **ACCEPTABLE SPECIALITY SYNONYMS ONLY:**
# # - General Physician = General Practice = Family Medicine = Internal Medicine
# # - Paediatrics = Pediatrics = Child Specialist
# # - Gastroenterology = Gastroenterologist = GI Specialist
# # - Cardiology = Cardiologist = Heart Specialist
# # - Use ONLY these exact synonyms, nothing else

# # ❌ **STRICT REJECTION CRITERIA:**
# # - Different medical specialities (Paediatrics ≠ General Physician)
# # - Different medical systems (MBBS ≠ BAMS ≠ BHMS) 
# # - Different doctor names (even similar ones like Sharma ≠ Verma)
# # - Unrelated specialities regardless of location match

# # ✅ **ACCEPTANCE CRITERIA:**
# # - Name must match closely (handle Dr./Prof. variations)
# # - Speciality must be EXACTLY one of the acceptable synonyms
# # - Location can be flexible (city aliases OK)

# # Return EXACTLY this JSON structure:
# # {{
# #     "accepted_profiles": [
# #         // ONLY profiles with EXACT speciality match
# #     ],
# #     "rejected_profiles": [
# #         // All profiles that don't match EXACTLY
# #     ],
# #     "merged_profile": {{
# #         "Cleaned_Query": "{query_string}",
# #         "Primary_Email": "",
# #         "Primary_Phone": "",
# #         "Degree_1": "",
# #         "Degree_2": "",
# #         "Degree_3": "",
# #         "Degree_4": "",
# #         "Degree_5": "",
# #         "License_1_Issue_Year": "",
# #         "License_1_Number": "",
# #         "License_1_Type": "",
# #         "License_1_Body": "",
# #         "License_2_Number": "",
# #         "License_2_Type": "",
# #         "License_2_Body": "",
# #         "Experience_Years": "",
# #         "First_Name": "",
# #         "Full_Name": "",
# #         "Last_Name": "",
# #         "Focus_Area": "",
# #         "Primary_Speciality": "",
# #         "Secondary_Speciality": "",
# #         "Languages_Spoken_1": "",
# #         "Data_Source": "",
# #         "Practice_Address": "",
# #         "Practice_City": "",
# #         "Practice_Pincode": "",
# #         "Practice_State": "",
# #         "Practice_Country": "",
# #         "Practice_Latitude": "",
# #         "Practice_Longitude": "",
# #         "Practice_Plus_Code": "",
# #         "Practice_HCO_Name": "",
# #         "Practice_HCO_Speciality": "",
# #         "Practice_HCO_Type": "",
# #         "Practice_Phone": "",
# #         "Practice_Consultation_Fee": "",
# #         "Practice_Timing": "",
# #         "Practice_Website": ""
# #     }},
# #     "accepted_count": 0,
# #     "rejected_count": 0
# # }}

# # IMPORTANT: Be STRICT with speciality matching. DO NOT accept profiles with different specialities even if names and locations match.
# # Fill merged_profile with consolidated data from accepted profiles only. Use "NA" for missing fields.
# # Return ONLY JSON, no additional text.
# # """
#     merge_prompt = f"""
# You are an expert medical data analyst. Analyze these doctor profiles with STRICT speciality and name matching for the query: {query_string}

# {profiles_text}

# CRITICAL STRICT MATCHING RULES:

# 🔒 **SPECIALITY MATCHING - MUST BE EXACT:**
# - If query specifies "General Physician" → ONLY accept profiles with exactly "General Physician" or "General Practice" or "Family Medicine" or "Internal Medicine"
# - If query specifies "Paediatrics" → ONLY accept profiles with exactly "Paediatrics" or "Pediatrics" or "Child Specialist"
# - If query specifies "Cardiology" → ONLY accept profiles with exactly "Cardiology" or "Cardiologist" or "Heart Specialist"
# - If query specifies "Gastroenterology" → ONLY accept profiles with exactly "Gastroenterology" or "Gastroenterologist" or "GI Specialist"
# - DO NOT accept cross-speciality matches (e.g., Paediatrics for General Physician query)
# - DO NOT be flexible with specialities - be STRICT and EXACT
# - Use ONLY the exact synonyms listed above for acceptance

# 👤 **NAME MATCHING RULES (Real-world Indian Scenarios — apply BEFORE speciality check where name matching is required):**

# Normalization steps (apply before any comparisons):
# - Remove honorifics/prefixes: "Dr.", "Dr", "Prof.", "Dr (Mrs.)", etc.
# - Trim leading/trailing whitespace and collapse multiple spaces.
# - Remove trailing punctuation and normalize internal punctuation (e.g., "D'Souza" → "DSouza").
# - Remove diacritics. Do NOT reorder tokens during normalization unless a comma/explicit delimiter indicates last-first.

# ✅ **Allowed / ACCEPT matches (real-world aware):**
# 1. **Exact token / abbreviation mapping**  
#    - "Rahul Kumar Patel" == "Rahul K Patel" == "R K Patel" == "Rahul Patel"
# 2. **Initials vs full-name (initial-first or name-first) — ACCEPT**  
#    - "K. Ramesh" == "Ramesh K"  
#    - "R K Sharma" == "Rajesh Kumar Sharma"
# 3. **Initial expansion — ACCEPT ambiguous initials**  
#    - "A K Rao" == "Arun Karthik Rao" → ACCEPT  
#    - "A K Rao" == "Anil Kumar Rao" → ACCEPT  
#    - *Rationale:* in real-world data initial-based records often map to multiple valid full-name expansions; treat as match when tokens can plausibly correspond to initials.
# 4. **Surname-first / token flip — ACCEPT if token set matches exactly**  
#    - "Amar Singh" == "Singh Amar"  
#    - "Ramesh Kumar" == "Kumar Ramesh"  
#    - *Rationale:* many portals use surname-first ordering; accept when tokens (ignoring order) are the same.
# 5. **Comma-delimited explicit last-first**  
#    - "Singh, Amar" → normalize to "Amar Singh" and match.
# 6. **Father's-initial / South-Indian convention — ACCEPT when initial maps to family/father token**  
#    - "S. Ramesh" == "Subramanian Ramesh"
# 7. **Minor punctuation/spacing variants allowed**  
#    - "Rajiv-Kumar Singh" == "Rajiv Kumar Singh"  
#    - "D'Souza" == "DSouza"

# ❌ **Strict Rejections (always treat as different persons):**
# - Completely different surnames → "Ramesh Sharma" ≠ "Ramesh Verma"
# - Different given names (different token strings) → "Amar Singh" ≠ "Amaresh Singh"
# - Mononyms must match exactly → "Rajesh" ≠ "Raj"
# - Phonetic/sound-alike matches are NOT enough → "Krishna" ≠ "Krishnan", "Sharma" ≠ "Sharmma"
# - If initials are present but DO NOT correspond to any plausible mapping with other tokens (no deterministic or plausible expansion), then REJECT.
#   - Note: per acceptance rules above, many ambiguous initials should still be ACCEPTED if tokens plausibly map; only reject when mapping is impossible.

# 🔑 **Simplified Deterministic Acceptance Rule (apply in order):**
# 1. Normalize both names (remove honorifics/punctuation/diacritics).
# 2. If the set of name tokens (ignoring order) is identical across profiles → ACCEPT.
# 3. Else, if initials exist on one/both sides and can plausibly expand to match tokens on the other side (including ambiguous but plausible expansions) → ACCEPT.
# 4. Else, if comma-delimited explicit last-first representation maps tokens correctly → ACCEPT.
# 5. Otherwise → REJECT.

# ❌ **STRICT REJECTION CRITERIA (summary):**
# - Different medical specialities (Paediatrics ≠ General Physician)
# - Different medical systems (MBBS ≠ BAMS ≠ BHMS)
# - Different doctor names per the name-matching rules above
# - Unrelated specialities regardless of location match

# ✅ **ACCEPTANCE CRITERIA:**
# - Name must match per the NAME MATCHING RULES above
# - Speciality must be EXACTLY one of the acceptable synonyms
# - Location can be flexible (city aliases OK)

# OUTPUT RULES (MANDATORY):
# - Every input profile MUST appear in either `accepted_profiles` OR `rejected_profiles` — no profile may be left uncategorized.
# - `rejected_profiles` MUST include a `"reason_for_rejection"` field describing *why* the profile was rejected (e.g., "speciality_mismatch", "name_mismatch", "ambiguous_initials_unresolvable", "different_medical_system", etc.).
# - `accepted_count + rejected_count MUST = total_profiles`.
# - When consolidating `merged_profile`, use accepted profiles only. If multiple accepted profiles supply the same field, prefer the most authoritative source (order of sources may be provided in {profiles_text} — otherwise pick the most complete non-empty value). For any missing field use "NA".
# - Provide `accepted_count`, `rejected_count`, and `total_profiles` filled correctly.

# Return EXACTLY this JSON structure:
# {{
#     "accepted_profiles": [
#         // ONLY profiles accepted (include original profile metadata; include a "source" field if available)
#     ],
#     "rejected_profiles": [
#         // Each rejected profile must be an object: { "profile": <original_profile_data>, "reason_for_rejection": "<explicit_reason>" }
#     ],
#     "merged_profile": {{
#         "Cleaned_Query": "{query_string}",
#         "Primary_Email": "",
#         "Primary_Phone": "",
#         "Degree_1": "",
#         "Degree_2": "",
#         "Degree_3": "",
#         "Degree_4": "",
#         "Degree_5": "",
#         "License_1_Issue_Year": "",
#         "License_1_Number": "",
#         "License_1_Type": "",
#         "License_1_Body": "",
#         "License_2_Number": "",
#         "License_2_Type": "",
#         "License_2_Body": "",
#         "Experience_Years": "",
#         "First_Name": "",
#         "Full_Name": "",
#         "Last_Name": "",
#         "Focus_Area": "",
#         "Primary_Speciality": "",
#         "Secondary_Speciality": "",
#         "Languages_Spoken_1": "",
#         "Data_Source": "",
#         "Practice_Address": "",
#         "Practice_City": "",
#         "Practice_Pincode": "",
#         "Practice_State": "",
#         "Practice_Country": "",
#         "Practice_Latitude": "",
#         "Practice_Longitude": "",
#         "Practice_Plus_Code": "",
#         "Practice_HCO_Name": "",
#         "Practice_HCO_Speciality": "",
#         "Practice_HCO_Type": "",
#         "Practice_Phone": "",
#         "Practice_Consultation_Fee": "",
#         "Practice_Timing": "",
#         "Practice_Website": ""
#     }},
#     "accepted_count": 0,
#     "rejected_count": 0,
#     "total_profiles": 0
# }}

# IMPORTANT:
# - Be STRICT with speciality matching; be REAL-WORLD PRACTICAL with name matching per the rules above.
# - Fill merged_profile only from accepted profiles. Use "NA" for any missing values.
# - Always include detailed `reason_for_rejection` for rejected profiles.
# - Return ONLY JSON, no additional text.
# """

    

#     try:
#         client = genai.Client(api_key="AIzaSyCi8c6ikevd0oA6JC3OrbidXCdAlDmcxYQ")
        
#         response = client.models.generate_content(
#             model="gemini-2.0-flash",
#             contents=merge_prompt,
#             config={
#                 "response_mime_type": "application/json",
#                 "temperature": temperature,
#             },
#         )
        
#         response_text = getattr(response, 'text', str(response))
#         result = safe_json_parse(response_text)
        
#         # Ensure required structure
#         if 'accepted_profiles' not in result:
#             result['accepted_profiles'] = []
#         if 'rejected_profiles' not in result:
#             result['rejected_profiles'] = []
#         if 'merged_profile' not in result:
#             result['merged_profile'] = {}
        
#         # Add counts
#         result['accepted_count'] = len(result['accepted_profiles'])
#         result['rejected_count'] = len(result['rejected_profiles'])
        
#         print(f"🔒 STRICT matching completed: {result['accepted_count']} accepted, {result['rejected_count']} rejected")
        
#         return result
        
#     except Exception as e:
#         print(f"LLM processing failed: {e}")
#         return {
#             "accepted_profiles": [],
#             "rejected_profiles": all_profiles,
#             "merged_profile": {"Cleaned_Query": query_string},
#             "accepted_count": 0,
#             "rejected_count": len(all_profiles)
#         }

async def process_urls_strict(urls: List[str], query_string: str, temperature: float = 0.1) -> dict:
    """Process URLs with STRICT speciality matching"""
    print(f"🚀 Processing {len(urls)} URLs with STRICT speciality matching...")
    
    # Crawl URLs
    crawled_texts = await asyncio.gather(*[crawl_single_url(url) for url in urls])

    # Extract with clean structure
    all_profiles = []
    successful_extractions = 0
    
    for url, text in zip(urls, crawled_texts):
        if text.strip():
            print(f"📄 Extracting from: {url}")
            try:
                data = extract_with_gemini_clean(text, temperature)
                if data:
                    # Add source info
                    if 'verification_info' not in data:
                        data['verification_info'] = {}
                    data['verification_info']['source'] = url
                    all_profiles.append(data)
                    successful_extractions += 1
                    print(f"✅ Extraction complete: {url}")
                else:
                    print(f"⚠️ No valid data: {url}")
            except Exception as e:
                print(f"⚠️ Error extracting from {url}: {e}")
        else:
            print(f"⚠️ No content: {url}")

    if not all_profiles:
        return {
            "accepted_profiles": [],
            "rejected_profiles": [],
            "merged_profile": {},
            "accepted_count": 0,
            "rejected_count": 0
        }

    print(f"📊 Extracted from {successful_extractions}/{len(urls)} URLs")
    
    # STRICT speciality matching merge
    result = llm_strict_speciality_merge(all_profiles, query_string, temperature)
    return result

# ---- Flask Routes ----
@app.route('/extract-doctor', methods=['POST'])
def extract_doctor_info():
    """Extract doctor information with STRICT speciality matching"""
    try:
        request_data = request.get_json()
        
        if not request_data or 'urls' not in request_data:
            return jsonify({
                "error": "Missing 'urls' field in request body",
                "status": "error"
            }), 400
        
        urls = request_data['urls']
        raw_query = request_data.get('query', '')
        temperature = request_data.get('temperature', 0.1)  # Lower temperature for strict matching
        
        clean_query = clean_query_string(raw_query)
        
        # Validate inputs
        if not isinstance(urls, list) or len(urls) == 0:
            return jsonify({
                "error": "URLs must be a non-empty list",
                "status": "error"
            }), 400
        
        if len(urls) > 10:
            return jsonify({
                "error": "Maximum 10 URLs allowed per request",
                "status": "error"
            }), 400
        
        for url in urls:
            if not isinstance(url, str) or not (url.startswith('http://') or url.startswith('https://')):
                return jsonify({
                    "error": f"Invalid URL: {url}",
                    "status": "error"
                }), 400
        
        print(f"🔒 STRICT SPECIALITY MATCHING")
        print(f"📊 URLs: {len(urls)}")
        print(f"🔍 Query: '{clean_query}'")
        print(f"🌡️ Temperature: {temperature}")
        print("⚠️ STRICT MODE: Only exact speciality matches will be accepted")
        
        # Process with STRICT matching
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(process_urls_strict(urls, clean_query, temperature))
        finally:
            loop.close()
        
        # Calculate fill rate
        merged_profile = result.get("merged_profile", {})
        total_fields = len(merged_profile) if merged_profile else 0
        filled_fields = 0
        
        if merged_profile:
            for v in merged_profile.values():
                if v and str(v) not in ["NA", "Not specified", "", "null"]:
                    filled_fields += 1
        
        fill_rate = (filled_fields / total_fields * 100) if total_fields > 0 else 0
        
        print(f"🔒 STRICT PROCESSING COMPLETED - Fill Rate: {fill_rate:.1f}%")
        
        return jsonify({
            "status": "success",
            "data": merged_profile,
            "accepted_profiles": result.get("accepted_profiles", []),
            "rejected_profiles": result.get("rejected_profiles", []),
            "accepted_count": result.get("accepted_count", 0),
            "rejected_count": result.get("rejected_count", 0),
            "total_profiles": result.get("accepted_count", 0) + result.get("rejected_count", 0),
            "urls_processed": len(urls),
            "original_query": raw_query,
            "cleaned_query": clean_query,
            "temperature_used": temperature,
            "processing_method": "STRICT speciality matching - exact matches only",
            "fill_rate_percentage": round(fill_rate, 1),
            "filled_fields": filled_fields,
            "total_fields": total_fields,
            "matching_mode": "STRICT - only exact speciality matches accepted",
            "message": f"STRICT processing completed - {result.get('accepted_count', 0)} accepted, {result.get('rejected_count', 0)} rejected - {fill_rate:.1f}% fields filled"
        })
        
    except Exception as e:
        print(f"❌ API Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "message": "STRICT Speciality Matching Doctor API",
        "version": "13.0.0-StrictSpeciality"
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "STRICT Speciality Matching Doctor API",
        "version": "13.0.0-StrictSpeciality", 
        "description": "Doctor information extraction with STRICT speciality matching - only exact matches accepted",
        "strict_matching_rules": {
            "General Physician": ["General Physician", "General Practice", "Family Medicine", "Internal Medicine"],
            "Paediatrics": ["Paediatrics", "Pediatrics", "Child Specialist"],
            "Cardiology": ["Cardiology", "Cardiologist", "Heart Specialist"],
            "Gastroenterology": ["Gastroenterology", "Gastroenterologist", "GI Specialist"],
            "note": "Cross-speciality matches are REJECTED (e.g., Paediatrics for General Physician query)"
        },
        "key_changes": [
            "STRICT speciality matching - no cross-speciality acceptance",
            "Only exact synonyms allowed for each speciality",
            "Lower default temperature (0.1) for precise matching",
            "Clear rejection of different specialities",
            "Paediatrics will NOT be accepted for General Physician queries"
        ]
    })

if __name__ == '__main__':
    print("🚀 Starting STRICT Speciality Matching Doctor API v13.0.0")
    print("🔒 Focus: STRICT speciality matching - only exact matches accepted")
    print("⚠️ CRITICAL CHANGE:")
    print("   • General Physician queries will REJECT Paediatrics profiles")
    print("   • Only exact speciality matches or approved synonyms accepted")
    print("   • Cross-speciality matching is DISABLED")
    print()
    print("🎯 Strict Matching Rules:")
    print("   • General Physician = General Practice = Family Medicine = Internal Medicine")
    print("   • Paediatrics = Pediatrics = Child Specialist")
    print("   • Cardiology = Cardiologist = Heart Specialist")
    print("   • NO cross-speciality acceptance (Paediatrics ≠ General Physician)")
    print()
    print("Available endpoints:")
    print("  POST /extract-doctor - Extract with STRICT speciality matching")
    print("  GET /health - Health check")
    print("  GET / - API documentation")
    
    app.run(debug=True, host='0.0.0.0', port=5558)
