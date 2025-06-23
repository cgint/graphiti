import asyncio
import json
import os
from time import time
from typing import Any, Optional

from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig, GenerateContentResponse, FinishReason
from google.api_core.exceptions import ResourceExhausted, GoogleAPICallError
from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.config import LLMConfig, ModelSize
from graphiti_core.prompts.models import Message
from pydantic import BaseModel, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

GCP_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
GCP_LOCATION = os.getenv("GCP_LOCATION", "global") # suggested by google to use 'global' for maximum number of requests

class StopNotEndException(Exception):
    """Exception raised when the generation is stopped but not ended."""
    pass

class VertexAIClient(LLMClient):
    """
    A client for interacting with Google's Vertex AI models using the google-genai SDK.
    
    Features:
    - Concurrency limiting to prevent rate limit errors
    - Queue statistics to monitor waiting requests
    - Automatic retry with exponential backoff
    - Structured output support
    
    Environment Variables:
    - GOOGLE_CLOUD_PROJECT: GCP project ID (required)
    - GCP_LOCATION: GCP region (default: global) # suggested by google to use 'global' for maximum number of requests
    - VERTEX_AI_MAX_CONCURRENT: Max concurrent requests (default: 3)
    - GOOGLE_GENAI_USE_VERTEXAI: Must be "True"
    """

    # Class-level concurrency control and queue statistics
    _semaphore: asyncio.Semaphore | None = None
    _max_concurrent = int(os.getenv("VERTEX_AI_MAX_CONCURRENT", "3"))
    _waiting_count = 0  # Track requests waiting for semaphore
    _queue_lock = asyncio.Lock()  # Protect the waiting counter

    def __init__(
        self,
        config: LLMConfig,
        project_id: str | None = GCP_PROJECT_ID,
        location: str | None = GCP_LOCATION,
    ):
        super().__init__(config)
        
        # Validate required environment
        if not project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT environment variable not set.")
        if not location:
            raise ValueError("GCP_LOCATION environment variable not set.")

        # Initialize class-level semaphore (shared across all instances)
        if VertexAIClient._semaphore is None:
            VertexAIClient._semaphore = asyncio.Semaphore(self._max_concurrent)

        # Configure Vertex AI environment
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
        os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
        os.environ["GOOGLE_CLOUD_LOCATION"] = location
        
        # Initialize client
        self.client = genai.Client()
        self.model_name = self.config.model or "gemini-2.5-flash-lite-preview-06-17"

        print("🚀 VertexAI Client initialized:")
        print(f"   Model: {self.model_name}")
        print(f"   Max concurrent requests: {self._max_concurrent}")
        print("   Retry strategy: 5 attempts with exponential backoff")

    def _is_gemini_2_5_model(self) -> bool:
        """Check if model supports thinking budget (Gemini 2.5+ models)."""
        return "gemini-2.5" in (self.model_name or "").lower()

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Detect rate limiting errors that should trigger retry logic."""
        if isinstance(error, ResourceExhausted):
            return True
            
        if isinstance(error, GoogleAPICallError):
            error_str = str(error).lower()
            rate_limit_indicators = [
                'rate limit', 'quota exceeded', 'too many requests', 
                'resource exhausted', '429', 'quotaexceeded'
            ]
            return any(indicator in error_str for indicator in rate_limit_indicators)
            
        error_message = str(error).lower()
        return any(pattern in error_message for pattern in [
            'rate limit', 'quota', '429', 'too many requests'
        ])

    def _extract_response_text(self, response: GenerateContentResponse) -> str:
        """Extract text content from Vertex AI response."""
        try:
            # Primary: direct text access (modern SDK)
            if hasattr(response, 'text') and response.text:
                return str(response.text)
            
            # Fallback: extract from candidates structure
            if (hasattr(response, 'candidates') and response.candidates and
                len(response.candidates) > 0):
                candidate = response.candidates[0]
                if (hasattr(candidate, 'content') and candidate.content and
                    hasattr(candidate.content, 'parts') and candidate.content.parts):
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            return str(part.text)
            
            return ""
            
        except Exception as e:
            print(f"⚠️ Could not extract text from response: {e}")
            return ""

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = 65500,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, Any]:
        """
        Generate response with concurrency limiting to prevent rate limits.
        
        The semaphore ensures only N requests run simultaneously across all instances.
        Queue statistics show both active and waiting requests.
        """
        
        # 🔒 CONCURRENCY CONTROL: Wait for available slot
        # At this point _semaphore is guaranteed to be initialized
        semaphore = self._semaphore
        if semaphore is None:
            raise RuntimeError("Semaphore not initialized")
        
        # Check if we need to wait (before entering queue)
        will_wait = semaphore._value == 0
        if will_wait:
            # Atomically increment waiting counter
            async with VertexAIClient._queue_lock:
                VertexAIClient._waiting_count += 1
                active_requests = self._max_concurrent - semaphore._value
                waiting_requests = VertexAIClient._waiting_count
                print(f"⏳ Waiting for slot... ({active_requests}/{self._max_concurrent} active, {waiting_requests} waiting)")
        
        try:
            async with semaphore:
                # If we were waiting, decrement the counter
                if will_wait:
                    async with VertexAIClient._queue_lock:
                        VertexAIClient._waiting_count -= 1
                
                active_requests = self._max_concurrent - semaphore._value
                waiting_requests = VertexAIClient._waiting_count
                print(f"🔒 Request slot acquired ({active_requests}/{self._max_concurrent} active, {waiting_requests} waiting)")
                
                try:
                    return await self._do_generate(messages, response_model, max_tokens)
                finally:
                    waiting_requests = VertexAIClient._waiting_count
                    print(f"🔓 Request slot released ({waiting_requests} still waiting)")
        except asyncio.CancelledError:
            # If we were waiting and the task gets cancelled, decrement counter
            if will_wait:
                async with VertexAIClient._queue_lock:
                    VertexAIClient._waiting_count = max(0, VertexAIClient._waiting_count - 1)
            raise
        except Exception:
            # If we were waiting and an exception occurs, decrement counter
            if will_wait:
                async with VertexAIClient._queue_lock:
                    VertexAIClient._waiting_count = max(0, VertexAIClient._waiting_count - 1)
            raise

    async def _do_generate(
        self,
        messages: list[Message], 
        response_model: type[BaseModel] | None,
        max_tokens: int
    ) -> dict[str, Any]:
        """Handle the actual generation logic (separated for cleaner code)."""
        
        # Prepare request format
        contents = []
        system_instruction = None
        
        for msg in messages:
            if msg.role == "user":
                contents.append({"role": "user", "parts": [{"text": msg.content}]})
            elif msg.role == "assistant":
                contents.append({"role": "model", "parts": [{"text": msg.content}]})
            elif msg.role == "system":
                system_instruction = msg.content

        # Build configuration
        config_params: dict[str, Any] = {
            "temperature": self.config.temperature,
            "max_output_tokens": max_tokens
        }

        if system_instruction:
            config_params["system_instruction"] = system_instruction

        if self._is_gemini_2_5_model():
            config_params["thinking_config"] = ThinkingConfig(
                thinking_budget=0, 
                include_thoughts=False
            )

        if response_model:
            config_params["response_mime_type"] = "application/json"
            config_params["response_schema"] = response_model.model_json_schema()
            print(f"📝 Using structured JSON output: {response_model.__name__}")
            
        config = GenerateContentConfig(**config_params)

        # Generate with retry logic
        @retry(
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=2, min=1, max=30),
            retry=retry_if_exception_type((ResourceExhausted, GoogleAPICallError, StopNotEndException))
        )
        def generate_with_retry() -> str:
            """Sync generation with retry logic."""
            try:
                print(f"🤖 Generating with {self.model_name}...")
                start_time = time()
                
                response: GenerateContentResponse = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config
                )
                token_dict = {}
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    token_dict = {
                        "prompt": response.usage_metadata.prompt_token_count,
                        "cached_content": response.usage_metadata.cached_content_token_count,
                        "candidates": response.usage_metadata.candidates_token_count,
                        "total": response.usage_metadata.total_token_count,
                        "thoughts": response.usage_metadata.thoughts_token_count
                    }

                duration = time() - start_time

                if (hasattr(response, 'prompt_feedback') and 
                    response.prompt_feedback and 
                    hasattr(response.prompt_feedback, 'block_reason') and
                    response.prompt_feedback.block_reason):
                    raise RuntimeError(f"Blocked by safety filters: {response.prompt_feedback.block_reason} after {duration:.2f} seconds with token usage: {token_dict} for request: {str(contents)[:500]}")
                
                if (hasattr(response, 'candidates') and response.candidates and 
                    len(response.candidates) > 0):
                    candidate = response.candidates[0]
                    if (hasattr(candidate, 'finish_reason') and 
                        candidate.finish_reason != FinishReason.STOP):
                        print(f"🔢 Unexpected finish reason: {candidate.finish_reason} after {duration:.2f} seconds with token usage: {token_dict}\n  === request: {str(contents)}\n  === response: {str(response)}")
                        raise StopNotEndException()

                # Extract response
                response_text = self._extract_response_text(response)
                if not response_text:
                    raise RuntimeError("No response text extracted")

                print(f"🔢 Successfully generated {len(response_text)} characters in {duration:.2f} seconds using {self.model_name}. Tokens: {token_dict}\n"
                      f"  - Preview: {response_text[:250]}...")
                
                return response_text
                
            except (ResourceExhausted, GoogleAPICallError) as e:
                if self._is_rate_limit_error(e):
                    print(f"🔄 Rate limit detected, will retry: {e}")
                    raise
                else:
                    print(f"❌ Non-retryable API error: {e}")
                    return ""
                    
            except Exception as e:
                print(f"❌ Generation error: {e}")
                if self._is_rate_limit_error(e):
                    raise ResourceExhausted(f"Potential rate limit: {e}")
                return ""

        # Execute in thread pool (Google client is sync)
        loop = asyncio.get_running_loop()
        response_text = await loop.run_in_executor(None, generate_with_retry)

        # Handle structured output
        if response_model and response_text:
            try:
                parsed = json.loads(response_text)
                response_model.model_validate(parsed)
                print("✅ Structured response validated")
                # Ensure we return dict[str, Any] as expected
                if isinstance(parsed, dict):
                    return parsed
                else:
                    return {"content": str(parsed)}
            except (json.JSONDecodeError, ValidationError) as e:
                print(f"⚠️ JSON validation failed: {e}")
                return {"content": response_text}
        
        return {"content": response_text} 