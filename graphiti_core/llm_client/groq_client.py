"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import logging
import re
import typing
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import groq
    from groq import AsyncGroq
    from groq.types.chat import ChatCompletionMessageParam
else:
    try:
        import groq
        from groq import AsyncGroq
        from groq.types.chat import ChatCompletionMessageParam
    except ImportError:
        raise ImportError(
            'groq is required for GroqClient. Install it with: pip install graphiti-core[groq]'
        ) from None
from pydantic import BaseModel

from ..prompts.models import Message
from .client import LLMClient, MULTILINGUAL_EXTRACTION_RESPONSES
from .config import LLMConfig, ModelSize
from .errors import RateLimitError, RefusalError

logger = logging.getLogger(__name__)

# Updated to latest Groq models as of 2025
DEFAULT_MODEL = 'llama-3.3-70b-versatile'
DEFAULT_SMALL_MODEL = 'llama-3.1-8b-instant'
DEFAULT_MAX_TOKENS = 2048
MAX_RETRIES = 2

# Models that support structured outputs (json_schema)
STRUCTURED_OUTPUT_MODELS = {
    'moonshotai/kimi-k2-instruct-0905',
    'openai/gpt-oss-20b',
    'openai/gpt-oss-120b',
    'meta-llama/llama-4-maverick-17b-16e-instruct',
    'meta-llama/llama-4-scout-17b-16e-instruct',
}


class GroqClient(LLMClient):
    """Enhanced Groq client with OpenAI-level capabilities.

    Features:
    - Comprehensive error handling
    - Retry logic with exponential backoff
    - Model size selection
    - Input cleaning and validation
    - Multilingual support
    - Timeout configuration
    """

    def __init__(self, config: LLMConfig | None = None, cache: bool = False):
        if config is None:
            config = LLMConfig(max_tokens=DEFAULT_MAX_TOKENS)
        elif config.max_tokens is None:
            config.max_tokens = DEFAULT_MAX_TOKENS
        super().__init__(config, cache)

        # Initialize Groq client with timeout and retry configuration
        self.client = AsyncGroq(
            api_key=config.api_key,
            max_retries=MAX_RETRIES,
            timeout=30.0  # Default 30 second timeout
        )

    def _get_model_for_size(self, model_size: ModelSize) -> str:
        """Get the appropriate Groq model based on the requested size."""
        if model_size == ModelSize.small:
            return self.small_model or DEFAULT_SMALL_MODEL
        else:
            return self.model or DEFAULT_MODEL

    def _convert_messages_to_groq_format(
        self, messages: list[Message]
    ) -> list[ChatCompletionMessageParam]:
        """Convert internal Message format to Groq ChatCompletionMessageParam format."""
        groq_messages: list[ChatCompletionMessageParam] = []
        for m in messages:
            # Clean input to prevent unicode and control character issues
            m.content = self._clean_input(m.content)
            if m.role == 'user':
                groq_messages.append({'role': 'user', 'content': m.content})
            elif m.role == 'system':
                groq_messages.append({'role': 'system', 'content': m.content})
        return groq_messages

    def _supports_structured_output(self, model: str | None = None) -> bool:
        """Check if the current model supports structured outputs (json_schema)."""
        if model is None:
            model = self.model or DEFAULT_MODEL
        return model in STRUCTURED_OUTPUT_MODELS

    def _build_response_format(self, response_model: type[BaseModel] | None = None) -> dict[str, typing.Any]:
        """Build the response format - use structured output for compatible Groq models."""
        if response_model is not None and self._supports_structured_output():
            # Use Groq's structured output format for supported models
            schema = response_model.model_json_schema()
            return {
                'type': 'json_schema',
                'json_schema': {
                    'name': response_model.__name__,
                    'schema': schema
                }
            }
        else:
            # Fallback to simple JSON object for unsupported models or no response_model
            return {'type': 'json_object'}

    def _handle_groq_response(self, response, use_structured_output: bool = False) -> dict[str, typing.Any]:
        """Handle Groq response with structured output support."""
        try:
            result = response.choices[0].message.content or '{}'
            if not result.strip():
                raise ValueError('Empty response from Groq API')

            # For structured outputs, trust Groq's validation (binary output guarantee)
            if use_structured_output:
                # Groq guarantees valid JSON for structured outputs
                return json.loads(result)

            # Legacy handling for basic JSON mode
            try:
                return json.loads(result)
            except json.JSONDecodeError as e:
                logger.error(f'JSON decode error. Raw response: {result[:200]}...')
                # Try to extract JSON if it's embedded in other text
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        pass
                raise ValueError(f'Invalid JSON response from Groq: {str(e)}') from e

        except (IndexError, AttributeError) as e:
            raise ValueError(f'Malformed response structure from Groq: {str(e)}') from e

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        """Generate response with comprehensive error handling and validation."""
        groq_messages = self._convert_messages_to_groq_format(messages)
        model = self._get_model_for_size(model_size)

        # Build response format (simple approach)
        response_format = self._build_response_format(response_model)
        use_structured_output = response_model is not None and self._supports_structured_output()

        # Log model selection and structured output usage
        logger.debug(f'Using Groq model: {model} for {model_size.value} size task')
        if use_structured_output:
            logger.debug(f'Using structured outputs (json_schema) for {response_model.__name__}')
        else:
            logger.debug('Using basic JSON mode')

        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=groq_messages,
                temperature=self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                response_format=response_format,
            )
            return self._handle_groq_response(response, use_structured_output)

        except groq.APIConnectionError as e:
            logger.error(f'Groq API connection error: {e}')
            raise Exception(f'Failed to connect to Groq API: {str(e)}') from e
        except groq.RateLimitError as e:
            logger.warning(f'Groq rate limit exceeded: {e}')
            raise RateLimitError from e
        except groq.APITimeoutError as e:
            logger.error(f'Groq API timeout: {e}')
            raise Exception(f'Groq API request timed out: {str(e)}') from e
        except groq.APIStatusError as e:
            logger.error(f'Groq API status error: {e.status_code} - {e.message}')
            if e.status_code == 400:
                raise ValueError(f'Invalid request to Groq API: {e.message}') from e
            elif e.status_code == 401:
                raise ValueError('Invalid Groq API key') from e
            elif e.status_code == 403:
                raise ValueError('Groq API access forbidden') from e
            elif e.status_code >= 500:
                raise Exception(f'Groq server error: {e.message}') from e
            else:
                raise Exception(f'Groq API error: {e.status_code} - {e.message}') from e
        except Exception as e:
            logger.error(f'Unexpected error in Groq response generation: {e}')
            raise

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        """Generate response with intelligent structured output handling."""
        if max_tokens is None:
            max_tokens = self.max_tokens

        # Check if we'll use structured outputs (simple check)
        use_structured_output = response_model is not None and self._supports_structured_output()

        # Only add schema to messages for models that don't support structured outputs
        if response_model is not None and not use_structured_output:
            serialized_model = json.dumps(response_model.model_json_schema())
            messages[
                -1
            ].content += (
                f'\n\nRespond with a JSON object in the following format:\n\n{serialized_model}'
            )

        # Add multilingual extraction instructions
        messages[0].content += MULTILINGUAL_EXTRACTION_RESPONSES

        # For structured outputs, trust Groq's binary guarantee (no retries needed for JSON parsing)
        if use_structured_output:
            logger.debug('Using structured output - simplified error handling')
            try:
                return await self._generate_response(
                    messages, response_model, max_tokens, model_size
                )
            except Exception as e:
                # For structured outputs, if Groq fails, it's likely a real API issue
                logger.error(f'Structured output request failed: {e}')
                raise

        # Legacy retry logic for basic JSON mode
        retry_count = 0
        last_error = None

        while retry_count <= MAX_RETRIES:
            try:
                response = await self._generate_response(
                    messages, response_model, max_tokens, model_size
                )
                return response
            except (RateLimitError, ValueError) as e:
                # Don't retry rate limits or validation errors
                if 'rate limit' in str(e).lower():
                    raise
                if 'api key' in str(e).lower():
                    raise
                if retry_count >= MAX_RETRIES:
                    raise
                last_error = e
            except Exception as e:
                last_error = e

                # Don't retry if we've hit the max retries
                if retry_count >= MAX_RETRIES:
                    logger.error(f'Max retries ({MAX_RETRIES}) exceeded. Last error: {e}')
                    raise

            retry_count += 1

            # Construct detailed error context for the LLM
            error_context = (
                f'The previous response attempt was invalid. '
                f'Error type: {last_error.__class__.__name__}. '
                f'Error details: {str(last_error)}. '
                f'Please try again with a valid JSON response, ensuring the output matches '
                f'the expected format and constraints. Focus on proper JSON syntax and '
                f'include all required fields.'
            )

            error_message = Message(role='user', content=error_context)
            messages.append(error_message)
            logger.warning(
                f'Retrying Groq request after error (attempt {retry_count}/{MAX_RETRIES}): {last_error}'
            )

        # If we somehow get here, raise the last error
        raise last_error or Exception('Max retries exceeded with no specific error')
