"""
OpenRouter client with provider routing support.
"""

import json
import logging
import typing
from typing import ClassVar

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from graphiti_core.llm_client.client import LLMClient, get_extraction_language_instruction
from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from graphiti_core.llm_client.errors import RateLimitError, RefusalError
from graphiti_core.prompts.models import Message

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'openai/gpt-4-turbo-preview'


class OpenRouterClient(LLMClient):
    """
    OpenRouterClient extends LLMClient to support OpenRouter's provider routing features.

    This client allows configuration of provider preferences, fallback behavior,
    and routing strategies for maximum reliability and cost optimization.
    """

    # Class-level constants
    MAX_RETRIES: ClassVar[int] = 2

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        provider_order: list[str] | None = None,
        allow_fallbacks: bool = True,
        provider_sort: str | None = None,
    ):
        """
        Initialize the OpenRouterClient with provider routing configuration.

        Args:
            config: The configuration for the LLM client
            cache: Whether to use caching (not implemented)
            provider_order: List of preferred providers in order
            allow_fallbacks: Whether to allow fallback to other providers
            provider_sort: Sort providers by 'price', 'throughput', or 'latency'
        """
        if cache:
            raise NotImplementedError('Caching is not implemented for OpenRouter')

        if config is None:
            config = LLMConfig()

        super().__init__(config, cache)

        # Add OpenRouter app attribution headers
        default_headers = {
            "HTTP-Referer": "https://github.com/getzep/graphiti",
            "X-Title": "Graphiti Knowledge Graph"
        }

        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            default_headers=default_headers
        )
        self.provider_order = provider_order
        self.allow_fallbacks = allow_fallbacks
        self.provider_sort = provider_sort

    def _build_provider_config(self) -> dict[str, typing.Any] | None:
        """Build the provider configuration for OpenRouter API requests."""
        provider_config = {}

        if self.provider_order:
            provider_config['order'] = self.provider_order

        if not self.allow_fallbacks:
            provider_config['allow_fallbacks'] = False

        if self.provider_sort:
            provider_config['sort'] = self.provider_sort

        return provider_config if provider_config else None

    def _is_openrouter_provider(self) -> bool:
        """Check if we're using OpenRouter (not direct provider through OpenRouter)."""
        base_url = getattr(self.client, 'base_url', None)
        if base_url:
            return 'openrouter.ai' in str(base_url)
        return False

    def _build_response_format(self, response_model: type[BaseModel] | None = None) -> dict[str, typing.Any]:
        """Build the response format - use structured output only for OpenRouter."""
        if response_model is not None and self._is_openrouter_provider():
            # Use OpenRouter's structured output format only when using OpenRouter
            schema = response_model.model_json_schema()
            return {
                'type': 'json_schema',
                'json_schema': {
                    'name': response_model.__name__,
                    'strict': True,
                    'schema': schema
                }
            }
        else:
            # Fallback to simple JSON object for non-OpenRouter or no response_model
            return {'type': 'json_object'}

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        openai_messages: list[ChatCompletionMessageParam] = []
        for m in messages:
            m.content = self._clean_input(m.content)
            if m.role == 'user':
                openai_messages.append({'role': 'user', 'content': m.content})
            elif m.role == 'system':
                openai_messages.append({'role': 'system', 'content': m.content})

        try:
            # Build the request parameters with proper structured output format
            request_params = {
                'model': self.model or DEFAULT_MODEL,
                'messages': openai_messages,
                'temperature': self.temperature,
                'max_tokens': self.max_tokens,
                'response_format': self._build_response_format(response_model),
            }

            # Add provider routing configuration if specified
            provider_config = self._build_provider_config()
            if provider_config:
                request_params['provider'] = provider_config
                logger.debug(f'Using OpenRouter provider config: {provider_config}')

            is_structured = response_model is not None and self._is_openrouter_provider()
            logger.debug(f'OpenRouter request - Structured output: {is_structured}, Format: {request_params["response_format"]}')
            response = await self.client.chat.completions.create(**request_params)
            result = response.choices[0].message.content or ''
            return json.loads(result)
        except openai.RateLimitError as e:
            raise RateLimitError from e
        except Exception as e:
            logger.error(f'Error in generating LLM response: {e}')
            raise

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None,
    ) -> dict[str, typing.Any]:
        if max_tokens is None:
            max_tokens = self.max_tokens

        retry_count = 0
        last_error = None

        if response_model is not None:
            serialized_model = json.dumps(response_model.model_json_schema())
            messages[
                -1
            ].content += (
                f'\n\nRespond with a JSON object in the following format:\n\n{serialized_model}'
            )

        # Add multilingual extraction instructions
        messages[0].content += get_extraction_language_instruction(group_id)

        while retry_count <= self.MAX_RETRIES:
            try:
                response = await self._generate_response(
                    messages, response_model, max_tokens=max_tokens, model_size=model_size
                )
                return response
            except (RateLimitError, RefusalError):
                # These errors should not trigger retries
                raise
            except (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError):
                # Let OpenAI's client handle these retries
                raise
            except Exception as e:
                last_error = e

                # Don't retry if we've hit the max retries
                if retry_count >= self.MAX_RETRIES:
                    logger.error(f'Max retries ({self.MAX_RETRIES}) exceeded. Last error: {e}')
                    raise

                retry_count += 1

                # Construct a detailed error message for the LLM
                error_context = (
                    f'The previous response attempt was invalid. '
                    f'Error type: {e.__class__.__name__}. '
                    f'Error details: {str(e)}. '
                    f'Please try again with a valid response, ensuring the output matches '
                    f'the expected format and constraints.'
                )

                error_message = Message(role='user', content=error_context)
                messages.append(error_message)
                logger.warning(
                    f'Retrying after application error (attempt {retry_count}/{self.MAX_RETRIES}): {e}'
                )

        # If we somehow get here, raise the last error
        raise last_error or Exception('Max retries exceeded with no specific error')