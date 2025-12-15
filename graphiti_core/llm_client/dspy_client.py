"""
Done by Christian Gintenreiter
"""

import asyncio
import logging
import typing
from typing import TYPE_CHECKING

from pydantic import BaseModel

from ..prompts.models import Message
from .client import LLMClient
from .config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from .errors import RateLimitError

if TYPE_CHECKING:
    import dspy
else:
    try:
        import dspy
    except ImportError:
        raise ImportError(
            'dspy is required for DSPyClient. '
            'Install it with: pip install graphiti-core[dspy]'
        ) from None


logger = logging.getLogger(__name__)


class DSPyClient(LLMClient):
    """
    DSPyClient is a client class for interacting with LLMs via DSPy framework.

    This client leverages DSPy's robust type enforcement and automatic retry
    mechanisms to guarantee that LLM responses conform to Pydantic schemas.
    When validation fails, DSPy retries with the validation error as feedback.

    DSPy supports multiple LLM providers through LiteLLM, including:
    - Gemini: "gemini/gemini-2.5-flash"
    - OpenAI: "openai/gpt-4o"
    - Anthropic: "anthropic/claude-3-5-sonnet"
    - And many more via LiteLLM prefixes

    Attributes:
        model (str): The model name in LiteLLM format (e.g., "gemini/gemini-2.5-flash").
        temperature (float): The temperature to use for generating responses.
        max_tokens (int): The maximum number of tokens to generate in a response.
        lm (dspy.LM): The DSPy language model instance.
        adapter (dspy.Adapter): The DSPy adapter for structured output parsing.

    Methods:
        __init__(config, cache, model, adapter):
            Initializes the DSPyClient with the provided configuration.

        _generate_response(messages, response_model, max_tokens, model_size):
            Generates a response from the language model with type enforcement.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        model: str | None = None,
        adapter: 'dspy.Adapter | None' = None,
    ):
        """
        Initialize the DSPyClient with the provided configuration.

        Args:
            config (LLMConfig | None): The configuration for the LLM client.
            cache (bool): Whether to use caching for responses. Defaults to False.
            model (str | None): The model name in LiteLLM format (e.g., "gemini/gemini-2.5-flash").
                If provided, overrides the model from config.
            adapter (dspy.Adapter | None): The DSPy adapter to use. Defaults to JSONAdapter
                for reliable structured output parsing.
        """
        if config is None:
            config = LLMConfig()

        super().__init__(config, cache)

        # Override model from parameter if provided
        if model is not None:
            self.model = model

        if self.model is None:
            raise ValueError(
                'Model must be specified either in config or as model parameter. '
                'Use LiteLLM format like "gemini/gemini-2.5-flash" or "openai/gpt-4o".'
            )

        # Initialize DSPy LM
        self.lm: dspy.LM = dspy.LM(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Use JSONAdapter by default for reliable structured output
        self.adapter: dspy.Adapter = adapter if adapter is not None else dspy.JSONAdapter()

    def _create_signature_for_model(
        self,
        response_model: type[BaseModel],
        system_prompt: str,
    ) -> type['dspy.Signature']:
        """
        Dynamically create a DSPy Signature using the Pydantic model directly as OutputField type.

        This is the key innovation: the Pydantic model is used directly as the type annotation
        for the OutputField, enabling DSPy to validate responses against the actual schema
        and retry with validation error feedback when parsing fails.

        Args:
            response_model: The Pydantic model class to use for structured output.
            system_prompt: The system prompt to use as the signature docstring.

        Returns:
            A dynamically created DSPy Signature class with the Pydantic model as output type.
        """
        # Create a new signature class dynamically
        # The response_model is used directly as the type annotation
        signature_attrs = {
            '__doc__': system_prompt if system_prompt else 'Generate a structured response.',
            '__annotations__': {
                'user_input': str,
                'result': response_model,  # Direct Pydantic type!
            },
            'user_input': dspy.InputField(desc='User message content'),
            'result': dspy.OutputField(
                desc=f'Response matching {response_model.__name__} schema'
            ),
        }

        DynamicSignature = type('DynamicSignature', (dspy.Signature,), signature_attrs)
        return DynamicSignature

    def _extract_prompts(self, messages: list[Message]) -> tuple[str, str]:
        """
        Extract system and user prompts from message list.

        Args:
            messages: List of Message objects with role and content.

        Returns:
            Tuple of (system_prompt, user_prompt).
        """
        system_prompt = ''
        user_parts: list[str] = []

        for msg in messages:
            cleaned_content = self._clean_input(msg.content)
            if msg.role == 'system':
                system_prompt = cleaned_content
            else:
                user_parts.append(cleaned_content)

        return system_prompt, '\n'.join(user_parts)

    def _run_dspy_prediction(
        self,
        response_model: type[BaseModel],
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, typing.Any]:
        """
        Run DSPy prediction synchronously with type enforcement.

        Args:
            response_model: The Pydantic model for structured output.
            system_prompt: The system prompt content.
            user_prompt: The user prompt content.

        Returns:
            Dictionary representation of the validated Pydantic model.

        Raises:
            RateLimitError: If rate limit is exceeded.
            Exception: If generation fails after retries.
        """
        try:
            # Create signature with Pydantic type directly
            SignatureClass = self._create_signature_for_model(response_model, system_prompt)

            # Configure DSPy context and predict
            with dspy.context(lm=self.lm, adapter=self.adapter):
                predictor = dspy.Predict(SignatureClass)
                prediction = predictor(user_input=user_prompt)

                # prediction.result is already validated Pydantic model
                result = prediction.result
                if isinstance(result, BaseModel):
                    return result.model_dump()
                elif isinstance(result, dict):
                    # In case DSPy returns a dict, validate it
                    validated = response_model.model_validate(result)
                    return validated.model_dump()
                else:
                    raise ValueError(f'Unexpected result type: {type(result)}')

        except Exception as e:
            error_message = str(e).lower()
            if (
                'rate limit' in error_message
                or 'quota' in error_message
                or 'resource_exhausted' in error_message
                or '429' in str(e)
            ):
                raise RateLimitError from e

            logger.error(f'Error in DSPy prediction: {e}')
            raise

    def _run_dspy_unstructured(self, user_prompt: str) -> dict[str, typing.Any]:
        """
        Run DSPy for unstructured text generation.

        Args:
            user_prompt: The user prompt content.

        Returns:
            Dictionary with 'content' key containing the response.
        """
        try:
            with dspy.context(lm=self.lm):
                # Direct LM call for unstructured output
                response = self.lm(user_prompt)
                # DSPy LM returns a list of responses
                if isinstance(response, list) and len(response) > 0:
                    content = response[0] if isinstance(response[0], str) else str(response[0])
                else:
                    content = str(response)
                return {'content': content}

        except Exception as e:
            error_message = str(e).lower()
            if (
                'rate limit' in error_message
                or 'quota' in error_message
                or 'resource_exhausted' in error_message
                or '429' in str(e)
            ):
                raise RateLimitError from e

            logger.error(f'Error in DSPy unstructured generation: {e}')
            raise

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        """
        Generate a response from the language model via DSPy with type enforcement.

        When a response_model is provided, DSPy:
        1. Creates a Signature with the Pydantic model as OutputField type
        2. Validates the LLM response against the schema
        3. Retries with validation error feedback if parsing fails

        Args:
            messages (list[Message]): A list of messages to send to the language model.
            response_model (type[BaseModel] | None): Pydantic model for structured output.
            max_tokens (int): Maximum tokens to generate (used for context, actual limit set on LM).
            model_size (ModelSize): Model size selection (currently ignored, uses configured model).

        Returns:
            dict[str, typing.Any]: The validated response as a dictionary.

        Raises:
            RateLimitError: If the API rate limit is exceeded.
            Exception: If there is an error generating the response.
        """
        # Extract system and user prompts from messages
        system_prompt, user_prompt = self._extract_prompts(messages)

        if response_model is not None:
            # Run structured prediction with type enforcement
            # Use asyncio.to_thread for async compatibility since DSPy is synchronous
            return await asyncio.to_thread(
                self._run_dspy_prediction,
                response_model,
                system_prompt,
                user_prompt,
            )
        else:
            # Unstructured generation
            return await asyncio.to_thread(
                self._run_dspy_unstructured,
                user_prompt,
            )

    def _get_provider_type(self) -> str:
        """Get provider type from model string."""
        if self.model:
            model_lower = self.model.lower()
            if model_lower.startswith('gemini/') or model_lower.startswith('vertex_ai/'):
                return 'gemini'
            elif model_lower.startswith('openai/'):
                return 'openai'
            elif model_lower.startswith('anthropic/'):
                return 'anthropic'
            elif model_lower.startswith('groq/'):
                return 'groq'
        return 'dspy'

