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

# Running tests: pytest -xvs tests/llm_client/test_dspy_client.py

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from graphiti_core.llm_client.config import LLMConfig, ModelSize
from graphiti_core.llm_client.errors import RateLimitError
from graphiti_core.prompts.models import Message


# Test models for response testing
class SimpleResponseModel(BaseModel):
    """Simple test model for response testing."""

    test_field: str
    optional_field: int = 0


class ComplexResponseModel(BaseModel):
    """Complex test model with nested fields."""

    name: str = Field(..., description='Name of the entity')
    value: int = Field(..., description='Numeric value')
    tags: list[str] = Field(default_factory=list, description='List of tags')


# Skip all tests if dspy is not installed
pytest.importorskip('dspy')

from graphiti_core.llm_client.dspy_client import DSPyClient  # noqa: E402


@pytest.fixture
def mock_dspy_lm():
    """Fixture to mock the DSPy LM."""
    with patch('dspy.LM') as mock_lm_class:
        mock_lm = MagicMock()
        mock_lm_class.return_value = mock_lm
        yield mock_lm


@pytest.fixture
def mock_dspy_predict():
    """Fixture to mock dspy.Predict."""
    with patch('dspy.Predict') as mock_predict_class:
        mock_predictor = MagicMock()
        mock_predict_class.return_value = mock_predictor
        yield mock_predictor, mock_predict_class


@pytest.fixture
def mock_dspy_context():
    """Fixture to mock dspy.context."""
    with patch('dspy.context') as mock_context:
        mock_context.return_value.__enter__ = MagicMock(return_value=None)
        mock_context.return_value.__exit__ = MagicMock(return_value=None)
        yield mock_context


@pytest.fixture
def dspy_client(mock_dspy_lm):
    """Fixture to create a DSPyClient with a mocked LM."""
    config = LLMConfig(
        api_key='test_api_key',
        model='gemini/gemini-2.5-flash',
        temperature=0.5,
        max_tokens=1000,
    )
    client = DSPyClient(config=config, cache=False)
    return client


class TestDSPyClientInitialization:
    """Tests for DSPyClient initialization."""

    def test_init_with_config(self, mock_dspy_lm):
        """Test initialization with a config object."""
        config = LLMConfig(
            api_key='test_api_key',
            model='gemini/gemini-2.5-flash',
            temperature=0.5,
            max_tokens=1000,
        )
        client = DSPyClient(config=config, cache=False)

        assert client.config == config
        assert client.model == 'gemini/gemini-2.5-flash'
        assert client.temperature == 0.5
        assert client.max_tokens == 1000

    def test_init_with_model_override(self, mock_dspy_lm):
        """Test initialization with model override parameter."""
        config = LLMConfig(
            api_key='test_api_key',
            model='gemini/gemini-2.5-flash',
        )
        client = DSPyClient(config=config, model='openai/gpt-4o')

        # Model parameter should override config
        assert client.model == 'openai/gpt-4o'

    def test_init_without_model_raises_error(self, mock_dspy_lm):
        """Test initialization without model raises ValueError."""
        config = LLMConfig(api_key='test_api_key')  # No model specified

        with pytest.raises(ValueError, match='Model must be specified'):
            DSPyClient(config=config)

    def test_init_with_custom_adapter(self, mock_dspy_lm):
        """Test initialization with custom adapter."""
        import dspy

        custom_adapter = dspy.ChatAdapter()
        config = LLMConfig(model='gemini/gemini-2.5-flash')
        client = DSPyClient(config=config, adapter=custom_adapter)

        assert client.adapter == custom_adapter


class TestDSPyClientPromptExtraction:
    """Tests for prompt extraction from messages."""

    def test_extract_prompts_simple(self, dspy_client):
        """Test extracting prompts from simple message list."""
        messages = [
            Message(role='system', content='System message'),
            Message(role='user', content='User message'),
        ]

        system_prompt, user_prompt = dspy_client._extract_prompts(messages)

        assert system_prompt == 'System message'
        assert user_prompt == 'User message'

    def test_extract_prompts_multiple_user_messages(self, dspy_client):
        """Test extracting prompts with multiple user messages."""
        messages = [
            Message(role='system', content='System message'),
            Message(role='user', content='First user message'),
            Message(role='user', content='Second user message'),
        ]

        system_prompt, user_prompt = dspy_client._extract_prompts(messages)

        assert system_prompt == 'System message'
        assert 'First user message' in user_prompt
        assert 'Second user message' in user_prompt

    def test_extract_prompts_no_system_message(self, dspy_client):
        """Test extracting prompts without system message."""
        messages = [
            Message(role='user', content='User message'),
        ]

        system_prompt, user_prompt = dspy_client._extract_prompts(messages)

        assert system_prompt == ''
        assert user_prompt == 'User message'


class TestDSPyClientSignatureCreation:
    """Tests for dynamic signature creation."""

    def test_create_signature_for_simple_model(self, dspy_client):
        """Test signature creation for simple Pydantic model."""
        SignatureClass = dspy_client._create_signature_for_model(
            SimpleResponseModel, 'Test system prompt'
        )

        # Check signature has correct structure
        assert hasattr(SignatureClass, '__annotations__')
        assert 'user_input' in SignatureClass.__annotations__
        assert 'result' in SignatureClass.__annotations__
        # The result type should be the Pydantic model
        assert SignatureClass.__annotations__['result'] == SimpleResponseModel

    def test_create_signature_for_complex_model(self, dspy_client):
        """Test signature creation for complex Pydantic model."""
        SignatureClass = dspy_client._create_signature_for_model(
            ComplexResponseModel, 'Extract entities'
        )

        assert SignatureClass.__annotations__['result'] == ComplexResponseModel
        assert 'Extract entities' in SignatureClass.__doc__

    def test_create_signature_empty_system_prompt(self, dspy_client):
        """Test signature creation with empty system prompt."""
        SignatureClass = dspy_client._create_signature_for_model(SimpleResponseModel, '')

        # Should use default docstring
        assert 'Generate a structured response' in SignatureClass.__doc__


class TestDSPyClientGenerateResponse:
    """Tests for DSPyClient generate_response method."""

    @pytest.mark.asyncio
    async def test_generate_response_with_structured_output(
        self, dspy_client, mock_dspy_predict, mock_dspy_context
    ):
        """Test response generation with structured output."""
        mock_predictor, mock_predict_class = mock_dspy_predict

        # Setup mock prediction result
        mock_result = SimpleResponseModel(test_field='test_value', optional_field=42)
        mock_prediction = MagicMock()
        mock_prediction.result = mock_result
        mock_predictor.return_value = mock_prediction

        # Call method
        messages = [
            Message(role='system', content='System message'),
            Message(role='user', content='User message'),
        ]
        result = await dspy_client.generate_response(
            messages=messages, response_model=SimpleResponseModel
        )

        # Assertions
        assert isinstance(result, dict)
        assert result['test_field'] == 'test_value'
        assert result['optional_field'] == 42

    @pytest.mark.asyncio
    async def test_generate_response_unstructured(
        self, dspy_client, mock_dspy_context
    ):
        """Test response generation without structured output."""
        # Mock the LM direct call
        dspy_client.lm.return_value = ['Test response text']

        # Call method
        messages = [Message(role='user', content='Test message')]
        result = await dspy_client.generate_response(messages)

        # Assertions
        assert isinstance(result, dict)
        assert result['content'] == 'Test response text'

    @pytest.mark.asyncio
    async def test_generate_response_rate_limit_error(
        self, dspy_client, mock_dspy_predict, mock_dspy_context
    ):
        """Test handling of rate limit errors."""
        mock_predictor, mock_predict_class = mock_dspy_predict
        mock_predictor.side_effect = Exception('Rate limit exceeded')

        messages = [Message(role='user', content='Test message')]
        with pytest.raises(RateLimitError):
            await dspy_client.generate_response(messages, response_model=SimpleResponseModel)

    @pytest.mark.asyncio
    async def test_generate_response_quota_error(
        self, dspy_client, mock_dspy_predict, mock_dspy_context
    ):
        """Test handling of quota errors."""
        mock_predictor, mock_predict_class = mock_dspy_predict
        mock_predictor.side_effect = Exception('Quota exceeded for requests')

        messages = [Message(role='user', content='Test message')]
        with pytest.raises(RateLimitError):
            await dspy_client.generate_response(messages, response_model=SimpleResponseModel)

    @pytest.mark.asyncio
    async def test_generate_response_resource_exhausted_error(
        self, dspy_client, mock_dspy_predict, mock_dspy_context
    ):
        """Test handling of resource exhausted errors."""
        mock_predictor, mock_predict_class = mock_dspy_predict
        mock_predictor.side_effect = Exception('resource_exhausted: Request limit exceeded')

        messages = [Message(role='user', content='Test message')]
        with pytest.raises(RateLimitError):
            await dspy_client.generate_response(messages, response_model=SimpleResponseModel)

    @pytest.mark.asyncio
    async def test_generate_response_dict_result(
        self, dspy_client, mock_dspy_predict, mock_dspy_context
    ):
        """Test handling when DSPy returns a dict instead of Pydantic model."""
        mock_predictor, mock_predict_class = mock_dspy_predict

        # Return dict instead of Pydantic model
        mock_prediction = MagicMock()
        mock_prediction.result = {'test_field': 'dict_value', 'optional_field': 10}
        mock_predictor.return_value = mock_prediction

        messages = [Message(role='user', content='Test message')]
        result = await dspy_client.generate_response(
            messages=messages, response_model=SimpleResponseModel
        )

        assert result['test_field'] == 'dict_value'
        assert result['optional_field'] == 10


class TestDSPyClientProviderType:
    """Tests for provider type detection."""

    def test_get_provider_type_gemini(self, mock_dspy_lm):
        """Test provider type detection for Gemini."""
        config = LLMConfig(model='gemini/gemini-2.5-flash')
        client = DSPyClient(config=config)
        assert client._get_provider_type() == 'gemini'

    def test_get_provider_type_vertex_ai(self, mock_dspy_lm):
        """Test provider type detection for Vertex AI."""
        config = LLMConfig(model='vertex_ai/gemini-2.0-flash')
        client = DSPyClient(config=config)
        assert client._get_provider_type() == 'gemini'

    def test_get_provider_type_openai(self, mock_dspy_lm):
        """Test provider type detection for OpenAI."""
        config = LLMConfig(model='openai/gpt-4o')
        client = DSPyClient(config=config)
        assert client._get_provider_type() == 'openai'

    def test_get_provider_type_anthropic(self, mock_dspy_lm):
        """Test provider type detection for Anthropic."""
        config = LLMConfig(model='anthropic/claude-3-5-sonnet')
        client = DSPyClient(config=config)
        assert client._get_provider_type() == 'anthropic'

    def test_get_provider_type_groq(self, mock_dspy_lm):
        """Test provider type detection for Groq."""
        config = LLMConfig(model='groq/llama-3-70b')
        client = DSPyClient(config=config)
        assert client._get_provider_type() == 'groq'

    def test_get_provider_type_unknown(self, mock_dspy_lm):
        """Test provider type detection for unknown provider."""
        config = LLMConfig(model='custom/some-model')
        client = DSPyClient(config=config)
        assert client._get_provider_type() == 'dspy'


class TestDSPyClientModelSize:
    """Tests for model size handling."""

    @pytest.mark.asyncio
    async def test_model_size_parameter_accepted(
        self, dspy_client, mock_dspy_predict, mock_dspy_context
    ):
        """Test that model_size parameter is accepted."""
        mock_predictor, mock_predict_class = mock_dspy_predict

        mock_result = SimpleResponseModel(test_field='test')
        mock_prediction = MagicMock()
        mock_prediction.result = mock_result
        mock_predictor.return_value = mock_prediction

        messages = [Message(role='user', content='Test')]

        # Should not raise even though we pass model_size
        result = await dspy_client.generate_response(
            messages=messages,
            response_model=SimpleResponseModel,
            model_size=ModelSize.small,
        )

        assert result['test_field'] == 'test'


if __name__ == '__main__':
    pytest.main(['-v', 'test_dspy_client.py'])

