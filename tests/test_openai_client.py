"""
Tests for OpenAI client implementation.
"""

import os
import pytest
from ace.llm import OpenAIClient


def test_openai_client_initialization():
    """Test that OpenAI client initializes correctly."""
    # Skip if no API key available
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    
    client = OpenAIClient()
    assert client.model == "gpt-5"
    assert client.default_temperature == 1.0
    assert client.default_max_tokens == 4096


def test_openai_client_custom_model():
    """Test initialization with custom model."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    
    client = OpenAIClient(model="gpt-5-mini", temperature=1.0)
    assert client.model == "gpt-5-mini"
    assert client.default_temperature == 1.0


def test_openai_client_no_api_key():
    """Test that client raises error without API key."""
    # Temporarily remove API key from environment
    original_key = os.environ.pop("OPENAI_API_KEY", None)
    
    try:
        with pytest.raises(ValueError, match="OpenAI API key must be provided"):
            OpenAIClient()
    finally:
        # Restore original key if it existed
        if original_key:
            os.environ["OPENAI_API_KEY"] = original_key


def test_openai_client_generate():
    """Test basic text generation."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    
    client = OpenAIClient()
    response = client.generate("Say 'Hello, ACE!' and nothing else.")
    
    assert isinstance(response, str)
    assert len(response) > 0
    print(f"\nGenerate response: {response}")


def test_openai_client_generate_json():
    """Test JSON generation."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    
    client = OpenAIClient()
    prompt = """Return a JSON object with these exact fields:
    - greeting: a friendly greeting
    - number: the number 42
    
    Return ONLY valid JSON, no other text."""
    
    response = client.generate_json(prompt)
    
    assert isinstance(response, dict)
    assert "greeting" in response
    assert "number" in response
    assert response["number"] == 42
    print(f"\nJSON response: {response}")


def test_openai_client_generate_with_messages():
    """Test chat-based generation with messages."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    
    client = OpenAIClient()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    
    response = client.generate_with_messages(messages)
    
    assert isinstance(response, str)
    assert len(response) > 0
    assert "4" in response
    print(f"\nMessages response: {response}")


def test_openai_client_temperature_override():
    """Test that temperature can be overridden per call."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    
    client = OpenAIClient(temperature=1.0)
    
    # Generate with temperature explicitly set (GPT-5 only supports 1.0)
    response = client.generate(
        "Say exactly: 'Temperature test'",
        temperature=1.0
    )
    
    assert isinstance(response, str)
    assert len(response) > 0
    print(f"\nTemperature override response: {response}")
