import pytest
import requests
import json
import time
import os
from typing import Dict, Any

DEFAULT_URL = "http://localhost:8080"
TEST_TIMEOUT = 10

SAMPLE_TEXT = (
    "Apple Inc. was founded by Steve Jobs and Steve Wozniak in Cupertino, California."
)
ENTITY_TYPES = ["PERSON", "ORGANIZATION", "LOCATION"]


@pytest.fixture
def service_url() -> str:
    """Get the service URL from environment variable or use default."""
    return os.environ.get("SERVICE_URL", DEFAULT_URL)


def request_prediction(
    url: str, text: str, entities=None, mask_entities=None
) -> Dict[str, Any]:
    """
    Send a prediction request to the GLiNER service

    Args:
        url: Service URL
        text: Text to analyze
        entities: Entity types to extract
        mask_entities: Entity types to mask

    Returns:
        Response JSON
    """
    endpoint = f"{url}/v1/models/gliner-multi-v2-1:predict"
    payload = {"text": text}

    if entities is not None:
        payload["entities"] = entities

    if mask_entities is not None:
        payload["mask_entities"] = mask_entities

    headers = {"Content-Type": "application/json"}
    response = requests.post(
        endpoint, data=json.dumps(payload), headers=headers, timeout=TEST_TIMEOUT
    )

    return response.json()


def test_service_readiness(service_url: str):
    """Test if the service is ready."""
    max_retries = 5
    retry_delay = 2

    for i in range(max_retries):
        try:
            response = requests.get(
                f"{service_url}/v1/models/gliner-multi-v2-1", timeout=TEST_TIMEOUT
            )
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass

        time.sleep(retry_delay)

    pytest.fail(f"Service at {service_url} is not ready after {max_retries} attempts")


def test_basic_prediction(service_url: str):
    """Test basic entity prediction without masking."""
    response = request_prediction(service_url, SAMPLE_TEXT, ENTITY_TYPES)

    assert response.get("status") == "success", f"Prediction failed: {response}"
    assert "raw_output" in response, "Response missing raw_output field"
    assert "input" in response, "Response missing input field"

    entities = response.get("raw_output", [])
    assert len(entities) > 0, "No entities detected in sample text"

    if entities:
        print(f"Debug - Entity keys: {entities[0].keys()}")

    entity_types = set()
    for entity in entities:
        if "type" in entity:
            entity_types.add(entity["type"])
        elif "label" in entity:
            entity_types.add(entity["label"])
        else:
            print(f"Debug - Entity structure: {entity}")

    print(f"Debug - Found entity types: {entity_types}")
    assert entity_types, "No entity types could be identified in the response"

    assert (
        response.get("output") == SAMPLE_TEXT
    ), "Output text doesn't match input when no masking is applied"


def test_entity_masking(service_url: str):
    """Test entity prediction with masking."""
    mask_types = ["PERSON"]
    response = request_prediction(service_url, SAMPLE_TEXT, ENTITY_TYPES, mask_types)

    assert (
        response.get("status") == "success"
    ), f"Prediction with masking failed: {response}"
    assert "mask_values" in response, "Response missing mask_values field"

    assert (
        response.get("output") != SAMPLE_TEXT
    ), "Output text matches input despite masking"

    mask_values = response.get("mask_values", {})
    assert len(mask_values) > 0, "No mask values returned despite requesting masking"

    for entity in response.get("raw_output", []):
        entity_text = SAMPLE_TEXT[entity["start"] : entity["end"]]

        entity_type = entity.get("type", entity.get("label", "unknown"))

        if entity_type in mask_types:
            assert (
                entity_text in mask_values
            ), f"Entity '{entity_text}' of type {entity_type} was not masked"


def test_error_handling(service_url: str):
    """Test error handling with invalid inputs."""
    response = request_prediction(service_url, "", ENTITY_TYPES)
    assert response.get("status") == "error", "Empty text should return an error status"

    long_text = "This is a test. " * 1000  # 15,000+ characters
    response = request_prediction(service_url, long_text, ENTITY_TYPES)
    assert "status" in response, "Response to long text missing status field"


if __name__ == "__main__":
    import sys

    service_url = os.environ.get("SERVICE_URL", DEFAULT_URL)
    print(f"Testing GLiNER service at {service_url}")

    test_service_readiness(service_url)
    print("✓ Service is ready")

    test_basic_prediction(service_url)
    print("✓ Basic prediction test passed")

    test_entity_masking(service_url)
    print("✓ Entity masking test passed")

    test_error_handling(service_url)
    print("✓ Error handling test passed")

    print("All tests passed!")
    sys.exit(0)
