"""
Unit tests for Weather Dashboard Application

Run tests with: pytest tests/test_app.py -v
"""

import pytest
import json
from unittest.mock import patch, Mock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app


@pytest.fixture
def client():
    """Create test client for Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_homepage_loads(client):
    """Test that homepage loads successfully."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Weather Dashboard' in response.data


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get('/api/health')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert data['status'] == 'healthy'
    assert 'timestamp' in data
    assert data['service'] == 'weather-dashboard'


@patch('app.requests.get')
def test_weather_api_success(mock_get, client):
    """Test successful weather data retrieval."""
    # Mock API response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        'name': 'London',
        'sys': {'country': 'GB'},
        'main': {
            'temp': 15.5,
            'feels_like': 14.2,
            'humidity': 72,
            'pressure': 1013
        },
        'weather': [{'description': 'cloudy'}],
        'wind': {'speed': 5.2}
    }
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response
    
    response = client.get('/api/weather/London')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert data['city'] == 'London'
    assert data['country'] == 'GB'
    assert data['temperature'] == 15.5


@patch('app.requests.get')
def test_weather_api_city_not_found(mock_get, client):
    """Test handling of city not found error."""
    mock_response = Mock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = Exception("City not found")
    mock_get.return_value = mock_response
    
    response = client.get('/api/weather/InvalidCityName123')
    assert response.status_code == 404
    
    data = json.loads(response.data)
    assert 'error' in data


def test_weather_api_invalid_input(client):
    """Test handling of invalid city name input."""
    response = client.get('/api/weather/' + 'x' * 150)
    assert response.status_code == 400
    
    data = json.loads(response.data)
    assert 'error' in data
    assert 'Invalid' in data['error']


@patch('app.requests.get')
def test_weather_api_timeout(mock_get, client):
    """Test handling of API timeout."""
    import requests
    mock_get.side_effect = requests.Timeout("Request timed out")
    
    response = client.get('/api/weather/London')
    assert response.status_code == 504
    
    data = json.loads(response.data)
    assert 'timeout' in data['error'].lower()


def test_404_error_handler(client):
    """Test 404 error handler."""
    response = client.get('/nonexistent-route')
    assert response.status_code == 404
    
    data = json.loads(response.data)
    assert 'error' in data


@patch('app.os.getenv')
def test_missing_api_key(mock_getenv, client):
    """Test handling when API key is not configured."""
    def getenv_side_effect(key, default=None):
        if key == 'WEATHER_API_KEY':
            return None
        return default
    
    mock_getenv.side_effect = getenv_side_effect
    
    response = client.get('/api/weather/London')
    assert response.status_code == 500
    
    data = json.loads(response.data)
    assert 'not configured' in data['error'].lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
