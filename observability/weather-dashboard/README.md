# Weather Dashboard - AI Agent Observability Demo

A Flask-based weather dashboard application demonstrating comprehensive observability patterns for AI agents using OpenTelemetry, suitable for monitoring autonomous agent actions in software development environments.

## Overview

This application serves as a practical reference implementation for the observability concepts outlined in the parent project's documentation. It demonstrates:

- OpenTelemetry instrumentation for distributed tracing
- Custom span creation for business logic tracking
- Exception recording and error propagation
- Integration with OTLP-compatible observability platforms
- Agent activity visualization utilities

## Project Structure

📁 **weather-dashboard/**
- 📄 `app.py` - Flask application with OpenTelemetry instrumentation
- 📄 `agent_visualizer.py` - Custom visualization utilities for agent traces
- 📄 `requirements.txt` - Python dependencies
- 📄 `.env` - Environment configuration (API keys, OTEL endpoints)
- 📄 `README.md` - Project documentation
- 📁 `templates/` - HTML templates for dashboard
- 📁 `static/` - CSS and JavaScript assets
- 📁 `tests/` - Automated test suite

## Prerequisites

- Python 3.11 or higher
- Active virtual environment (see Setup section)
- OpenWeatherMap API key (free tier available)
- Optional: Langfuse, Grafana Cloud, or other OTLP-compatible backend

## Setup Instructions

### 1. Verify Virtual Environment

Ensure you are in the parent project directory and have an active virtual environment:

```bash
# Navigate to parent directory
cd /home/laptop/EXERCISES/AUTONOMOUS/autonomous-artificial-intelligence/observability

# Check if virtual environment exists
ls -la venv/

# Activate virtual environment (if not already active)
source venv/bin/activate

# Verify Python version
python --version  # Should show Python 3.12.3 or higher
```

### 2. Install Dependencies

Install the weather dashboard dependencies in the virtual environment:

```bash
# Navigate to weather-dashboard directory
cd weather-dashboard

# Install requirements
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy the `.env` template and configure your API keys:

```bash
# The .env file already exists with template values
# Edit it with your actual configuration
nano .env  # or use your preferred editor
```

Required configuration:

```bash
# Get a free API key from https://openweathermap.org/api
WEATHER_API_KEY=your-actual-api-key-here

# Optional: Configure OTLP endpoint for observability
# OTEL_EXPORTER_OTLP_ENDPOINT=https://your-platform.com
# OTEL_EXPORTER_OTLP_HEADERS=Authorization=Bearer your-token
```

### 4. Run the Application

Start the Flask development server:

```bash
# Ensure virtual environment is active
python app.py
```

The application will start on `http://localhost:5000`

You should see output similar to:

```
Starting Weather Dashboard on port 5000
OpenTelemetry endpoint: Console (local development)
 * Running on http://0.0.0.0:5000
```

### 5. Access the Dashboard

Open your web browser and navigate to:

```
http://localhost:5000
```

Enter a city name (e.g., "London", "Tokyo", "New York") to fetch weather data.

## Running Tests

Execute the test suite to verify installation:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests with coverage
pytest tests/ -v --cov=. --cov-report=term-missing

# Run specific test file
pytest tests/test_app.py -v
```

## Observability Features

### Local Development (Console Mode)

By default, without configuring an OTLP endpoint, traces are output to the console. You will see detailed span information including:

- Request traces with timing information
- Custom spans for weather API calls and data processing
- Exception details when errors occur
- Performance metrics (token count analogy for API calls)

### Production Mode (OTLP Backend)

Configure environment variables to export traces to your observability platform:

```bash
# Langfuse configuration
OTEL_EXPORTER_OTLP_ENDPOINT=https://cloud.langfuse.com
OTEL_EXPORTER_OTLP_HEADERS=Authorization=Bearer your-langfuse-key

# Or Grafana Cloud
OTEL_EXPORTER_OTLP_ENDPOINT=https://tempo-us-central1.grafana.net/tempo
OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic your-base64-credentials
```

### Trace Structure

Each weather request creates a hierarchical trace:

```
fetch-weather (root span)
├── external-api-call (OpenWeatherMap API)
└── process-weather-data (data transformation)
```

Each span includes:
- Operation name and type
- Timing information (latency)
- Attributes (city name, temperature, HTTP status)
- Events (success/failure notifications)
- Exceptions (when errors occur)

## Agent Visualizer Usage

The `agent_visualizer.py` module provides utilities for tracking agent operations:

```python
from agent_visualizer import AgentVisualizer

# Create visualizer instance
visualizer = AgentVisualizer(session_id="weather_session")

# Log activities
visualizer.print_activity("Fetching weather data", level="INFO")
visualizer.print_activity("API call completed", level="DEBUG", 
                         context={"latency_ms": 245})

# Display conversation trace
visualizer.visualize_conversation()

# Export trace to file
visualizer.export_trace("trace_output.json", format="json")

# Get statistics
stats = visualizer.get_summary_statistics()
print(stats)
```

## API Endpoints

### GET /

Homepage - renders the weather dashboard interface.

### GET /api/weather/\<city\>

Fetch weather data for specified city.

**Response (Success - 200):**
```json
{
  "city": "London",
  "country": "GB",
  "temperature": 15.5,
  "feels_like": 14.2,
  "humidity": 72,
  "pressure": 1013,
  "weather": "cloudy",
  "wind_speed": 5.2,
  "timestamp": "2026-04-16T10:30:00Z"
}
```

**Response (Error - 404):**
```json
{
  "error": "City not found"
}
```

### GET /api/health

Health check endpoint for monitoring.

**Response (200):**
```json
{
  "status": "healthy",
  "timestamp": "2026-04-16T10:30:00Z",
  "service": "weather-dashboard",
  "version": "1.0.0"
}
```

## Troubleshooting

### Virtual Environment Verification

**Always verify your virtual environment is active before running any commands:**

```bash
# Check if virtual environment is active (look for (venv) in prompt)
# Your prompt should show: (venv) user@machine:~/path$

# Verify Python is from virtual environment
which python
# Should output: /path/to/observability/venv/bin/python

# Check Python version
python --version

# List installed packages
pip list
```

**If virtual environment is not showing as active:**

```bash
# Navigate to parent directory
cd /home/laptop/EXERCISES/AUTONOMOUS/autonomous-artificial-intelligence/observability

# Activate virtual environment
source venv/bin/activate

# Verify activation
which python  # Should point to venv/bin/python

# Return to weather-dashboard
cd weather-dashboard
```

**If activation fails, recreate the virtual environment:**

```bash
cd /home/laptop/EXERCISES/AUTONOMOUS/autonomous-artificial-intelligence/observability
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
cd weather-dashboard
pip install -r requirements.txt
```

### VS Code Environment File Configuration Error

**Error Message:**
```
An environment file is configured but terminal environment injection is disabled. 
Enable "python.terminal.useEnvFile" to use environment variables from .env files in terminals.
```

**Cause:** VS Code is not configured to automatically load environment variables from `.env` files into integrated terminal sessions.

**Solution 1: Enable via VS Code Settings UI**

1. Open VS Code Settings: `File > Preferences > Settings` (or `Ctrl+,` on Linux)
2. Search for: `python.terminal.useEnvFile`
3. Check the box to enable this setting
4. Restart any open terminal windows in VS Code

**Solution 2: Enable via settings.json**

1. Open Command Palette: `Ctrl+Shift+P`
2. Type: `Preferences: Open User Settings (JSON)`
3. Add the following configuration:

```json
{
  "python.terminal.useEnvFile": true
}
```

4. Save the file and restart VS Code terminals

**Solution 3: Manual Environment Loading (Alternative)**

If you prefer not to enable automatic environment injection, manually load the `.env` file:

```bash
# Load environment variables manually
export $(grep -v '^#' .env | xargs)

# Or use python-dotenv in your terminal session
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print('Environment loaded')"
```

**Verification:**

After enabling the setting, verify environment variables are loaded:

```bash
# In VS Code integrated terminal
echo $WEATHER_API_KEY
echo $OTEL_SERVICE_NAME

# If variables are not showing, restart the terminal or source .env manually
```

**Best Practice:**

Always verify your environment before running the application:

```bash
# 1. Check virtual environment is active
which python  # Must point to venv

# 2. Check environment variables are loaded
echo $WEATHER_API_KEY  # Should show your API key

# 3. Run the application
python app.py
```

### Import Errors

```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# Verify installation
pip list | grep -E "flask|opentelemetry|requests"
```

### API Key Issues

If you see "Weather API not configured" error:

1. Verify `.env` file exists in `weather-dashboard/` directory
2. Ensure `WEATHER_API_KEY` is set to a valid key
3. Get a free API key from: https://openweathermap.org/api
4. Restart the application after updating `.env`

### Port Already in Use

If port 5000 is already in use:

```bash
# Use a different port
export PORT=5001
python app.py

# Or find and kill the process using port 5000
lsof -ti:5000 | xargs kill -9
```

## Integration with Parent Project

This weather dashboard demonstrates the observability concepts documented in the parent project's README.md:

- **Traces and Spans**: Hierarchical operation tracking
- **OpenTelemetry Integration**: Standards-compliant instrumentation
- **Agent Framework**: Custom visualization utilities
- **Security Considerations**: Input validation and error handling
- **Spec-Driven Development**: API contract adherence

Refer to the parent project's README.md for theoretical foundations and additional observability tools.

## Contributing

This is a demonstration project for educational purposes. To contribute:

1. Ensure all tests pass: `pytest tests/ -v`
2. Follow Python PEP 8 style guidelines
3. Add tests for new features
4. Update documentation as needed

## License

MIT License - see parent project for details.

## Resources

- [OpenTelemetry Python Documentation](https://opentelemetry.io/docs/languages/python/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [OpenWeatherMap API](https://openweathermap.org/api)
- [Langfuse Documentation](https://langfuse.com/docs)
- [Parent Project README](../README.md) - Comprehensive observability guide
