"""
Weather Dashboard Application with OpenTelemetry Observability

This Flask application demonstrates AI agent observability patterns by:
1. Instrumenting HTTP requests with OpenTelemetry traces
2. Creating custom spans for business logic
3. Recording exceptions and performance metrics
4. Exporting telemetry to OTLP-compatible backends
"""

import os
from flask import Flask, render_template, jsonify, request
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.sdk.resources import Resource
from dotenv import load_dotenv
import requests
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure OpenTelemetry with resource attributes
resource = Resource.create({
    "service.name": os.getenv("OTEL_SERVICE_NAME", "weather-dashboard"),
    "service.version": "1.0.0",
    "deployment.environment": os.getenv("OTEL_RESOURCE_ATTRIBUTES", "development")
})

# Set up tracer provider
trace.set_tracer_provider(TracerProvider(resource=resource))

# Configure OTLP exporter if endpoint is provided
otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
if otlp_endpoint:
    otlp_exporter = OTLPSpanExporter(
        endpoint=otlp_endpoint,
        headers={"Authorization": os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "")}
    )
    trace.get_tracer_provider().add_span_processor(
        BatchSpanProcessor(otlp_exporter)
    )
else:
    # Fallback to console exporter for local development
    console_exporter = ConsoleSpanExporter()
    trace.get_tracer_provider().add_span_processor(
        BatchSpanProcessor(console_exporter)
    )

# Create Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')

# Instrument Flask for automatic request tracing
FlaskInstrumentor().instrument_app(app)

# Get tracer for custom spans
tracer = trace.get_tracer(__name__)


@app.route('/')
def index():
    """Render the weather dashboard home page."""
    with tracer.start_as_current_span("render-homepage") as span:
        span.set_attribute("page.name", "index")
        span.set_attribute("user.agent", request.headers.get("User-Agent", "unknown"))
        return render_template('index.html')


@app.route('/api/weather/<city>')
def get_weather(city):
    """
    Fetch weather data for a specified city with full observability.
    
    Args:
        city: Name of the city to fetch weather for
        
    Returns:
        JSON response with weather data or error message
    """
    with tracer.start_as_current_span("fetch-weather") as span:
        span.set_attribute("city.name", city)
        span.set_attribute("operation.type", "weather-api-call")
        span.set_attribute("request.timestamp", datetime.utcnow().isoformat())
        
        try:
            # Validate input
            if not city or len(city) > 100:
                span.set_attribute("validation.status", "failed")
                span.add_event("Input validation failed")
                return jsonify({"error": "Invalid city name"}), 400
            
            # Fetch weather data from external API
            api_key = os.getenv("WEATHER_API_KEY")
            if not api_key:
                span.set_attribute("error.type", "configuration")
                span.record_exception(ValueError("WEATHER_API_KEY not configured"))
                return jsonify({"error": "Weather API not configured"}), 500
            
            # Make API request with tracing
            with tracer.start_as_current_span("external-api-call") as api_span:
                url = f"https://api.openweathermap.org/data/2.5/weather"
                params = {
                    "q": city,
                    "appid": api_key,
                    "units": "metric"
                }
                
                api_span.set_attribute("http.url", url)
                api_span.set_attribute("http.method", "GET")
                
                response = requests.get(url, params=params, timeout=5)
                
                api_span.set_attribute("http.status_code", response.status_code)
                api_span.set_attribute("http.response_size", len(response.content))
            
            response.raise_for_status()
            data = response.json()
            
            # Extract and enrich weather data
            with tracer.start_as_current_span("process-weather-data") as process_span:
                weather_info = {
                    "city": data.get("name"),
                    "country": data.get("sys", {}).get("country"),
                    "temperature": data.get("main", {}).get("temp"),
                    "feels_like": data.get("main", {}).get("feels_like"),
                    "humidity": data.get("main", {}).get("humidity"),
                    "pressure": data.get("main", {}).get("pressure"),
                    "weather": data.get("weather", [{}])[0].get("description"),
                    "wind_speed": data.get("wind", {}).get("speed"),
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                process_span.set_attribute("temperature.celsius", weather_info["temperature"])
                process_span.set_attribute("humidity.percent", weather_info["humidity"])
            
            span.set_attribute("response.status", "success")
            span.add_event("Weather data fetched successfully", {
                "city": weather_info["city"],
                "temperature": weather_info["temperature"]
            })
            
            return jsonify(weather_info)
            
        except requests.Timeout as e:
            span.set_attribute("response.status", "error")
            span.set_attribute("error.type", "timeout")
            span.record_exception(e)
            return jsonify({"error": "Weather service timeout"}), 504
            
        except requests.HTTPError as e:
            span.set_attribute("response.status", "error")
            span.set_attribute("error.type", "http_error")
            span.record_exception(e)
            
            if response.status_code == 404:
                return jsonify({"error": "City not found"}), 404
            return jsonify({"error": "Weather service error"}), 502
            
        except Exception as e:
            span.set_attribute("response.status", "error")
            span.set_attribute("error.type", "unexpected")
            span.record_exception(e)
            app.logger.error(f"Unexpected error fetching weather: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500


@app.route('/api/health')
def health_check():
    """Health check endpoint for monitoring."""
    with tracer.start_as_current_span("health-check") as span:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "weather-dashboard",
            "version": "1.0.0"
        }
        span.set_attribute("health.status", "healthy")
        return jsonify(health_status)


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    app.logger.error(f"Internal error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    # Run the application
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    
    print(f"Starting Weather Dashboard on port {port}")
    print(f"OpenTelemetry endpoint: {otlp_endpoint or 'Console (local development)'}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )
