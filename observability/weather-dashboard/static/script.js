// Weather Dashboard JavaScript
// Handles API calls and UI updates with observability considerations

document.addEventListener('DOMContentLoaded', function() {
    const cityInput = document.getElementById('cityInput');
    const searchBtn = document.getElementById('searchBtn');
    const weatherDisplay = document.getElementById('weatherDisplay');
    const errorDisplay = document.getElementById('errorDisplay');
    const loadingDisplay = document.getElementById('loadingDisplay');
    
    // Add event listeners
    searchBtn.addEventListener('click', searchWeather);
    cityInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            searchWeather();
        }
    });
    
    // Load default city on page load
    fetchWeatherData('London');
    
    function searchWeather() {
        const city = cityInput.value.trim();
        if (city) {
            fetchWeatherData(city);
        }
    }
    
    async function fetchWeatherData(city) {
        // Hide all display sections
        hideAllDisplays();
        
        // Show loading
        loadingDisplay.classList.remove('hidden');
        
        // Record start time for client-side performance tracking
        const startTime = performance.now();
        
        try {
            const response = await fetch(`/api/weather/${encodeURIComponent(city)}`);
            
            // Calculate client-side latency
            const endTime = performance.now();
            const latency = (endTime - startTime).toFixed(2);
            
            console.log(`Weather API call completed in ${latency}ms`);
            console.log(`HTTP Status: ${response.status}`);
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to fetch weather data');
            }
            
            const data = await response.json();
            
            // Log successful data retrieval
            console.log('Weather data received:', data);
            
            // Display weather data
            displayWeatherData(data);
            
        } catch (error) {
            console.error('Error fetching weather:', error);
            displayError(error.message);
        } finally {
            loadingDisplay.classList.add('hidden');
        }
    }
    
    function displayWeatherData(data) {
        hideAllDisplays();
        
        // Populate weather card with data
        document.getElementById('cityName').textContent = data.city || 'Unknown';
        document.getElementById('country').textContent = data.country || '--';
        document.getElementById('temperature').textContent = 
            data.temperature ? Math.round(data.temperature) : '--';
        document.getElementById('description').textContent = 
            data.weather || 'No description';
        document.getElementById('feelsLike').textContent = 
            data.feels_like ? `${Math.round(data.feels_like)}°C` : '--';
        document.getElementById('humidity').textContent = 
            data.humidity ? `${data.humidity}%` : '--';
        document.getElementById('pressure').textContent = 
            data.pressure ? `${data.pressure} hPa` : '--';
        document.getElementById('windSpeed').textContent = 
            data.wind_speed ? `${data.wind_speed} m/s` : '--';
        
        // Show weather display
        weatherDisplay.classList.remove('hidden');
        
        // Log observability information
        console.log('Observability: Weather data displayed successfully');
        console.log('Timestamp:', data.timestamp);
    }
    
    function displayError(message) {
        hideAllDisplays();
        
        document.getElementById('errorMessage').textContent = 
            message || 'An unexpected error occurred';
        errorDisplay.classList.remove('hidden');
        
        // Log error for observability
        console.error('Weather fetch error displayed:', message);
    }
    
    function hideAllDisplays() {
        weatherDisplay.classList.add('hidden');
        errorDisplay.classList.add('hidden');
        loadingDisplay.classList.add('hidden');
    }
    
    // Health check on page load
    async function checkHealth() {
        try {
            const response = await fetch('/api/health');
            const health = await response.json();
            console.log('Health check:', health);
        } catch (error) {
            console.error('Health check failed:', error);
        }
    }
    
    // Run health check
    checkHealth();
    
    // Add performance observer for observability
    if ('PerformanceObserver' in window) {
        const observer = new PerformanceObserver((list) => {
            for (const entry of list.getEntries()) {
                if (entry.entryType === 'navigation') {
                    console.log('Page load performance:');
                    console.log(`  DNS: ${entry.domainLookupEnd - entry.domainLookupStart}ms`);
                    console.log(`  TCP: ${entry.connectEnd - entry.connectStart}ms`);
                    console.log(`  Request: ${entry.responseStart - entry.requestStart}ms`);
                    console.log(`  Response: ${entry.responseEnd - entry.responseStart}ms`);
                    console.log(`  DOM: ${entry.domContentLoadedEventEnd - entry.domContentLoadedEventStart}ms`);
                    console.log(`  Total: ${entry.loadEventEnd - entry.fetchStart}ms`);
                }
            }
        });
        
        try {
            observer.observe({ entryTypes: ['navigation'] });
        } catch (e) {
            // Browser might not support navigation timing
            console.log('Navigation timing not available');
        }
    }
});
