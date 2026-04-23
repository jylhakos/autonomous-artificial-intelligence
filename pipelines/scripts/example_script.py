#!/usr/bin/env python3
"""
Example script demonstrating virtual environment usage.
This script uses the 'requests' library to check internet connectivity.
"""

import sys
import requests

def check_internet_connection():
    """Check if we can reach the internet."""
    try:
        response = requests.get('https://www.google.com', timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False

def main():
    print("=" * 60)
    print("Example Script - Virtual Environment Demo")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print()
    
    print("Testing internet connectivity with 'requests' library...")
    if check_internet_connection():
        print("✓ Internet connection is working")
        print("✓ 'requests' library is installed and functional")
    else:
        print("✗ Internet connection check failed")
    
    print()
    print("This script demonstrates that the virtual environment")
    print("has the 'requests' package installed correctly.")
    print("=" * 60)

if __name__ == "__main__":
    main()
