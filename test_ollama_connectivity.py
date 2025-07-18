import os
import requests
import sys
import json

def test_ollama_connection():
    base_url = os.getenv('BASE_URL', 'http://host.docker.internal:11434')
    url = f"{base_url}/api/tags"
    
    print(f"Testing connection to Ollama at {url}")
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            print("✅ Successfully connected to Ollama!")
            print("\nAvailable models:")
            for model in data.get('models', []):
                print(f"- {model.get('name', 'N/A')} (size: {model.get('size', 'N/A')} bytes)")
            
            # Test a basic generation
            print("\nTesting a basic generation with the first available model...")
            if data.get('models') and len(data.get('models')) > 0:
                model_name = data.get('models')[0].get('name')
                generate_url = f"{base_url}/api/generate"
                generate_payload = {
                    "model": model_name,
                    "prompt": "Hello, world!"
                }
                generate_response = requests.post(
                    generate_url, 
                    data=json.dumps(generate_payload)
                )
                
                if generate_response.status_code == 200:
                    print(f"✅ Generation successful with model {model_name}!")
                else:
                    print(f"❌ Generation failed: {generate_response.status_code}")
                    print(generate_response.text)
            
            return True
        else:
            print(f"❌ Error: Received status code {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Failed to connect to Ollama: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_ollama_connection()
    sys.exit(0 if success else 1)
