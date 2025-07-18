
import requests

def test_ollama():
    try:
        response = requests.get("http://host.docker.internal:11434/api/tags")
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"Available models: {[model.get('name') for model in models]}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Exception: {str(e)}")

if __name__ == "__main__":
    test_ollama()
