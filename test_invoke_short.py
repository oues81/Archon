
import requests
import uuid
import json
import sys

def test_invoke():
    thread_id = str(uuid.uuid4())
    prompt = "Hello, just say hi back"
    
    print(f"Thread ID: {thread_id}")
    print(f"Prompt: {prompt}")
    
    try:
        # First, check the health endpoint
        health_response = requests.get("http://localhost:8110/health")
        print(f"Health status: {health_response.status_code} - {health_response.json()}")
        
        # Now try to invoke the agent with a very short timeout
        request_data = {
            "message": prompt,
            "thread_id": thread_id,
            "is_first_message": True
        }
        
        print(f"Sending request to /invoke: {json.dumps(request_data)}")
        
        response = requests.post(
            "http://localhost:8110/invoke",
            json=request_data,
            timeout=3  # Very short timeout
        )
        
        print(f"Response status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {result}")
        else:
            print(f"Error: {response.text}")
    except requests.exceptions.Timeout:
        print("Request timed out after 3 seconds")
        print("This is expected behavior as the agent processing takes longer")
        print("Check the logs to see if the agent is processing the request")
    except Exception as e:
        print(f"Exception: {str(e)}")
        print(f"Exception type: {type(e)}")

if __name__ == "__main__":
    test_invoke()
