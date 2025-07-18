
import requests
import uuid
import json

def create_thread():
    thread_id = str(uuid.uuid4())
    print(f"Created thread: {thread_id}")
    return thread_id

def run_agent(thread_id, prompt):
    print(f"Running agent with thread {thread_id} and prompt: {prompt}")
    try:
        response = requests.post(
            "http://localhost:8110/invoke",
            json={
                "message": prompt,
                "thread_id": thread_id,
                "is_first_message": True
            },
            timeout=10
        )
        print(f"Status code: {response.status_code}")
        return response.json()
    except requests.exceptions.Timeout:
        print("Request timed out")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    thread_id = create_thread()
    result = run_agent(thread_id, "Create a simple Hello World agent that returns 'Hello, World!'")
    print(f"Result: {result}")
