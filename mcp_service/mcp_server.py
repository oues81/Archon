from mcp.server.fastmcp import FastMCP
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, List
import threading
import requests
import asyncio
import uuid
import sys
import os

# Load environment variables from .env file
load_dotenv()

# Initialize FastMCP server with DEBUG logging level for better visibility
mcp = FastMCP("archon", log_level="DEBUG")

# Store active threads
active_threads: Dict[str, List[str]] = {}

# FastAPI service URL
GRAPH_SERVICE_URL = os.getenv("GRAPH_SERVICE_URL", "http://localhost:8100")

def write_to_log(message: str):
    """Write a message to the logs.txt file in the workbench directory and print to stderr.
    
    Args:
        message: The message to log
    """
    try:
        # Get the directory one level up from the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        workbench_dir = os.path.join(parent_dir, "workbench")
        log_path = os.path.join(workbench_dir, "logs.txt")
        os.makedirs(workbench_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        # Print to stderr (visible in Docker logs)
        print(log_entry, file=sys.stderr)
        
        # Also write to log file
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{log_entry}\n")
            
    except Exception as e:
        # If logging fails, at least print the error to stderr
        print(f"[ERROR] Failed to write to log: {str(e)}", file=sys.stderr)
        print(f"[ERROR] Original message: {message}", file=sys.stderr)

@mcp.tool()
async def create_thread() -> str:
    """Create a new conversation thread for Archon.
    Always call this tool before invoking Archon for the first time in a conversation.
    (if you don't already have a thread ID)
    
    Returns:
        str: A unique thread ID for the conversation
    """
    thread_id = str(uuid.uuid4())
    active_threads[thread_id] = []
    write_to_log(f"Created new thread: {thread_id}")
    return thread_id


def _make_request(thread_id: str, user_input: str, config: dict) -> str:
    """Make synchronous request to graph service"""
    try:
        response = requests.post(
            f"{GRAPH_SERVICE_URL}/invoke",
            json={
                "message": user_input,
                "thread_id": thread_id,
                "is_first_message": not active_threads[thread_id],
                "config": config
            },
            timeout=300  # 5 minute timeout for long-running operations
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        write_to_log(f"Request timed out for thread {thread_id}")
        raise TimeoutError("Request to graph service timed out. The operation took longer than expected.")
    except requests.exceptions.RequestException as e:
        write_to_log(f"Request failed for thread {thread_id}: {str(e)}")
        raise


@mcp.tool()
async def run_agent(thread_id: str, user_input: str) -> str:
    """Run the Archon agent with user input.
    Only use this tool after you have called create_thread in this conversation to get a unique thread ID.
    If you already created a thread ID in this conversation, do not create another one. Reuse the same ID.
    After you receive the code from Archon, always implement it into the codebase unless asked not to.

    After using this tool and implementing the code it gave back, ask the user if they want you to refine the agent
    autonomously (they can just say 'refine') or they can just give feedback and you'll improve the agent that way.

    If they want to refine the agent, just give 'refine' for user_input.
    
    Args:
        thread_id: The conversation thread ID
        user_input: The user's message to process
    
    Returns:
        str: The agent's response which generally includes the code for the agent
    """
    if thread_id not in active_threads:
        write_to_log(f"Error: Thread not found - {thread_id}")
        raise ValueError("Thread not found")

    write_to_log(f"Processing message for thread {thread_id}: {user_input}")

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    
    try:
        result = await asyncio.to_thread(_make_request, thread_id, user_input, config)
        active_threads[thread_id].append(user_input)
        return result['response']
        
    except Exception as e:
        raise


if __name__ == "__main__":
    try:
        write_to_log("Démarrage du serveur MCP...")
        
        # Run MCP server with stdio transport
        write_to_log("Configuration du serveur MCP...")
        
        # Démarrer le serveur MCP dans un thread séparé
        import threading
        import time
        
        def run_mcp():
            try:
                mcp.run(transport='stdio')
            except Exception as e:
                write_to_log(f"Erreur dans le thread MCP: {str(e)}")
                os._exit(1)
        
        # Démarrer le thread MCP
        mcp_thread = threading.Thread(target=run_mcp, daemon=True)
        mcp_thread.start()
        
        write_to_log("Serveur MCP démarré avec succès")
        
        # Garder le processus principal en vie
        while True:
            time.sleep(1)
            
    except Exception as e:
        error_msg = f"Erreur lors du démarrage du serveur MCP: {str(e)}"
        write_to_log(error_msg)
        # Écrire l'erreur dans stderr pour qu'elle soit visible dans les logs Docker
        sys.stderr.write(f"{error_msg}\n")
        raise