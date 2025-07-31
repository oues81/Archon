import asyncio
import inspect
from functools import wraps
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

class FastMCP:
    def __init__(self, name: str, log_level: str = "INFO", request_timeout: int = 30):
        self.app = FastAPI(title=name)
        self.log_level = log_level
        self.timeout = request_timeout
        self.tools = {}
        self.app.add_api_route("/", self.handle_rpc, methods=["POST"])
        self.app.add_api_route("/health", self.health_check, methods=["GET"])

    async def health_check(self):
        return JSONResponse({"status": "ok"})

    def tool(self):
        def decorator(func):
            tool_name = func.__name__
            self.tools[tool_name] = func
            @wraps(func)
            async def wrapper(*args, **kwargs):
                return await func(*args, **kwargs)
            return wrapper
        return decorator

    async def handle_rpc(self, request: Request):
        body = await request.json()
        method = body.get("method")
        params = body.get("params", {})
        request_id = body.get("id")

        if not method or method not in self.tools:
            return JSONResponse({
                "jsonrpc": "2.0",
                "error": {"code": -32601, "message": "Method not found"},
                "id": request_id
            }, status_code=404)

        tool_func = self.tools[method]
        
        try:
            if isinstance(params, dict):
                result = await tool_func(**params)
            elif isinstance(params, list):
                result = await tool_func(*params)
            else:
                 return JSONResponse({
                    "jsonrpc": "2.0",
                    "error": {"code": -32602, "message": "Invalid params"},
                    "id": request_id
                }, status_code=400)

            return JSONResponse({
                "jsonrpc": "2.0",
                "result": result,
                "id": request_id
            })
        except Exception as e:
            return JSONResponse({
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": f"Internal error: {str(e)}"},
                "id": request_id
            }, status_code=500)

    def run(self, host: str = "0.0.0.0", port: int = 8100):
        uvicorn.run(self.app, host=host, port=port, log_level=self.log_level.lower())