from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Tool + Resource Demo")

@mcp.tool()
def add(a: int, b: int) -> int:
    return a + b

@mcp.resource("greeting://{name}")
def greeting(name: str) -> str:
    return f"Hello {name} — served via MCP resource!"

if __name__ == "__main__":
    mcp.run()
