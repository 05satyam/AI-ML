from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MathServer", stateless_http=True)

@mcp.tool()
def add(a: int, b: int) -> int:
    return a + b

@mcp.tool()
def subtract(a: int, b: int) -> int:
    return a - b

if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8002)
