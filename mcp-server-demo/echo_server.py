from mcp.server.fastmcp import FastMCP

mcp = FastMCP("EchoServer", stateless_http=True)

@mcp.tool()
def echo(message: str) -> str:
    return f"Echo: {message}"

if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8001)
