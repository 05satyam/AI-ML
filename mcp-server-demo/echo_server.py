from mcp.server.fastmcp import FastMCP

mcp = FastMCP("EchoServer", stateless_http=True)

@mcp.tool()
def echo(message: str) -> str:
    return f"Echo: {message}"

if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8001)



# below code shows on how to run stdio:
# echo_server.py

# from mcp import Tool, Server
# import asyncio

# # Your tool function
# async def echo(message: str) -> str:
#     return f"Echo: {message}"

# # MCP Server with the tool
# server = Server(
#     tools=[Tool(name="echo", func=echo, input_schema={"message": str})]
# )

# # Entry point to run using stdio
# if __name__ == "__main__":
#     asyncio.run(server.run_stdio())
