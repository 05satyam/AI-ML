import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("GitHubServer", stateless_http=True)

@mcp.tool()
async def get_github_user(username: str) -> dict:
    """Fetch GitHub user information."""
    url = f"https://api.github.com/users/{username}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()

if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8003)
