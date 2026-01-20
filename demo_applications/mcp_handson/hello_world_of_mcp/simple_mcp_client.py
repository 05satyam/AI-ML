import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def extract_tool_result(res):
    # Prefer structuredContent if present
    if getattr(res, "structuredContent", None):
        sc = res.structuredContent
        if isinstance(sc, dict) and "result" in sc:
            return sc["result"]
        return sc

    # Fallback: look for first text content item
    for item in getattr(res, "content", []) or []:
        txt = getattr(item, "text", None)
        if txt is not None:
            return txt
    return res


def extract_resource_text(r):
    # Most versions return .contents list with TextResourceContents
    contents = getattr(r, "contents", None)
    if contents:
        first = contents[0]
        return getattr(first, "text", None) or str(first)

    # Fallback: some versions use .text
    t = getattr(r, "text", None)
    if t is not None:
        return t

    return str(r)


async def main():
    params = StdioServerParameters(
        command=sys.executable,
        args=["simple_mcp_server.py"],
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("✅ Tools:", [t.name for t in tools.tools])

            add_res = await session.call_tool("add", {"a": 3, "b": 4})
            print("✅ add(3,4) =", extract_tool_result(add_res))

            if hasattr(session, "read_resource"):
                r = await session.read_resource("greeting://HELLO_WORLD")
            else:
                r = await session.get_resource("greeting://HELLO_WORLD")

            print("✅ greeting://HELLO_WORLD =", extract_resource_text(r))


if __name__ == "__main__":
    asyncio.run(main())
