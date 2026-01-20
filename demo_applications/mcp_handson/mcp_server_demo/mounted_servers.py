### mounted_servers.py
### to run: uvicorn mounted_servers:app --port 8000
# uv pip install google-api-python-client google-auth google-auth-oauthlib
# uv pip install langchain openai
# uv pip install langchain langchain-community openai




from fastapi import FastAPI
import contextlib
import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base

'''
Stateless servers don’t hold user or session-specific data in memory.
You can run multiple instances behind a load balancer.
Any instance can handle any request without needing shared memory

Return streamed responses using Server-Sent Events (SSE) or chunked transfer encoding (depending on the tool implementation)

'''
# Echo server
echo_mcp = FastMCP("EchoServer", stateless_http=True)

@echo_mcp.tool()
def echo(message: str) -> str:
    return f"Echo: {message}"

@echo_mcp.resource("hello://{name}")
def greeting(name: str) -> str:
    return f"Hello, {name}! Welcome to the MCP world."

@echo_mcp.prompt()
def help_with_code(error: str) -> list[base.Message]:
    return [
        base.UserMessage(f"I'm getting this error: {error}"),
        base.AssistantMessage("Let’s work through this together. What have you tried so far?")
    ]

@echo_mcp.tool()
def directions(start: str, end: str) -> str:
    return f"Directions from {start} to {end}:\n1. Take the highway.\n2. Pass through Agra.\n3. Arrive at {end}."

@echo_mcp.tool()
def maps(start: str, end: str) -> str:
    return f"Map route: {start} → Agra → {end}"



# Math server
math_mcp = FastMCP("MathServer", stateless_http=True)

@math_mcp.tool()
def add(a: int, b: int) -> int:
    return a + b

@math_mcp.tool()
def subtract(a: int, b: int) -> int:
    return a - b

@math_mcp.prompt()
def explain_add(a: int, b: int) -> str:
    return f"Explain what {a} + {b} is and how it works."

# GitHub server
github_mcp = FastMCP("GitHubServer", stateless_http=True)

@github_mcp.tool()
async def get_github_user(username: str) -> dict:
    url = f"https://api.github.com/users/{username}"
    async with httpx.AsyncClient() as client:
        r = await client.get(url)
        return r.json()

# gmail server
# Gmail server
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from email.message import EmailMessage
import os
import base64

gmail_mcp = FastMCP("GmailServer", stateless_http=True)

TOKEN_FILE = "token.json"
CREDENTIALS_FILE = "credentials.json"
SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

@gmail_mcp.tool()
def send_email(to: str, subject: str, message: str) -> str:
    try:
        creds = None
        if os.path.exists(TOKEN_FILE):
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
            with open(TOKEN_FILE, "w") as token:
                token.write(creds.to_json())

        service = build("gmail", "v1", credentials=creds)

        email = EmailMessage()
        email.set_content(message)
        email["To"] = to
        email["From"] = "me"
        email["Subject"] = subject

        encoded_message = base64.urlsafe_b64encode(email.as_bytes()).decode()
        result = service.users().messages().send(userId="me", body={"raw": encoded_message}).execute()

        return f"✅ Email sent: {result['id']}"
    except Exception as e:
        return f"❌ Failed to send email: {str(e)}"


# FastAPI app to mount them
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    async with contextlib.AsyncExitStack() as stack:
        await stack.enter_async_context(echo_mcp.session_manager.run())
        await stack.enter_async_context(math_mcp.session_manager.run())
        await stack.enter_async_context(github_mcp.session_manager.run())
        await stack.enter_async_context(gmail_mcp.session_manager.run())
        yield

app = FastAPI(lifespan=lifespan)
app.mount("/echo", echo_mcp.streamable_http_app())
app.mount("/math", math_mcp.streamable_http_app())
app.mount("/github", github_mcp.streamable_http_app())
app.mount("/gmail", gmail_mcp.streamable_http_app())