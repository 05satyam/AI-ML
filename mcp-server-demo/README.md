# Main streamlit app files are: mounted_server.py and app.py
 - check the respective files on step to execute the app.


### 📧 Gmail MCP Setup Guide
To enable the `send_email` tool in this app, you need to set up Gmail API:

#### 1. Create Google Cloud Project
- Visit: https://console.cloud.google.com/
- Create a new project

#### 2. Enable Gmail API
- Navigate to: **APIs & Services → Library**
- Search for "Gmail API" and click **Enable**

#### 3. Configure OAuth Consent Screen
- Navigate to: **APIs & Services → OAuth consent screen**
- Choose "External", add app name, email, and yourself as test user

#### 4. Create OAuth Credentials
- Go to: **APIs & Services → Credentials → Create Credentials → OAuth client ID**
- Choose: Application type → Desktop App
- Download the generated `credentials.json`

#### 5. First-time Authentication
- Save `credentials.json` in your project root
- On first run, browser window will open → grant permission
- This creates a local `token.json` file to authenticate future requests

Use a test Gmail account and keep your credentials private.


# Why Did We Use mount() in FastAPI?
  - mount() lets you embed multiple MCP servers into a single FastAPI app under different paths

## Can We Do External Tool Mounts?
  - Yes, you can expose external tools (from other systems, APIs, scripts, or microservices) by:
  - Wrapping them in MCP @mcp.tool() decorators.
  - Using HTTP clients (like httpx) to call remote APIs.
  - Streaming output back via MCP.