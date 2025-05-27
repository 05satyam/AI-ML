'''
python3 -m pip install streamlit

1. Verify the MCP Package Installation
uv pip list | grep mcp

installed mcp version: uv pip show mcp

install uv streamlit:
uv pip install streamlit

run streamlit app:
uv run streamlit run app.py


'''

import streamlit as st
import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

st.set_page_config(page_title="MCP  Presentation", layout="wide")
st.title("🔌 Model Context Protocol")

st.markdown("""
Welcome to a production-ready demo and education platform for **MCP**.
This app shows how LLMs interact with tools, data, and agents using standardized protocols.

### 🔄 Analogy with REST APIs:
- **Resource** → Like a REST `GET` endpoint: it provides data.
- **Tool** → Like a REST `POST` handler: it executes business logic.
- **Prompt** → Like a reusable request/response template or workflow definition.

Together, they offer a clean interface to bridge AI reasoning with structured data and actions.
""")

ACTIONS_LOG = []
### 🔁 Shared async tool call function
# This function is conceptually similar to a REST API POST request handler that triggers business logic.
async def call_tool(server_url, tool_name, args):
    async with streamablehttp_client(server_url) as (r, w, _):
        async with ClientSession(r, w) as session:
            result = await session.call_tool(tool_name, args)
            ACTIONS_LOG.append(f"Called `{tool_name}` at `{server_url}` with `{args}` => {result}")
            return result

### 📊 Tabs
tab_titles = [
    "📚 Intro", "🛠️ Echo", "➕ Math Tools", "🌐 GitHub",
    "🧠 Protocol Diagrams", "🧰 Tool Explorer", "📦 Resource Fetcher", "💬 Prompt Tester", 
    "📤 Send Summary Email", "🧪 Resource + Prompt Demo"
]
tabs = st.tabs(tab_titles)

async def send_summary_email(to: str):
    summary = "\n".join(ACTIONS_LOG)
    print(f"summary: {summary}")
    llm = ChatOpenAI(model="gpt-4")
    response = llm([HumanMessage(content=f"Summarize this log of actions in 5 lines:\n{summary}")])
    msg = response.content
    email_payload = {
        "to": to,
        "subject": "🧠 MCP Agent Action Summary",
        "message": msg
    }
    return await call_tool("http://localhost:8000/gmail/mcp", "send_email", email_payload)


# ----------------- Tab 0 -----------------
with tabs[0]:
    st.header("📚 What is MCP?")
    st.markdown("""
    MCP allows AI agents to use tools, access data, and follow prompts in a structured way.
    It's like giving an LLM a toolbox and instructions on how to use it — securely and efficiently.

    ### Why it matters:
    - Decouples tools from models.
    - Enables reproducibility and auditing.
    - Facilitates multi-agent communication (see A2A).
    """)

# ----------------- Tab 1 -----------------
with tabs[1]:
    st.subheader("Echo Tool via MCP")
    msg = st.text_input("Type a message:")
    if st.button("Send to Echo Tool"):
        result = asyncio.run(call_tool("http://localhost:8000/echo/mcp", "echo", {"message": msg}))
        st.success(result)

# ----------------- Tab 2 -----------------
with tabs[2]:
    st.subheader("Math Tools")
    a = st.number_input("A", 0)
    b = st.number_input("B", 0)
    if st.button("Add"):
        r = asyncio.run(call_tool("http://localhost:8000/math/mcp", "add", {"a": a, "b": b}))
        st.info(f"Result: {r}")
    if st.button("Subtract"):
        r = asyncio.run(call_tool("http://localhost:8000/math/mcp", "subtract", {"a": a, "b": b}))
        st.warning(f"Result: {r}")

# ----------------- Tab 3 -----------------
with tabs[3]:
    st.subheader("GitHub Integration Tool")
    uname = st.text_input("GitHub Username:")
    if st.button("Fetch GitHub Info"):
        user = asyncio.run(call_tool("http://localhost:8000/github/mcp", "get_github_user", {"username": uname}))
        st.json(user)

# ----------------- Tab 4 -----------------
with tabs[4]:
    st.image("assets/mcp_architecture.png", caption="MCP Architecture")
    st.markdown("""
    ### MCP
    - **MCP** is one agent talking to tools and data.

    Together, they form the base of robust AI ecosystems.
    """)

# ----------------- Tab 5 -----------------
with tabs[5]:
    st.subheader("🧰 Available Tools from MCP Servers")
    server_urls = {
        "Echo": "http://localhost:8000/echo/mcp",
        "Math": "http://localhost:8000/math/mcp",
        
        "Gmail": "http://localhost:8000/gmail/mcp"
    }
    for label, url in server_urls.items():
        st.markdown(f"**🔌 {label} Server**")

        async def list_all_tools():
            async with streamablehttp_client(url) as (r, w, _):
                async with ClientSession(r, w) as session:
                    await session.initialize()
                    raw = await session.list_tools()
                    if hasattr(raw, 'tools'):
                        return raw.tools
                    return []

        try:
            tools = asyncio.run(list_all_tools())
            if not tools:
                st.info("No tools found.")
            for tool in tools:
                name = getattr(tool, 'name', 'Unknown')
                description = getattr(tool, 'description', '(no description)')
                st.code(f"🛠️ {name}: {description}", language="yaml")

                with st.expander(f"Test '{name}'"):
                    args_input = st.text_area(f"Enter JSON args for {name}", "{}", key=f"{label}_{name}")
                    if st.button(f"Invoke {name}", key=f"btn_{label}_{name}"):
                        try:
                            parsed_args = json.loads(args_input)
                            result = asyncio.run(call_tool(url, name, parsed_args))
                            st.success(result)
                        except Exception as e:
                            st.error(f"Invocation failed: {e}")
        except Exception as e:
            st.error(f"Error fetching tools from {label}: {e}")


# ----------------- Tab 6 -----------------
with tabs[6]:
    st.subheader("📦 Resource Fetcher")
    # This tab demonstrates accessing a resource — similar to a REST API GET request.
    name = st.text_input("Enter name to fetch greeting:")
    if st.button("Fetch Greeting Resource"):
        async def read_resource():
            async with streamablehttp_client("http://localhost:8000/echo/mcp") as (r, w, _):
                async with ClientSession(r, w) as session:
                    await session.initialize()
                    return await session.read_resource(f"hello://{name}")
        try:
            result = asyncio.run(read_resource())
            if result.contents and hasattr(result.contents[0], "text"):
                st.code(result.contents[0].text)
            else:
                st.warning("No content found.")   
        except Exception as e:
            st.error(f"Error fetching resource: {e}")

# ----------------- Tab 7 -----------------
# ----------------- Tab 7 -----------------
with tabs[7]:
    st.subheader("💬 Prompt Tester (Auto-Discover)")

    prompt_servers = {
        "Echo": "http://localhost:8000/echo/mcp",
        "Math": "http://localhost:8000/math/mcp",
        "Gmail": "http://localhost:8000/gmail/mcp",
    }

    selected_server = st.selectbox("Choose a server", list(prompt_servers.keys()))
    server_url = prompt_servers[selected_server]

    async def fetch_prompts():
        async with streamablehttp_client(server_url) as (r, w, _):
            async with ClientSession(r, w) as session:
                await session.initialize()
                res = await session.list_prompts()
                return res.prompts

    try:
        prompts = asyncio.run(fetch_prompts())
        if not prompts:
            st.info("No prompts found.")
        else:
            selected_prompt_name = st.selectbox("Pick a prompt", [p.name for p in prompts])
            prompt_obj = next(p for p in prompts if p.name == selected_prompt_name)

            st.markdown(f"📝 **{prompt_obj.description or '(No description)'}**")
            prompt_args = {}
            if prompt_obj.arguments:
                for arg in prompt_obj.arguments:
                    val = st.text_input(f"{arg.name} ({'required' if arg.required else 'optional'})", "")
                    if val:
                        prompt_args[arg.name] = val

            if st.button("Run Prompt"):
                async def run_prompt():
                    async with streamablehttp_client(server_url) as (r, w, _):
                        async with ClientSession(r, w) as session:
                            await session.initialize()
                            return await session.get_prompt(selected_prompt_name, prompt_args)

                try:
                    result = asyncio.run(run_prompt())
                    if result.description:
                        st.markdown(f"📝 Prompt Description: {result.description}")
                    st.markdown("🧠 Messages:")
                    for m in result.messages:
                        st.markdown(f"**{m.role}**: {m.content.text}")
                except Exception as e:
                    st.error(f"Prompt error: {e}")
    except Exception as e:
        st.error(f"Failed to list prompts: {e}")


# send email with summary of actions
with tabs[8]:
    st.subheader("📤 Email Me MCP Agent Summary")
    email_input = st.text_input("Your Email:")
    if st.button("Send Agent Summary via Email"):
        if email_input:
            try:
                result = asyncio.run(send_summary_email(email_input))
                st.success(result)
            except Exception as e:
                st.error(f"Failed to send summary: {e}")
        else:
            st.warning("Please enter a valid email address.")






# ----------------- Agent Section -----------------
st.markdown("---")
st.subheader("🧠 Agent Assistant (Single Step)")
st.markdown("Enter a task and let the agent decide which MCP tool to use!")

user_input = st.text_input("What do you want the agent to do?", key="agent_input")

if st.button("🧠 Run Agent"):
    if user_input.strip():
        with st.spinner("🤖 Agent thinking..."):

            # Step 1: Use LLM to get plan
            llm = ChatOpenAI(model="gpt-4")
            system_prompt = """You are an AI assistant that converts natural language into MCP tool calls.
Your response must be a JSON with keys: server, tool, args.

Available servers and tools:
- echo: echo(message: str)
- math: add(a: int, b: int), subtract(a: int, b: int)
- github: get_github_user(username: str)
- gmail: send_email(to: str, subject: str, message: str)

Only return valid JSON.
"""

            response = llm([
                HumanMessage(content=system_prompt),
                HumanMessage(content=f"User instruction: {user_input}")
            ])

            try:
                import json
                plan = json.loads(response.content)
                st.code(plan, language="json")
                ACTIONS_LOG.append(f"Agent Plan: {plan}")

                # Step 2: Construct server URL
                server_url = f"http://localhost:8000/{plan['server']}/mcp"
                tool_name = plan["tool"]
                args = plan["args"]

                st.info(f"🔌 Calling `{tool_name}` on `{plan['server']}` with args `{args}`")

                # Step 3: Call tool
                result = asyncio.run(call_tool(server_url, tool_name, args))
                st.success(f"✅ Result: {result}")

            except Exception as e:
                st.error(f"Failed to parse or execute agent plan: {e}")
                st.exception(e)
    else:
        st.warning("Please enter a task.")


import re
import json

st.markdown("---")
st.subheader("🧠 Agent Assistant (Multi-Step)")
st.markdown("Enter a task and let the agent decide which MCP tools to use in sequence!")

user_input = st.text_input("🧠 What do you want the agent to do?", key="agent_input_v2")

if st.button("▶️ Run Agent Plan"):
    if not user_input.strip():
        st.warning("Please enter a task.")
    else:
        with st.spinner("🧠 Thinking..."):
            try:
                llm = ChatOpenAI(model="gpt-4")
                system_prompt = """You are a planning agent that breaks down user instructions into a list of tool invocations for an MCP-based system.
Return a valid JSON list. Each step must include:
- "server": one of [echo, math, github, gmail]
- "tool": tool to invoke
- "args": dictionary of arguments

If you need output from a previous step, refer to it like {{step_0}}.

Example:

[
  {
    "server": "math",
    "tool": "add",
    "args": {"a": 2, "b": 2}
  },
  {
    "server": "gmail",
    "tool": "send_email",
    "args": {
      "to": "someone@example.com",
      "subject": "Answer",
      "message": "The result is {{step_0}}"
    }
  }
]
"""

                # Step 1: Generate plan
                response = llm([
                    HumanMessage(content=system_prompt),
                    HumanMessage(content=f"User: {user_input}")
                ])

                plan = json.loads(response.content)
                st.code(plan, language="json")
                ACTIONS_LOG.append(f"Agent Plan: {plan}")

                results = []

                # Step 2: Run each step
                for idx, step in enumerate(plan):
                    st.markdown(f"### Step {idx + 1}: `{step['tool']}` on `{step['server']}`")
                    args = step["args"]

                    # Step 2a: Replace any `{{step_n}}` placeholders
                    def substitute(val):
                        if isinstance(val, str):
                            matches = re.findall(r"{{step_(\d+)}}", val)
                            for m in matches:
                                val = val.replace(f"{{{{step_{m}}}}}", str(results[int(m)]))
                        return val

                    args = {k: substitute(v) for k, v in args.items()}
                    st.write("📤 Calling with args:", args)

                    # Step 2b: Call the MCP tool
                    server_url = f"http://localhost:8000/{step['server']}/mcp"
                    result = asyncio.run(call_tool(server_url, step["tool"], args))
                    st.success(f"✅ Result: {result}")
                    results.append(result)

            except Exception as e:
                st.error(f"Agent execution failed: {e}")


### 🧠 A2A Agent Collaboration (Improved)
st.markdown("---")
st.subheader("🤖 A2A Agent Collaboration (Agent to Agent)")
st.markdown("""
This simulates how agents can **plan**, **delegate**, and **collaborate** using the MCP tools.

### 📍 Scenario:
- **Agent A** (Planner): Breaks down the task into sequential tool calls.
- **Agent B** (Executor): Executes the plan step-by-step and records outputs.

📝 Example Tasks:
- "Send directions from Delhi to New York to my email."
- "Add 5 and 10, then email the result."
""")

user_input = st.text_input("🎯 What task should Agent A plan for Agent B?", key="a2a_input")

if st.button("🧠 Run A2A Collaboration"):
    if not user_input.strip():
        st.warning("Please enter a task.")
    else:
        with st.spinner("🧠 Agent A planning..."):
            try:
                llm = ChatOpenAI(model="gpt-4")

                planner_prompt = """
You are Agent A. Your job is to break down user tasks into a multi-step plan using MCP tool calls.
Respond ONLY with a **valid JSON list**.

Each step must include:
- "server": one of ["echo", "math", "gmail", "github"]
- "tool": the tool to invoke on that server
- "args": dictionary of required arguments

Allowed Tools and Args:
- echo → echo(message)
- directions → start, end
- maps → start, end
- send_email → to, subject, message
- add/subtract → a, b
- get_github_user → username

If a step depends on a previous output, use {{step_N}} placeholders (e.g., {{step_0}}).
"""

                response = llm([
                    HumanMessage(content=planner_prompt),
                    HumanMessage(content=f"Task: {user_input}")
                ])

                st.markdown("### 🔍 Agent A (Planner) Output")
                st.code(response.content, language="json")

                try:
                    plan = json.loads(response.content)
                except json.JSONDecodeError:
                    st.error("❌ Invalid JSON output from LLM. Fix the planner prompt or response format.")
                    raise

                if not isinstance(plan, list):
                    st.error("❌ Expected a list of steps. Got invalid format.")
                    raise ValueError("Invalid plan format")

                st.markdown("### 📋 Agent A Plan")
                st.code(plan, language="json")

                results = []
                st.markdown("---")
                st.markdown("### 🤝 Agent B Execution")

                for idx, step in enumerate(plan):
                    st.markdown(f"#### 🔧 Step {idx+1}: `{step['tool']}` on `{step['server']}`")

                    def substitute(val):
                        if isinstance(val, str):
                            matches = re.findall(r"{{step_(\d+)}}", val)
                            for m in matches:
                                if int(m) < len(results):
                                    val = val.replace(f"{{{{step_{m}}}}}", str(results[int(m)]))
                        return val

                    args = {k: substitute(v) for k, v in step.get("args", {}).items()}
                    server_url = f"http://localhost:8000/{step['server']}/mcp"

                    result = asyncio.run(call_tool(server_url, step["tool"], args))

                    if getattr(result, "isError", False) or ("Unknown tool" in str(result)):
                        st.error(f"❌ Step {idx+1} failed: {result}")
                        break

                    st.success(f"✅ Result: {result}")
                    results.append(result)

                    with st.expander("📇 Agent Card", expanded=True):
                        st.markdown(f"""
                        **Agent**: B  
                        **Role**: Executor  
                        **Tool**: `{step['tool']}`  
                        **Args**: `{args}`  
                        **Output**: `{result}`  
                        **Status**: ✅ Completed
                        """)

            except Exception as e:
                st.error(f"A2A execution failed: {e}")
