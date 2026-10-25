# Building Deep Agents with LangChain: A Complete Hands-On Tutorial

### Building deep agents with langchain and langsmith

If you have been building AI agents, you have probably hit the wall. Simple ReAct agents work great for one-shot questions, but the moment you throw a complex, multi-step research task at them, they fall apart. They lose context. They forget what they were doing. They cannot plan ahead.

That is exactly the problem Deep Agents solve.

In this tutorial, we will walk through building deep agents using LangChain’s `deepagents` library. These are agents that can plan, manage context through file systems, spawn subagents, and handle complex tasks that would make a regular agent choke. Think of them as the architecture behind tools like Claude Code, Deep Research, and Manus.

## What Are Deep Agents?

Deep agents are a standalone library built on top of LangGraph that brings production-grade capabilities to LLM agents. Unlike simple agents that just loop between "think, act, observe," deep agents come with built-in infrastructure for handling real-world complexity.

Here is what makes them different from a basic ReAct agent:

- **Planning** -- Deep agents automatically break down complex tasks into subtasks using a built-in `write_todos` tool. Before diving into execution, they create a plan of attack.
- **File System for Context Management** -- LLMs have limited context windows. Deep agents solve this by offloading large intermediate results (like search results or document content) into a virtual file system using `write_file` and `read_file` tools. This means they can handle tasks that produce far more data than a context window could hold.
- **Subagent Spawning** -- For complex tasks, a deep agent can delegate subtasks to specialized subagents. Each subagent operates in its own context isolation, preventing the "context pollution" problem where unrelated information from one subtask confuses another.
- **Persistent Memory** -- Deep agents can persist memory across conversations and threads, making them suitable for long-running tasks and multi-session workflows.

## When Should You Use Deep Agents?

Use deep agents when your task involves any of these characteristics:

- **Multi-step complexity** -- The task requires planning and decomposition, not just a single LLM call.
- **Large context requirements** -- You need to process more information than fits in a single context window.
- **Work delegation** -- Different parts of the task benefit from specialized handling by isolated subagents.
- **Persistent workflows** -- You need memory that survives across conversations and threads.

For simple Q&A or single-tool tasks, a basic agent is fine. Deep agents shine when the task feels more like a project than a question.

## Tutorial: Building Your First Deep Agent

### Prerequisites

You will need API keys for OpenAI (or any LLM provider supported by LangChain), Groq (if you want to use fast inference with models like Qwen), and Tavily (for web search). Install the required packages:

```bash
pip install langchain deepagents tavily-python python-dotenv
```

### Step 1: Environment Setup

```python
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
```

We are loading three API keys: OpenAI for GPT models, Groq for fast inference with open-source models, and Tavily for web search capabilities.

### Step 2: Create a Web Search Tool

Every useful agent needs access to external information. We will create a web search tool using Tavily that supports different search topics:

```python
from tavily import TavilyClient
from typing import Literal

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def web_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "sports", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )
```

A few things to notice here. The `topic` parameter lets you scope searches to specific domains like news or finance, which improves result quality. The `include_raw_content` flag is useful when you need the full page content, not just snippets. And the type hints with `Literal` ensure the LLM knows exactly what values are valid when calling this tool.

### Step 3: Initialize the LLM

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("groq:qwen/qwen3-32b")
```

We are using Qwen 3 32B through Groq for fast inference. The beauty of `init_chat_model` is that you can swap models by just changing the string -- `"openai:gpt-4o"`, `"groq:llama-3.3-70b"`, `"openai:gpt-5"`, etc. The rest of your code stays exactly the same.

### Step 4: Basic Agent vs Deep Agent

Before diving into deep agents, let us see what a basic agent looks like for comparison:

```python
from langchain.agents import create_agent

simple_agent = create_agent(
    model=model,
    tools=[web_search],
)
```

This creates a standard ReAct agent. It works, but it has no planning capabilities, no file system, and no ability to delegate work. Now let us see the difference with a deep agent:

```python
from deepagents import create_deep_agent

deepagent = create_deep_agent(
    model=model,
    tools=[web_search],
    system_prompt="Act as a researcher",
)
```

That is it. Three parameters and you have an agent with planning, file management, context offloading, and subagent capabilities built in. The `create_deep_agent` function wraps all the LangGraph orchestration for you.

### Step 5: Running the Deep Agent

```python
result = deepagent.invoke({
    "messages": [
        {"role": "user", "content": "What is deepagent?"}
    ]
})
```

The invoke call takes a messages list in the standard chat format. Let us look at the output:

```python
# Get the final response
result["messages"][-1].content

# Check what files the agent created
result["files"]
```

The `result` object contains both the message history and any files the agent created during its work. This is a key difference from basic agents -- deep agents produce artifacts, not just text.

## What Happened Under the Hood?

When you ran that single invoke call, your deep agent automatically performed a sophisticated multi-step workflow:

1. **Planned its approach** -- Used the built-in `write_todos` tool to break down the research task into manageable steps.
2. **Conducted research** -- Called the `web_search` tool (which we provided) to gather information from the internet.
3. **Managed context** -- Used file system tools (`write_file`, `read_file`) to offload large search results. Instead of stuffing everything into the context window, the agent wrote results to files and read them back when needed.
4. **Spanned subagents (if needed)** -- For complex subtasks, the agent may have delegated work to specialized subagents that operated in their own isolated context.
5. **Synthesized a report** -- Compiled all findings into a coherent, well-structured response.

All of this happened automatically. You did not have to orchestrate any of it -- the deep agent framework handles the planning and execution loop.

## Customizing Deep Agents

The real power of deep agents comes from customization. Let us explore the three main knobs you can turn.

### Customization 1: Swap the Model

Want to use a different model? Just change the model string:

```python
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent

model = init_chat_model(model="gpt-5")
agent = create_deep_agent(model=model)

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "What is deepagent?"}
    ]
})
```

This is incredibly useful for benchmarking. Run the same task with GPT-4o-mini, GPT-5, Claude, Qwen, or Llama and compare the quality of planning and execution. Different models have different strengths -- some plan better, some execute tool calls more reliably, some synthesize information more coherently.

### Customization 2: Custom System Prompts

Deep agents come with a built-in system prompt inspired by Claude Code’s system prompt. It contains detailed instructions for using the planning tool, file system tools, and subagents.

But for production use cases, you should always provide a custom system prompt tailored to your specific task:

```python
from deepagents import create_deep_agent

research_instructions = """\
You are an expert researcher. Your job is to conduct \
thorough research, and then write a polished report. \
"""

agent = create_deep_agent(
    model=model,
    system_prompt=research_instructions,
)

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "What is deepagent?"}
    ]
})
```

The custom system prompt is appended to the built-in prompt, not replacing it. This means your agent still knows how to use planning, file systems, and subagents -- but now it also has domain-specific instructions.

Some examples of effective custom prompts:

- **Research agent:** "You are an expert researcher. Conduct thorough research, verify claims from multiple sources, and produce a polished report with citations."
- **Code review agent:** "You are a senior software engineer. Analyze the codebase, identify bugs and security vulnerabilities, and produce a detailed code review with severity ratings."
- **Financial analyst agent:** "You are a financial analyst. Gather market data, analyze trends, and produce an investment memo with risk assessments."

### Customization 3: Custom Tools

You can provide any tools you want. The web search tool we created earlier is just one example:

```python
import os
from typing import Literal
from tavily import TavilyClient
from deepagents import create_deep_agent

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )

agent = create_deep_agent(
    model=model,
    tools=[internet_search],
)

result = agent.invoke({
    "messages": [
        {"role": "user",
         "content": "What is deepagent in Agentic AI?"}
    ]
})
```

Your custom tools are added alongside the built-in tools (planning, file system, subagents). This means even with no custom tools at all, a deep agent can still plan, manage files, and spawn subagents -- your tools extend its capabilities into the real world.

Some ideas for custom tools you could add:

- **Database query tool** -- Let the agent query your internal databases.
- **API integration tools** -- Connect to Slack, Jira, GitHub, or any service your workflow needs.
- **Code execution tool** -- Let the agent write and run Python code.
- **Document loader tool** -- Let the agent read PDFs, Word docs, or spreadsheets.

## Deep Agents vs Basic Agents: When to Use What

Here is a practical decision framework:

**Use a basic agent when:**

- The task is straightforward (single question, single tool call)
- Context requirements are small (fits in one context window)
- Latency matters more than thoroughness
- You need a quick prototype

**Use a deep agent when:**

- The task requires multiple steps and planning
- You are processing large amounts of information
- Quality and thoroughness matter more than speed
- You need the agent to produce structured artifacts (reports, files)
- Different subtasks benefit from context isolation

## The Architecture Behind Deep Agents

Understanding what is happening under the hood helps you debug and optimize.

Deep agents are built on LangGraph, which provides the state machine that orchestrates the planning-execution loop. Here is the simplified flow:

```
User Input --> Planner (write_todos) --> Execute Step 1 --> File System (write_file) --> Execute Step 2 --> Subagent (if needed) --> File System (read_file) --> Synthesize --> Final Response
```

The key architectural decisions:

- **State management** -- LangGraph maintains the full state of the agent, including messages, files, and todo lists. This state can be persisted, enabling long-running tasks.
- **Tool routing** -- The agent decides which tool to use at each step. Built-in tools (planning, files, subagents) are always available alongside your custom tools.
- **Context windowing** -- Large results are automatically offloaded to the file system, keeping the active context window focused on the current step.

## Best Practices

After building several deep agent systems, here are the lessons learned:

1. **Write specific system prompts.** "You are a helpful assistant" is not enough. Tell the agent exactly what kind of output you expect, what quality standards to follow, and what steps to take.
2. **Design tools with good docstrings.** The LLM reads your function docstrings and parameter descriptions to decide when and how to use tools. Clear documentation leads to better tool usage.
3. **Use type hints on tool parameters.** `Literal` types, clear parameter names, and default values help the model make correct tool calls. A parameter named `query: str` is clearer than `q: str`.
4. **Start simple, add complexity.** Begin with one tool and a simple prompt. Get that working well before adding subagents, multiple tools, or complex prompts.
5. **Monitor with LangSmith.** Deep agents make many tool calls internally. Without observability, debugging is nearly impossible. Enable LangSmith tracing to see every planning step, tool call, and file operation.
6. **Choose the right model.** Deep agents make many sequential decisions. More capable models (GPT-4o, GPT-5, Claude Sonnet) generally plan and execute better than smaller models. Use fast models like Groq for prototyping, then switch to more capable models for production.

## Key Takeaways

1. **Deep agents solve the complexity problem.** When your task is too complex for a single LLM call or even a basic agent loop, deep agents provide the infrastructure for planning, context management, and work delegation.
2. **The API is surprisingly simple.** Despite the sophisticated internals, creating a deep agent is just three lines of code -- model, tools, and system prompt.
3. **File systems are the secret weapon.** By offloading intermediate results to files, deep agents can handle tasks that produce far more data than any context window could hold.
4. **Subagents enable context isolation.** Different subtasks do not pollute each other’s context, leading to higher quality results on complex tasks.
5. **LangGraph powers the orchestration.** Deep agents are built on LangGraph’s state machine, giving you all the benefits of structured agent orchestration without having to build it yourself.
6. **Model choice matters.** Deep agents make many sequential decisions. More capable models plan better, execute tools more reliably, and synthesize information more coherently.

## Continue Your Learning Journey

If you found this tutorial helpful and want to go deeper into building production-grade AI systems, here are some resources to level up:

- **Live Cohorts in Gen AI and Agentic AI** -- Join our instructor-led, hands-on bootcamps covering the full modern GenAI stack: LLMs, Fine-Tuning, RAG, Agents, Guardrails, Evaluation, LLMOps, and Cloud Deployment. These are 5-6 month programs designed for AI engineers and developers who want to build and ship real-world systems. Explore Live Classes
- **Industry-Ready AI Projects** -- Get hands-on with end-to-end, production-level projects across AI, ML, Data Science, and Generative AI. Each project is designed to mirror real-world use cases so you can build a portfolio that actually impresses hiring managers. Explore Projects
- **Free YouTube Tutorials** -- 2000+ videos covering everything from Python basics to advanced Agentic AI systems, completely free. Krish Naik YouTube Channel | Krish Naik Hindi Channel
- **Learn On The Go** -- Download the Krish Naik Academy app for iOS and Android to access courses, live classes, and projects from anywhere. App Store | Google Play
- **Stay Updated** -- Subscribe to this Substack newsletter for weekly deep-dives into AI, Data Science, and Generative AI concepts with practical code examples.

## References

- LangChain Documentation -- Official docs for building LLM applications with LangChain.
- LangGraph Documentation -- The state machine framework that powers deep agents.
- Tavily API Documentation -- Web search API optimized for LLM applications.
- Groq Documentation -- Fast inference for open-source models like Qwen and Llama.
- Claude Code -- One of the applications that inspired the deep agents architecture.
