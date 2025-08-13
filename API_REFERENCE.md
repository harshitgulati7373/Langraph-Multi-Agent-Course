# 📚 LangGraph & OpenAI API Quick Reference

This reference guide provides quick access to common patterns, code snippets, and best practices for LangGraph multi-agent development with OpenAI integration.

## 🚀 Quick Start Patterns

### Basic Agent Setup

```python
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI model
llm = ChatOpenAI(
    model="gpt-4-turbo-preview",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7
)

# Define state schema
class AgentState(TypedDict):
    messages: Annotated[list, "Messages in conversation"]
    current_task: str
    result: str

# Create agent function
def agent_node(state: AgentState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": messages + [response]}

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

# Compile graph
app = workflow.compile()
```

## 🔧 OpenAI Integration Patterns

### Model Configuration

```python
# Different model configurations
models = {
    "creative": ChatOpenAI(model="gpt-4", temperature=0.9),
    "analytical": ChatOpenAI(model="gpt-4", temperature=0.1),
    "fast": ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7),
    "precise": ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.3)
}

# Cost-optimized model selection
def get_model(task_complexity: str):
    if task_complexity == "simple":
        return ChatOpenAI(model="gpt-3.5-turbo")
    elif task_complexity == "complex":
        return ChatOpenAI(model="gpt-4-turbo-preview")
    else:
        return ChatOpenAI(model="gpt-4")
```

### Function Calling with OpenAI

```python
from pydantic import BaseModel, Field
from langchain_core.tools import tool

# Define tool with Pydantic schema
class WeatherQuery(BaseModel):
    location: str = Field(description="City name for weather query")
    units: str = Field(default="celsius", description="Temperature units")

@tool("get_weather", args_schema=WeatherQuery)
def get_weather(location: str, units: str = "celsius") -> str:
    """Get current weather for a location."""
    # Implementation here
    return f"Weather in {location}: 22°{units[0].upper()}, sunny"

# Bind tools to model
llm_with_tools = llm.bind_tools([get_weather])

# Agent with tool calling
def tool_agent(state: AgentState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    
    # Check if tool calls are needed
    if response.tool_calls:
        # Process tool calls
        for tool_call in response.tool_calls:
            tool_result = get_weather.invoke(tool_call["args"])
            # Add tool result to messages
    
    return {"messages": messages + [response]}
```

### Structured Output with Pydantic

```python
from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser

# Define output schema
class AnalysisResult(BaseModel):
    summary: str = Field(description="Brief summary of analysis")
    confidence: float = Field(description="Confidence score 0-1")
    recommendations: list[str] = Field(description="List of recommendations")
    category: str = Field(description="Classification category")

# Create parser
parser = PydanticOutputParser(pydantic_object=AnalysisResult)

# Format prompt with instructions
def get_analysis_prompt(text: str) -> str:
    return f"""
    Analyze the following text and provide structured output:
    
    {text}
    
    {parser.get_format_instructions()}
    """

# Use with OpenAI
def structured_analysis(text: str) -> AnalysisResult:
    prompt = get_analysis_prompt(text)
    response = llm.invoke(prompt)
    return parser.parse(response.content)
```

## 🗄️ State Management Patterns

### Basic State Definition

```python
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage

# Simple state
class BasicState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "Conversation history"]
    current_step: str
    result: dict

# Complex state with metadata
class ComplexState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "Messages"]
    context: dict
    metadata: dict
    agent_states: dict
    workflow_status: str
    error_log: list
```

### State Persistence Options

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.postgres import PostgresSaver

# In-memory (development)
memory_saver = MemorySaver()

# SQLite (local persistence)
sqlite_saver = SqliteSaver.from_conn_string("sqlite:///checkpoints.db")

# PostgreSQL (production)
postgres_saver = PostgresSaver.from_conn_string(
    os.getenv("DATABASE_URL")
)

# Compile with checkpointer
app = workflow.compile(checkpointer=sqlite_saver)
```

## 🤖 Multi-Agent Patterns

### Supervisor Pattern

```python
def supervisor(state: AgentState):
    """Supervisor agent that routes to specialists."""
    messages = state["messages"]
    
    # Determine which agent to call next
    routing_prompt = """
    Based on the conversation, which specialist should handle this:
    - researcher: for information gathering
    - analyst: for data analysis  
    - writer: for content creation
    - reviewer: for quality check
    
    Respond with just the agent name.
    """
    
    response = llm.invoke([{"role": "user", "content": routing_prompt}])
    next_agent = response.content.strip().lower()
    
    return {"next_agent": next_agent}

# Build supervisor workflow
def create_supervisor_workflow():
    workflow = StateGraph(AgentState)
    
    # Add all agents
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("researcher", research_agent)
    workflow.add_node("analyst", analysis_agent)
    workflow.add_node("writer", writing_agent)
    
    # Add conditional routing
    workflow.add_conditional_edges(
        "supervisor",
        lambda x: x["next_agent"],
        {
            "researcher": "researcher",
            "analyst": "analyst", 
            "writer": "writer",
            "end": END
        }
    )
    
    return workflow.compile()
```

### Handoff Pattern

```python
from langgraph.prebuilt import Command

def agent_with_handoff(state: AgentState):
    """Agent that can hand off to another agent."""
    messages = state["messages"]
    
    # Process the request
    response = llm.invoke(messages)
    
    # Decide if handoff is needed
    if "need_specialist" in response.content.lower():
        return Command(
            update={"messages": messages + [response]},
            goto="specialist_agent"
        )
    
    return {"messages": messages + [response]}

# Graph with handoffs
workflow = StateGraph(AgentState)
workflow.add_node("general_agent", agent_with_handoff)
workflow.add_node("specialist_agent", specialist_agent)
workflow.add_edge(START, "general_agent")
```

## 💾 Memory Patterns

### Short-term Memory

```python
class ShortTermMemory:
    def __init__(self, max_messages: int = 10):
        self.max_messages = max_messages
        self.messages = []
    
    def add_message(self, message):
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            self.messages.pop(0)  # Remove oldest
    
    def get_context(self) -> str:
        return "\n".join([f"{m['role']}: {m['content']}" 
                         for m in self.messages[-5:]])

# Use in agent
def memory_agent(state: AgentState):
    memory = state.get("short_term_memory", ShortTermMemory())
    
    # Add current message to memory
    current_message = state["messages"][-1]
    memory.add_message(current_message)
    
    # Use memory context in prompt
    context = memory.get_context()
    enhanced_prompt = f"Context: {context}\n\nCurrent: {current_message['content']}"
    
    response = llm.invoke([{"role": "user", "content": enhanced_prompt}])
    
    return {
        "messages": state["messages"] + [response],
        "short_term_memory": memory
    }
```

### Long-term Memory with Embeddings

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Initialize embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Create vector store
vectorstore = Chroma(
    persist_directory="./memory_db",
    embedding_function=embeddings
)

class LongTermMemory:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
    
    def store_memory(self, content: str, metadata: dict = None):
        """Store content in long-term memory."""
        self.vectorstore.add_texts([content], metadatas=[metadata or {}])
    
    def retrieve_memories(self, query: str, k: int = 3) -> list:
        """Retrieve relevant memories."""
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

# Use in agent
def memory_enhanced_agent(state: AgentState):
    memory = LongTermMemory(vectorstore)
    current_message = state["messages"][-1]["content"]
    
    # Retrieve relevant memories
    relevant_memories = memory.retrieve_memories(current_message)
    
    # Enhanced prompt with memories
    prompt = f"""
    Relevant memories: {relevant_memories}
    
    Current message: {current_message}
    
    Respond considering the relevant memories.
    """
    
    response = llm.invoke([{"role": "user", "content": prompt}])
    
    # Store this interaction
    memory.store_memory(
        f"User: {current_message}\nAssistant: {response.content}",
        {"timestamp": "2024-01-01", "type": "conversation"}
    )
    
    return {"messages": state["messages"] + [response]}
```

## ⚡ Performance Optimization

### Async Operations

```python
import asyncio
from langchain_openai import ChatOpenAI

# Async OpenAI client
async_llm = ChatOpenAI(
    model="gpt-4-turbo-preview",
    api_key=os.getenv("OPENAI_API_KEY")
)

async def async_agent(state: AgentState):
    """Async agent for better performance."""
    messages = state["messages"]
    response = await async_llm.ainvoke(messages)
    return {"messages": messages + [response]}

# Parallel processing
async def parallel_agents(state: AgentState):
    """Run multiple agents in parallel."""
    tasks = [
        async_agent_1(state),
        async_agent_2(state),
        async_agent_3(state)
    ]
    
    results = await asyncio.gather(*tasks)
    
    # Combine results
    combined_messages = state["messages"]
    for result in results:
        combined_messages.extend(result["messages"])
    
    return {"messages": combined_messages}
```

### Caching Strategies

```python
from functools import lru_cache
import hashlib

# Simple LRU cache
@lru_cache(maxsize=128)
def cached_llm_call(prompt_hash: str, prompt: str):
    """Cache LLM responses for identical prompts."""
    response = llm.invoke([{"role": "user", "content": prompt}])
    return response.content

def smart_agent_with_cache(state: AgentState):
    """Agent with response caching."""
    prompt = state["current_task"]
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    
    # Try cache first
    try:
        cached_response = cached_llm_call(prompt_hash, prompt)
        response_message = {"role": "assistant", "content": cached_response}
    except:
        # Fallback to API call
        response = llm.invoke([{"role": "user", "content": prompt}])
        response_message = response
    
    return {"messages": state["messages"] + [response_message]}
```

## 🔒 Security Patterns

### Input Validation

```python
from pydantic import BaseModel, validator

class SecureInput(BaseModel):
    user_input: str
    user_id: str
    
    @validator("user_input")
    def validate_input(cls, v):
        # Check for potential injection attempts
        dangerous_patterns = ["<script>", "javascript:", "exec(", "eval("]
        if any(pattern in v.lower() for pattern in dangerous_patterns):
            raise ValueError("Potentially dangerous input detected")
        return v
    
    @validator("user_id")  
    def validate_user_id(cls, v):
        # Ensure user_id is properly formatted
        if not v or not v.isalnum():
            raise ValueError("Invalid user ID format")
        return v

def secure_agent(state: AgentState):
    """Agent with input validation."""
    try:
        # Validate input
        secure_input = SecureInput(
            user_input=state["messages"][-1]["content"],
            user_id=state.get("user_id", "")
        )
        
        # Process validated input
        response = llm.invoke([{
            "role": "user", 
            "content": secure_input.user_input
        }])
        
        return {"messages": state["messages"] + [response]}
        
    except ValueError as e:
        error_response = {"role": "assistant", "content": f"Input validation error: {e}"}
        return {"messages": state["messages"] + [error_response]}
```

### Rate Limiting

```python
import time
from collections import defaultdict

class RateLimiter:
    def __init__(self, max_requests: int = 60, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = defaultdict(list)
    
    def is_allowed(self, user_id: str) -> bool:
        now = time.time()
        user_requests = self.requests[user_id]
        
        # Remove old requests
        user_requests[:] = [req_time for req_time in user_requests 
                           if now - req_time < self.time_window]
        
        # Check limit
        if len(user_requests) >= self.max_requests:
            return False
        
        # Add current request
        user_requests.append(now)
        return True

rate_limiter = RateLimiter()

def rate_limited_agent(state: AgentState):
    """Agent with rate limiting."""
    user_id = state.get("user_id", "anonymous")
    
    if not rate_limiter.is_allowed(user_id):
        error_response = {
            "role": "assistant", 
            "content": "Rate limit exceeded. Please wait before making another request."
        }
        return {"messages": state["messages"] + [error_response]}
    
    # Normal processing
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}
```

## 📊 Monitoring & Debugging

### LangSmith Integration

```python
from langsmith import traceable

# Trace agent execution
@traceable
def traced_agent(state: AgentState):
    """Agent with LangSmith tracing."""
    messages = state["messages"]
    
    # This will automatically be traced
    response = llm.invoke(messages)
    
    return {"messages": messages + [response]}

# Custom traces with metadata
@traceable(name="custom_agent_step")
def custom_traced_step(state: AgentState, step_name: str):
    """Custom traced step with metadata."""
    # Add custom metadata
    return {
        "step_name": step_name,
        "input_length": len(str(state)),
        "timestamp": time.time()
    }
```

### Error Handling

```python
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def robust_agent(state: AgentState):
    """Agent with comprehensive error handling."""
    try:
        messages = state["messages"]
        
        # Validate state
        if not messages:
            raise ValueError("No messages in state")
        
        # Try primary model
        try:
            response = llm.invoke(messages)
        except Exception as e:
            logger.warning(f"Primary model failed: {e}, trying fallback")
            # Fallback to cheaper model
            fallback_llm = ChatOpenAI(model="gpt-3.5-turbo")
            response = fallback_llm.invoke(messages)
        
        return {"messages": messages + [response]}
        
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        error_response = {
            "role": "assistant",
            "content": "I apologize, but I encountered an error. Please try again."
        }
        return {
            "messages": state.get("messages", []) + [error_response],
            "error": str(e)
        }
```

## 💰 Cost Optimization

### Token Counting

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens for cost estimation."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def estimate_cost(prompt: str, response: str, model: str = "gpt-4") -> float:
    """Estimate API call cost."""
    # Token costs per 1K tokens (as of 2024)
    costs = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.001, "output": 0.002}
    }
    
    input_tokens = count_tokens(prompt, model)
    output_tokens = count_tokens(response, model)
    
    input_cost = (input_tokens / 1000) * costs[model]["input"]
    output_cost = (output_tokens / 1000) * costs[model]["output"]
    
    return input_cost + output_cost

# Cost-aware agent
def cost_optimized_agent(state: AgentState):
    """Agent that optimizes for cost."""
    messages = state["messages"]
    prompt = messages[-1]["content"]
    
    # Choose model based on complexity
    token_count = count_tokens(prompt)
    
    if token_count < 500:  # Simple queries
        model = ChatOpenAI(model="gpt-3.5-turbo")
    else:  # Complex queries
        model = ChatOpenAI(model="gpt-4")
    
    response = model.invoke(messages)
    
    # Track cost
    cost = estimate_cost(prompt, response.content)
    logger.info(f"API call cost: ${cost:.4f}")
    
    return {
        "messages": messages + [response],
        "cost": cost
    }
```

## 🔗 Common Graph Patterns

### Conditional Routing

```python
def router(state: AgentState) -> str:
    """Route based on state conditions."""
    last_message = state["messages"][-1]["content"].lower()
    
    if "help" in last_message:
        return "help_agent"
    elif "analyze" in last_message:
        return "analysis_agent"
    elif "write" in last_message:
        return "writing_agent"
    else:
        return "general_agent"

# Build graph with conditional edges
workflow = StateGraph(AgentState)
workflow.add_node("router", router_agent)
workflow.add_node("help_agent", help_agent)
workflow.add_node("analysis_agent", analysis_agent)
workflow.add_node("writing_agent", writing_agent)
workflow.add_node("general_agent", general_agent)

workflow.add_conditional_edges(
    "router",
    router,
    {
        "help_agent": "help_agent",
        "analysis_agent": "analysis_agent", 
        "writing_agent": "writing_agent",
        "general_agent": "general_agent"
    }
)
```

### Loops and Cycles

```python
def should_continue(state: AgentState) -> str:
    """Determine if processing should continue."""
    if state.get("iteration_count", 0) >= 5:
        return "end"
    
    last_response = state["messages"][-1]["content"]
    if "task complete" in last_response.lower():
        return "end"
    
    return "continue"

# Graph with loops
workflow = StateGraph(AgentState)
workflow.add_node("processor", processing_agent)
workflow.add_node("validator", validation_agent)

workflow.add_edge(START, "processor")
workflow.add_edge("processor", "validator")

workflow.add_conditional_edges(
    "validator",
    should_continue,
    {
        "continue": "processor",  # Loop back
        "end": END
    }
)
```

This reference covers the most common patterns you'll use throughout the LangGraph course. Bookmark this page for quick access while working through the daily exercises!