# 🛠️ LangGraph Course Troubleshooting Guide

This comprehensive guide helps you resolve common issues encountered during the LangGraph multi-agent systems course. Issues are organized by category with step-by-step solutions.

## 🚨 Quick Fixes for Common Issues

### Issue: "Module not found" errors
**Quick Fix:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# Reinstall requirements
pip install -r requirements.txt
```

### Issue: OpenAI API errors
**Quick Fix:**
```bash
# Check your .env file
cat .env | grep OPENAI_API_KEY

# Test API connection
python -c "
import openai, os
from dotenv import load_dotenv
load_dotenv()
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
print('API connection successful!')
"
```

### Issue: Jupyter kernel problems
**Quick Fix:**
```bash
# Reinstall kernel
python -m ipykernel install --user --name=venv --display-name "LangGraph Course" --force

# Restart Jupyter
jupyter notebook stop
jupyter notebook
```

---

## 🐍 Python Environment Issues

### Problem: Python version incompatibility
**Symptoms:**
- Import errors with modern Python features
- Syntax errors in course code
- Package installation failures

**Solution:**
```bash
# Check Python version
python --version

# Should be 3.9 or higher
# If not, install newer Python:

# macOS (using Homebrew)
brew install python@3.11

# Ubuntu/Debian
sudo apt update
sudo apt install python3.11

# Windows
# Download from python.org

# Create new virtual environment with correct Python
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Problem: Virtual environment not activating
**Symptoms:**
- Commands run in global Python environment
- Package installation affects system Python
- Import errors for course packages

**Solution:**

**macOS/Linux:**
```bash
# Ensure activation script exists
ls venv/bin/activate

# If missing, recreate environment
rm -rf venv
python -m venv venv
source venv/bin/activate

# Verify activation (prompt should show (venv))
which python
```

**Windows:**
```cmd
# Check for activation script
dir venv\Scripts\activate.bat

# If missing, recreate
rmdir /s venv
python -m venv venv
venv\Scripts\activate

# Verify activation
where python
```

### Problem: Package installation failures
**Symptoms:**
- Permission denied errors
- Compilation failures
- Dependency conflicts

**Solutions:**

**Permission Issues:**
```bash
# Use user install (not recommended, but works)
pip install --user -r requirements.txt

# Or fix permissions (macOS/Linux)
sudo chown -R $(whoami) venv/
```

**Compilation Issues (macOS):**
```bash
# Install Xcode command line tools
xcode-select --install

# Install with verbose output to debug
pip install -v langchain
```

**Dependency Conflicts:**
```bash
# Force reinstall all packages
pip install --upgrade --force-reinstall -r requirements.txt

# Or create fresh environment
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🔑 OpenAI API Issues

### Problem: Invalid API key
**Symptoms:**
- `AuthenticationError: Incorrect API key`
- 401 Unauthorized responses
- API calls fail immediately

**Solution:**
```bash
# 1. Check .env file format
cat .env
# Should show: OPENAI_API_KEY=sk-proj-...

# 2. Verify key format (should start with sk-proj- or sk-)
echo $OPENAI_API_KEY

# 3. Test key directly
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"

# 4. Regenerate key if needed
# Go to https://platform.openai.com/api-keys
# Delete old key and create new one
```

### Problem: Rate limiting
**Symptoms:**
- `RateLimitError: Rate limit exceeded`
- 429 Too Many Requests
- Requests timing out

**Solutions:**

**Immediate Fix:**
```python
import time
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def safe_api_call(messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            return response
        except RateLimitError:
            wait_time = 2 ** attempt  # Exponential backoff
            print(f"Rate limited, waiting {wait_time} seconds...")
            time.sleep(wait_time)
    
    raise Exception("Max retries exceeded")
```

**Long-term Solution:**
```python
# Add to your agent functions
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def robust_llm_call(messages):
    response = await client.chat.completions.acreate(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=1000
    )
    return response
```

### Problem: Quota exceeded
**Symptoms:**
- `InsufficientQuotaError: You exceeded your current quota`
- Billing-related errors
- API calls rejected

**Solution:**
1. **Check Usage:**
   - Visit [OpenAI Usage Dashboard](https://platform.openai.com/usage)
   - Review current month's spending

2. **Add Payment Method:**
   - Go to [Billing](https://platform.openai.com/account/billing)
   - Add credit card or increase limits

3. **Set Usage Limits:**
   ```bash
   # In your .env file
   MAX_DAILY_COST=10.00
   ENABLE_COST_TRACKING=true
   ```

4. **Cost Optimization:**
   ```python
   # Use cheaper models for simple tasks
   def choose_model(task_complexity):
       if task_complexity == "simple":
           return "gpt-3.5-turbo"  # $0.001/1K tokens
       else:
           return "gpt-4"  # $0.03/1K tokens
   ```

### Problem: Connection timeouts
**Symptoms:**
- Requests hang indefinitely
- Connection reset errors
- Network-related failures

**Solution:**
```python
import httpx
from openai import OpenAI

# Configure timeout and retry
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=httpx.Timeout(60.0, connect=10.0),
    max_retries=3
)

# Alternative: manual timeout handling
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("API call timed out")

def safe_llm_call_with_timeout(messages, timeout=30):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        signal.alarm(0)  # Cancel timeout
        return response
    except TimeoutError:
        print("API call timed out, retrying...")
        return None
```

---

## 📊 LangGraph Specific Issues

### Problem: State serialization errors
**Symptoms:**
- `TypeError: Object of type X is not JSON serializable`
- Checkpoint save/load failures
- State persistence issues

**Solution:**
```python
from typing import TypedDict, Any
import json
from datetime import datetime

# Define serializable state
class SerializableState(TypedDict):
    messages: list[dict]  # Use dict instead of BaseMessage for serialization
    metadata: dict[str, Any]
    timestamp: str  # Use string instead of datetime

# Custom serialization for complex objects
def serialize_state(state):
    """Convert state to JSON-serializable format."""
    serialized = {}
    for key, value in state.items():
        if isinstance(value, datetime):
            serialized[key] = value.isoformat()
        elif hasattr(value, 'dict'):  # Pydantic objects
            serialized[key] = value.dict()
        elif isinstance(value, list):
            serialized[key] = [
                item.dict() if hasattr(item, 'dict') else item 
                for item in value
            ]
        else:
            serialized[key] = value
    return serialized

def deserialize_state(serialized):
    """Convert from JSON back to state objects."""
    # Implementation depends on your specific objects
    return serialized
```

### Problem: Graph compilation errors
**Symptoms:**
- `ValueError: Node X not found in graph`
- Edge connection failures
- Workflow validation errors

**Solution:**
```python
from langgraph.graph import StateGraph, START, END

# Debug graph construction
def debug_workflow_creation():
    workflow = StateGraph(AgentState)
    
    # Add nodes with validation
    nodes = ["agent1", "agent2", "agent3"]
    for node in nodes:
        print(f"Adding node: {node}")
        workflow.add_node(node, globals()[f"{node}_function"])
    
    # Add edges with validation
    edges = [
        (START, "agent1"),
        ("agent1", "agent2"),
        ("agent2", END)
    ]
    
    for source, target in edges:
        print(f"Adding edge: {source} -> {target}")
        workflow.add_edge(source, target)
    
    # Compile with error handling
    try:
        app = workflow.compile()
        print("✅ Graph compiled successfully")
        return app
    except Exception as e:
        print(f"❌ Graph compilation failed: {e}")
        return None
```

### Problem: Infinite loops in agents
**Symptoms:**
- Agents never reach END state
- High API costs from repeated calls
- Memory usage continuously increasing

**Solution:**
```python
# Add loop detection and limits
class LoopDetector:
    def __init__(self, max_iterations=10):
        self.max_iterations = max_iterations
        self.iteration_count = 0
        self.state_history = []
    
    def check_loop(self, state):
        self.iteration_count += 1
        
        # Check iteration limit
        if self.iteration_count > self.max_iterations:
            raise Exception(f"Maximum iterations ({self.max_iterations}) exceeded")
        
        # Check for state repetition
        state_hash = hash(str(sorted(state.items())))
        if state_hash in self.state_history:
            raise Exception("Infinite loop detected - state repeated")
        
        self.state_history.append(state_hash)
        return True

# Use in conditional routing
def safe_conditional_router(state: AgentState):
    detector = state.get("loop_detector", LoopDetector())
    detector.check_loop(state)
    
    # Your routing logic here
    if some_condition:
        return "continue"
    else:
        return "end"
```

---

## 💾 Database and Persistence Issues

### Problem: SQLite database locked
**Symptoms:**
- `sqlite3.OperationalError: database is locked`
- Checkpoint operations fail
- Database file corruption

**Solution:**
```python
import sqlite3
import os
import time

def fix_locked_database(db_path):
    """Attempt to fix locked SQLite database."""
    
    # 1. Close all connections
    # (Make sure no other processes are using the DB)
    
    # 2. Check for write-ahead log files
    wal_file = db_path + "-wal"
    shm_file = db_path + "-shm"
    
    if os.path.exists(wal_file) or os.path.exists(shm_file):
        print("WAL files found, attempting checkpoint...")
        
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()
    
    # 3. Test connection
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        conn.execute("SELECT 1")
        conn.close()
        print("✅ Database accessible")
    except Exception as e:
        print(f"❌ Database still locked: {e}")
        
        # Last resort: backup and recreate
        backup_path = f"{db_path}.backup"
        os.rename(db_path, backup_path)
        print(f"Database backed up to {backup_path}")

# Use with timeout and retry
from langgraph.checkpoint.sqlite import SqliteSaver

def create_resilient_sqlite_saver(db_path):
    """Create SQLite saver with better error handling."""
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Configure with timeout
    conn_string = f"sqlite:///{db_path}?timeout=20"
    
    try:
        saver = SqliteSaver.from_conn_string(conn_string)
        return saver
    except Exception as e:
        print(f"Failed to create SQLite saver: {e}")
        fix_locked_database(db_path)
        return SqliteSaver.from_conn_string(conn_string)
```

### Problem: PostgreSQL connection issues
**Symptoms:**
- `psycopg2.OperationalError: connection failed`
- Connection timeout errors
- Authentication failures

**Solution:**
```bash
# 1. Check PostgreSQL service status
# macOS
brew services list | grep postgresql

# Ubuntu/Debian
sudo systemctl status postgresql

# 2. Test connection manually
psql postgresql://username:password@localhost:5432/database_name

# 3. Check connection parameters
```

```python
import psycopg2
from urllib.parse import urlparse
import os

def test_postgres_connection():
    """Test and debug PostgreSQL connection."""
    database_url = os.getenv("DATABASE_URL")
    
    if not database_url:
        print("❌ DATABASE_URL not set in environment")
        return False
    
    # Parse URL
    parsed = urlparse(database_url)
    
    print(f"Host: {parsed.hostname}")
    print(f"Port: {parsed.port}")
    print(f"Database: {parsed.path[1:]}")  # Remove leading /
    print(f"User: {parsed.username}")
    
    # Test connection
    try:
        conn = psycopg2.connect(database_url)
        print("✅ PostgreSQL connection successful")
        
        # Test basic query
        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            version = cur.fetchone()[0]
            print(f"PostgreSQL version: {version}")
        
        conn.close()
        return True
        
    except psycopg2.Error as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        
        # Common solutions
        print("\nTroubleshooting steps:")
        print("1. Check if PostgreSQL is running")
        print("2. Verify database exists: CREATE DATABASE langgraph_course;")
        print("3. Check user permissions: GRANT ALL ON DATABASE langgraph_course TO user;")
        print("4. Verify connection parameters in DATABASE_URL")
        
        return False

# Run test
test_postgres_connection()
```

---

## 📓 Jupyter Notebook Issues

### Problem: Kernel keeps dying
**Symptoms:**
- "Kernel died, restarting" messages
- Code cells fail to execute
- Memory errors in logs

**Solutions:**

**Memory Issues:**
```python
# Check memory usage
import psutil
import os

def check_memory():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
    
    # System memory
    system_memory = psutil.virtual_memory()
    print(f"System memory: {system_memory.percent}% used")

# Add to notebooks for monitoring
check_memory()
```

**Kernel Configuration:**
```bash
# Increase memory limits
export JUPYTER_MEMORY_LIMIT=4096  # 4GB

# Or create jupyter config
jupyter notebook --generate-config

# Edit ~/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.max_buffer_size = 1073741824  # 1GB" >> ~/.jupyter/jupyter_notebook_config.py
```

**Environment Issues:**
```bash
# Recreate kernel
jupyter kernelspec uninstall venv
python -m ipykernel install --user --name=venv --display-name "LangGraph Course"

# Clear Jupyter cache
jupyter --paths
# Remove cache directories shown
```

### Problem: Import errors in notebooks
**Symptoms:**
- `ModuleNotFoundError` in notebook but not terminal
- Wrong Python interpreter
- Package not found errors

**Solution:**
```python
# Check which Python is being used
import sys
print(sys.executable)
print(sys.path)

# Should point to your virtual environment
# If not, kernel is not properly configured

# Add current directory to path (temporary fix)
import sys
import os
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

# Install packages in notebook (emergency fix)
import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "package_name"])
```

### Problem: Notebook won't save
**Symptoms:**
- "Autosave failed" messages
- Changes not persisting
- File permission errors

**Solution:**
```bash
# Check file permissions
ls -la *.ipynb

# Fix permissions
chmod 644 *.ipynb

# Check disk space
df -h

# Clear Jupyter checkpoints
find . -name .ipynb_checkpoints -type d -exec rm -rf {} +

# Backup and recreate if needed
cp notebook.ipynb notebook_backup.ipynb
```

---

## 🔧 Development Environment Issues

### Problem: VS Code not recognizing virtual environment
**Symptoms:**
- Wrong Python interpreter shown
- Import errors in VS Code
- IntelliSense not working

**Solution:**
1. **Open Command Palette** (Ctrl+Shift+P / Cmd+Shift+P)
2. **Select "Python: Select Interpreter"**
3. **Choose interpreter from venv folder**
   - Should be: `./venv/bin/python` (macOS/Linux)
   - Or: `.\venv\Scripts\python.exe` (Windows)

**Alternative:**
```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.terminal.activateEnvironment": true
}
```

### Problem: Git showing too many changes
**Symptoms:**
- Notebook outputs cluttering git
- Large file size differences
- Merge conflicts in notebooks

**Solution:**
```bash
# Add to .gitignore
echo "**/.ipynb_checkpoints/" >> .gitignore
echo "**/*-checkpoint.ipynb" >> .gitignore

# Clean notebook outputs before committing
pip install nbstripout

# Strip outputs from notebooks
nbstripout *.ipynb

# Auto-strip on commit (optional)
nbstripout --install
```

---

## 🎯 Performance and Cost Issues

### Problem: High API costs
**Symptoms:**
- Unexpected billing charges
- Rapid quota consumption
- High token usage

**Solutions:**

**Token Optimization:**
```python
import tiktoken

def optimize_prompt(prompt, max_tokens=1000):
    """Truncate prompt to stay within token limits."""
    encoding = tiktoken.encoding_for_model("gpt-4")
    tokens = encoding.encode(prompt)
    
    if len(tokens) > max_tokens:
        # Truncate from middle, keep beginning and end
        keep_start = max_tokens // 3
        keep_end = max_tokens // 3
        
        truncated_tokens = tokens[:keep_start] + tokens[-keep_end:]
        return encoding.decode(truncated_tokens)
    
    return prompt

# Cost tracking
class CostTracker:
    def __init__(self):
        self.total_cost = 0
        self.call_count = 0
    
    def add_cost(self, input_tokens, output_tokens, model="gpt-4"):
        costs = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002}
        }
        
        cost = (input_tokens/1000 * costs[model]["input"] + 
                output_tokens/1000 * costs[model]["output"])
        
        self.total_cost += cost
        self.call_count += 1
        
        if self.total_cost > 5.0:  # $5 limit
            raise Exception(f"Daily cost limit exceeded: ${self.total_cost:.2f}")
```

**Model Selection:**
```python
def smart_model_selection(task_description):
    """Choose model based on task complexity."""
    
    # Simple tasks -> cheaper model
    simple_keywords = ["hello", "simple", "basic", "quick"]
    if any(keyword in task_description.lower() for keyword in simple_keywords):
        return "gpt-3.5-turbo"
    
    # Complex tasks -> better model
    complex_keywords = ["analyze", "complex", "reasoning", "detailed"]
    if any(keyword in task_description.lower() for keyword in complex_keywords):
        return "gpt-4"
    
    # Default to balanced option
    return "gpt-3.5-turbo"
```

---

## 🆘 Emergency Recovery Procedures

### Complete Environment Reset
```bash
# 1. Backup important files
cp -r . ../course_backup

# 2. Remove virtual environment
rm -rf venv

# 3. Clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} +
find . -name "*.pyc" -delete

# 4. Recreate environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install --upgrade pip
pip install -r requirements.txt

# 5. Reconfigure Jupyter
python -m ipykernel install --user --name=venv --display-name "LangGraph Course"
```

### Database Recovery
```bash
# SQLite
mv checkpoints.db checkpoints.db.backup
# Database will be recreated on next run

# PostgreSQL
dropdb langgraph_course
createdb langgraph_course
# Grant permissions again
```

### API Key Recovery
```bash
# 1. Regenerate OpenAI API key
# Go to https://platform.openai.com/api-keys

# 2. Update .env file
echo "OPENAI_API_KEY=new-key-here" > .env

# 3. Test connection
python -c "
import openai, os
from dotenv import load_dotenv
load_dotenv()
client = openai.OpenAI()
print('New API key working!')
"
```

---

## 📞 Getting Additional Help

### Course-Specific Help
1. **Check course notebooks** for inline troubleshooting tips
2. **Review error logs** in Jupyter notebook outputs
3. **Search this document** using Ctrl+F for specific error messages

### Community Resources
- **[LangChain Discord](https://discord.gg/langchain)** - Active community support
- **[OpenAI Community](https://community.openai.com/)** - Official OpenAI forums
- **[Stack Overflow](https://stackoverflow.com/questions/tagged/langchain)** - Technical Q&A

### Professional Support
- **[OpenAI Support](https://help.openai.com/)** - API-related issues
- **[LangChain Support](https://langchain.com/support)** - Framework-specific problems

### Emergency Contacts
If you encounter critical issues that prevent course completion:
1. **Save error logs** and system information
2. **Document steps to reproduce** the problem
3. **Check official documentation** for known issues
4. **Post in community forums** with detailed information

---

**Remember**: Most issues can be resolved by carefully following the setup guide and using the debugging techniques in this document. When in doubt, try the "Quick Fixes" section first!