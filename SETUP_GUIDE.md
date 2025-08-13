# 🔧 LangGraph Multi-Agent Systems Course - Complete Setup Guide

This comprehensive guide will walk you through setting up your development environment for the 7-day LangGraph course. Follow these steps carefully to ensure a smooth learning experience.

## 📋 Prerequisites Checklist

Before starting, ensure you have:

- [ ] **Python 3.9 or higher** installed ([Download here](https://python.org/downloads/))
- [ ] **OpenAI API account** with billing enabled ([Sign up here](https://platform.openai.com/))
- [ ] **Git** installed (optional, for version control) ([Download here](https://git-scm.com/))
- [ ] **Code editor** (VS Code, PyCharm, or similar)
- [ ] **Terminal/Command prompt** access
- [ ] **Stable internet connection** for API calls

## 🐍 Python Environment Setup

### Step 1: Verify Python Installation

```bash
python --version
# Should show Python 3.9.0 or higher

# If python command doesn't work, try:
python3 --version
```

**Troubleshooting Python Installation:**
- **Windows**: Download from python.org and ensure "Add to PATH" is checked
- **macOS**: Use Homebrew: `brew install python` or download from python.org
- **Linux**: Use package manager: `sudo apt-get install python3 python3-pip`

### Step 2: Create Virtual Environment

```bash
# Navigate to course directory
cd /path/to/course/directory

# Create virtual environment
python -m venv venv

# Alternative if python command doesn't work:
python3 -m venv venv
```

### Step 3: Activate Virtual Environment

**macOS/Linux:**
```bash
source venv/bin/activate
```

**Windows Command Prompt:**
```cmd
venv\Scripts\activate
```

**Windows PowerShell:**
```powershell
venv\Scripts\Activate.ps1
```

**Verification:** Your terminal prompt should now start with `(venv)`

### Step 4: Upgrade pip

```bash
pip install --upgrade pip
```

## 📦 Package Installation

### Install Course Dependencies

```bash
pip install -r requirements.txt
```

**Expected Installation Time:** 5-10 minutes depending on internet speed

### Verify Critical Packages

```bash
# Test LangGraph installation
python -c "import langgraph; print('LangGraph version:', langgraph.__version__)"

# Test OpenAI installation
python -c "import openai; print('OpenAI version:', openai.__version__)"

# Test Jupyter installation
jupyter --version
```

**Common Installation Issues:**

1. **Permission Errors (Windows)**:
   ```cmd
   pip install --user -r requirements.txt
   ```

2. **Compilation Errors (macOS)**:
   ```bash
   # Install Xcode command line tools
   xcode-select --install
   ```

3. **Dependency Conflicts**:
   ```bash
   pip install --upgrade --force-reinstall -r requirements.txt
   ```

## 🔑 OpenAI API Configuration

### Step 1: Get OpenAI API Key

1. **Visit**: [OpenAI Platform](https://platform.openai.com/)
2. **Sign in** or create account
3. **Add Payment Method**: Go to Billing → Payment methods
4. **Set Usage Limits**: Billing → Usage limits (recommended: $20 soft limit, $50 hard limit)
5. **Create API Key**: API keys → Create new secret key
6. **Copy Key**: Save it securely - you won't see it again!

### Step 2: Configure Environment Variables

Create `.env` file from template:
```bash
cp .env.example .env
```

Edit `.env` file with your settings:
```bash
# Required: OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4-turbo-preview
OPENAI_TEMPERATURE=0.7
OPENAI_MAX_TOKENS=4000

# Optional: Alternative Models
OPENAI_FALLBACK_MODEL=gpt-3.5-turbo
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Optional: LangSmith (for monitoring)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=langgraph-course

# Optional: Tavily (for web search)
TAVILY_API_KEY=your_tavily_api_key_here

# Optional: Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/langgraph_course

# Development Settings
JUPYTER_CONFIG_DIR=.jupyter
LOG_LEVEL=INFO
DEBUG_MODE=false
```

### Step 3: Test API Connection

Create test file `test_api.py`:
```python
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

try:
    # Test API connection
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello! This is a test."}],
        max_tokens=50
    )
    print("✅ OpenAI API connection successful!")
    print(f"Response: {response.choices[0].message.content}")
    
except Exception as e:
    print(f"❌ OpenAI API connection failed: {e}")
    print("\nTroubleshooting steps:")
    print("1. Check your API key in .env file")
    print("2. Verify billing is enabled in OpenAI account")
    print("3. Check usage limits haven't been exceeded")
```

Run the test:
```bash
python test_api.py
```

## 💾 Database Setup (Optional)

### SQLite (Default - No Setup Required)

SQLite is included with Python and requires no additional setup. Perfect for learning and development.

### PostgreSQL (Production Environments)

**Install PostgreSQL:**

**macOS (Homebrew):**
```bash
brew install postgresql
brew services start postgresql
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install postgresql postgresql-contrib
```

**Windows:**
Download installer from [postgresql.org](https://www.postgresql.org/download/windows/)

**Create Database:**
```bash
# Connect to PostgreSQL
psql postgres

# Create database and user
CREATE DATABASE langgraph_course;
CREATE USER course_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE langgraph_course TO course_user;
\q
```

**Update .env file:**
```bash
DATABASE_URL=postgresql://course_user:secure_password@localhost:5432/langgraph_course
```

## 📓 Jupyter Notebook Setup

### Configure Jupyter

```bash
# Install kernel for virtual environment
python -m ipykernel install --user --name=venv --display-name "LangGraph Course"

# Create Jupyter config directory
mkdir -p .jupyter

# Generate Jupyter config
jupyter notebook --generate-config
```

### Launch Jupyter

```bash
jupyter notebook
```

**Expected Result:** Browser opens with Jupyter interface showing course notebooks

### Jupyter Extensions (Optional)

```bash
# Install useful extensions
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user

# Enable helpful extensions
jupyter nbextension enable collapsible_headings/main
jupyter nbextension enable toc2/main
```

## 🔧 Development Tools Setup

### VS Code Configuration (Recommended)

**Install Extensions:**
1. Python
2. Jupyter
3. Python Docstring Generator
4. GitLens (if using git)

**VS Code Settings (`.vscode/settings.json`):**
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "jupyter.askForKernelRestart": false,
    "files.associations": {
        "*.ipynb": "jupyter-notebook"
    }
}
```

### Git Setup (Optional but Recommended)

```bash
# Initialize git repository
git init

# Create .gitignore
cat > .gitignore << EOF
# Environment
.env
venv/
__pycache__/
*.pyc

# Jupyter
.ipynb_checkpoints/
*.ipynb_meta

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Database
*.db
*.sqlite3
EOF

# Initial commit
git add .
git commit -m "Initial course setup"
```

## 🧪 Verification & Testing

### Run Complete Setup Test

Create `setup_verification.py`:
```python
import sys
import os
import importlib
from dotenv import load_dotenv

def test_python_version():
    """Test Python version compatibility."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Need 3.9+")
        return False

def test_package_imports():
    """Test critical package imports."""
    packages = [
        'langgraph', 'langchain', 'openai', 'pydantic', 
        'jupyter', 'psycopg2', 'numpy', 'requests'
    ]
    
    results = []
    for package in packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} - Installed")
            results.append(True)
        except ImportError:
            print(f"❌ {package} - Missing")
            results.append(False)
    
    return all(results)

def test_environment_variables():
    """Test environment variable configuration."""
    load_dotenv()
    
    required_vars = ['OPENAI_API_KEY']
    optional_vars = ['LANGCHAIN_API_KEY', 'TAVILY_API_KEY']
    
    results = []
    for var in required_vars:
        if os.getenv(var):
            print(f"✅ {var} - Configured")
            results.append(True)
        else:
            print(f"❌ {var} - Missing (Required)")
            results.append(False)
    
    for var in optional_vars:
        if os.getenv(var):
            print(f"✅ {var} - Configured (Optional)")
        else:
            print(f"⚠️  {var} - Not configured (Optional)")
    
    return all(results)

def test_openai_connection():
    """Test OpenAI API connection."""
    try:
        from openai import OpenAI
        load_dotenv()
        
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Simple test call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=10
        )
        
        print("✅ OpenAI API - Connected")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI API - Failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 LangGraph Course Setup Verification\n")
    
    tests = [
        test_python_version(),
        test_package_imports(),
        test_environment_variables(),
        test_openai_connection()
    ]
    
    if all(tests):
        print("\n🎉 Setup verification complete! You're ready to start the course.")
        print("Next step: Open 'day1_langgraph_foundations.ipynb' in Jupyter")
    else:
        print("\n⚠️  Setup issues detected. Please resolve the failed tests above.")
        print("Refer to TROUBLESHOOTING.md for help with common issues.")
```

Run verification:
```bash
python setup_verification.py
```

## 💰 Cost Management & Monitoring

### Set OpenAI Usage Limits

1. **Visit**: [OpenAI Usage Dashboard](https://platform.openai.com/usage)
2. **Set Soft Limit**: $20 (recommended for course)
3. **Set Hard Limit**: $50 (safety net)
4. **Enable Email Alerts**: Get notified at 50% and 90% usage

### Monitor Course Costs

**Expected Costs:**
- **Complete Course**: $10-20 total
- **Per Day**: $1-3 average
- **Per Notebook**: $0.50-2.00

**Cost Optimization Tips:**
- Use `gpt-3.5-turbo` for experimentation
- Use `gpt-4` only for complex tasks
- Set lower `max_tokens` limits
- Cache responses when possible

### Track Token Usage

```python
import tiktoken

def count_tokens(text, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Example usage
text = "Your prompt here"
tokens = count_tokens(text)
estimated_cost = tokens * 0.00003  # GPT-4 input cost per token
print(f"Estimated cost: ${estimated_cost:.4f}")
```

## 🆘 Common Issues & Solutions

### Issue: Import Errors

**Solution:**
```bash
# Refresh virtual environment
deactivate
source venv/bin/activate  # or activate script for Windows
pip install --upgrade -r requirements.txt
```

### Issue: OpenAI API Errors

**Solutions:**
1. **Invalid API Key**: Check `.env` file, regenerate key if needed
2. **Quota Exceeded**: Check usage limits in OpenAI dashboard
3. **Rate Limiting**: Add delays between API calls
4. **Billing Issues**: Ensure payment method is valid

### Issue: Jupyter Kernel Problems

**Solutions:**
```bash
# Reinstall kernel
python -m ipykernel install --user --name=venv --display-name "LangGraph Course" --force

# Restart Jupyter
jupyter notebook stop
jupyter notebook
```

### Issue: PostgreSQL Connection

**Solutions:**
1. **Check Service**: `brew services restart postgresql` (macOS)
2. **Verify Database**: `psql -l` to list databases
3. **Connection String**: Double-check DATABASE_URL format
4. **Permissions**: Ensure user has proper database access

## ✅ Setup Complete!

If you've reached this point successfully, your environment is ready for the LangGraph course!

**Next Steps:**
1. **Launch Jupyter**: `jupyter notebook`
2. **Open**: `day1_langgraph_foundations.ipynb`
3. **Start Learning**: Follow the structured course materials

**Need Help?**
- Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for specific issues
- Review [API_REFERENCE.md](API_REFERENCE.md) for quick references
- Join the [LangChain Discord](https://discord.gg/langchain) community

**Happy Learning!** 🚀