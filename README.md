# 🤖 7-Day LangGraph Multi-Agent Systems Course

**Master LangGraph multi-agent development from foundations to production in 7 comprehensive days**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-green)](https://openai.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.14-orange)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## 📖 Course Overview

This intensive 7-day course teaches you to build sophisticated multi-agent systems using LangGraph and OpenAI's API. Each day contains 2 hours of structured learning with hands-on coding exercises, practical projects, and real-world applications.

**🎯 Target Audience**: Software developers new to multi-agent systems  
**⏱️ Duration**: 7 days × 2 hours/day = 14 hours total  
**🔧 Format**: 7 comprehensive, self-contained Jupyter notebooks  
**🤖 LLM Provider**: OpenAI (GPT-4, GPT-3.5-turbo) via API keys  

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- Basic Python programming knowledge
- Familiarity with async/await concepts (helpful but not required)

### Installation

1. **Clone or download this course**
   ```bash
   cd /path/to/course/directory
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   
   # On macOS/Linux:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key and other settings
   ```

5. **Launch Jupyter**
   ```bash
   jupyter notebook
   ```

### ⚠️ Important Setup Notes
- **API Costs**: This course uses OpenAI's paid API. Budget approximately $10-20 for completion
- **Rate Limits**: Configure rate limiting in your OpenAI account to prevent unexpected charges
- **Security**: Never commit your `.env` file or expose API keys in notebooks

## 📚 Course Structure

### **Day 1: Foundations & Type-Safe Development**
**File**: `day1_langgraph_foundations.ipynb`
- Multi-agent systems introduction
- Graph architecture vs traditional approaches  
- Type safety with Pydantic models
- First type-safe agent with GPT-4
- Structured output and function calling

**🎯 Learning Outcomes**: Build your first LangGraph agent with type-safe operations

---

### **Day 2: State Management & Persistence**
**File**: `day2_state_persistence.ipynb`
- Graph state management patterns
- InMemorySaver, SqliteSaver, PostgresSaver
- State persistence and recovery
- Error handling with checkpoints
- Migration strategies

**🎯 Learning Outcomes**: Implement persistent, recoverable multi-agent systems

---

### **Day 3: Memory Systems & Knowledge Management**
**File**: `day3_memory_systems.ipynb`
- Semantic, episodic, and procedural memory
- Short-term vs long-term memory implementation
- Vector storage with OpenAI embeddings
- Knowledge extraction and management
- Cross-session persistence

**🎯 Learning Outcomes**: Build agents with sophisticated memory capabilities

---

### **Day 4: Multi-Agent Communication & Handoffs**
**File**: `day4_communication_handoffs.ipynb`
- Agent handoff mechanisms with Command objects
- Supervisor pattern implementation
- Secure data injection patterns
- Message validation with Pydantic
- Communication protocols

**🎯 Learning Outcomes**: Create coordinated multi-agent systems with secure communication

---

### **Day 5: Advanced Architectures & Tool Integration**
**File**: `day5_advanced_architectures.ipynb`
- Hierarchical multi-agent systems
- Parallel execution and map-reduce patterns
- Custom tool executors with validation
- External API integration (Tavily, custom APIs)
- Performance optimization strategies

**🎯 Learning Outcomes**: Build complex, production-ready architectures

---

### **Day 6: Production Tools & Monitoring**
**File**: `day6_production_monitoring.ipynb`
- LangSmith integration for observability
- LangGraph Studio workflows
- PostgreSQL for production environments
- Human-in-the-loop implementations
- Cost monitoring and optimization

**🎯 Learning Outcomes**: Deploy monitored, observable production systems

---

### **Day 7: Deployment & Real-World Applications**
**File**: `day7_deployment_realworld.ipynb`
- LangGraph Platform deployment
- Docker containerization strategies
- Security implementation patterns
- Rate limiting and error handling
- Complete business workflow examples

**🎯 Learning Outcomes**: Deploy secure, scalable multi-agent applications

## 🛠️ Tools & Technologies

### Core Technologies
- **[LangGraph](https://langchain-ai.github.io/langgraph/)**: Graph-based multi-agent framework
- **[OpenAI API](https://platform.openai.com/)**: GPT-4 and GPT-3.5-turbo for LLM capabilities
- **[Pydantic](https://docs.pydantic.dev/)**: Type-safe data validation and settings
- **[PostgreSQL](https://www.postgresql.org/)**: Production database for state persistence

### Development Environment
- **[Jupyter Notebooks](https://jupyter.org/)**: Interactive development environment
- **[Python 3.9+](https://python.org)**: Programming language
- **[LangSmith](https://smith.langchain.com/)**: Monitoring and observability
- **[Docker](https://docker.com)**: Containerization for deployment

### External Services
- **[Tavily](https://tavily.com/)**: Web search API integration
- **[ChromaDB](https://www.trychroma.com/)**: Vector database for embeddings
- **[Redis](https://redis.io/)**: Caching and session management

## 📖 Additional Resources

### Quick References
- **[SETUP_GUIDE.md](SETUP_GUIDE.md)**: Detailed environment setup instructions
- **[API_REFERENCE.md](API_REFERENCE.md)**: Common LangGraph and OpenAI patterns
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)**: Solutions to common issues

### External Documentation
- **[LangGraph Documentation](https://langchain-ai.github.io/langgraph/)**
- **[OpenAI API Documentation](https://platform.openai.com/docs)**
- **[LangChain Documentation](https://python.langchain.com/)**
- **[Pydantic Documentation](https://docs.pydantic.dev/)**

## 💡 Learning Tips

### Before You Start
1. **Review Prerequisites**: Ensure you're comfortable with Python basics and async programming
2. **Set Budget Limits**: Configure OpenAI usage limits to control costs
3. **Read Setup Guide**: Follow [SETUP_GUIDE.md](SETUP_GUIDE.md) carefully for environment configuration

### During the Course
1. **Run Code Incrementally**: Execute notebook cells step-by-step to understand each concept
2. **Experiment Freely**: Modify examples to test your understanding
3. **Save Your Work**: Regularly save notebooks and commit changes if using git
4. **Monitor Costs**: Check OpenAI usage regularly to stay within budget

### After Each Day
1. **Complete Exercises**: Don't skip the practical exercises - they reinforce learning
2. **Review Concepts**: Go back to theory sections if implementation is unclear
3. **Build on Examples**: Try extending examples with your own ideas

## 🆘 Getting Help

### Common Issues
- **API Errors**: Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for OpenAI API issues
- **Installation Problems**: Refer to [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed setup steps
- **Code Errors**: Each notebook includes debugging tips and common pitfalls

### Cost Management
- **Monitor Usage**: Use OpenAI dashboard to track API costs
- **Set Limits**: Configure usage limits in your OpenAI account
- **Optimize Calls**: Learn cost optimization techniques in Day 6

### Community Resources
- **[LangChain Discord](https://discord.gg/langchain)**: Active community support
- **[OpenAI Community](https://community.openai.com/)**: Official OpenAI forums
- **[Stack Overflow](https://stackoverflow.com/questions/tagged/langchain)**: Technical Q&A

## 🏆 Course Completion

Upon completing this course, you'll be able to:

✅ **Build Multi-Agent Systems**: Create sophisticated agent coordination patterns  
✅ **Implement State Management**: Design persistent, recoverable systems  
✅ **Integrate Memory Systems**: Add long-term knowledge capabilities  
✅ **Deploy Production Systems**: Launch monitored, scalable applications  
✅ **Optimize Costs**: Efficiently manage OpenAI API usage  
✅ **Handle Security**: Implement secure communication and data handling  

## 📄 License

This course is provided under the MIT License. See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[LangChain Team](https://langchain.com/)** for the LangGraph framework
- **[OpenAI](https://openai.com/)** for powerful language models
- **[Jupyter Project](https://jupyter.org/)** for the interactive development environment

---

**Ready to start?** Begin with [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed environment configuration, then open `day1_langgraph_foundations.ipynb` to start your journey into multi-agent systems!

**Questions?** Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) or refer to the resources section above.