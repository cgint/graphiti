# How to Run Graphiti with Local Ollama

This guide shows you how to configure Graphiti to use local Ollama for both LLM and embeddings instead of cloud-based OpenAI services.

## Key Requirements

- **Ollama installed locally** with OpenAI-compatible API endpoints
- **Graphiti supports custom clients** through its constructor parameters
- **Compatible embedding models** available in Ollama

## Prerequisites

### 1. Install and Setup Ollama

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve
```

### 2. Pull Required Models

```bash
# Pull a good LLM model (choose one)
ollama pull llama3.1:8b      # Recommended for most tasks
ollama pull llama3.1:70b     # Better quality, requires more resources
ollama pull codellama:13b    # Good for code-related tasks

# Pull an embedding model
ollama pull nomic-embed-text:latest  # Recommended for embeddings
# Alternative: ollama pull mxbai-embed-large:latest
```

### 3. Verify Ollama API

Test that Ollama's OpenAI-compatible API is working:

```bash
# Test LLM endpoint
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1:8b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Test embeddings endpoint
curl http://localhost:11434/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nomic-embed-text:latest",
    "input": "Hello world"
  }'
```

## Configuration Options

### Option 1: Using Environment Variables

Create a `.env` file:

```env
# Ollama LLM Configuration
GRAPHITI_LLM_API_BASE=http://localhost:11434/v1
GRAPHITI_LLM_MODEL=llama3.1:8b
GRAPHITI_LLM_API_KEY=ollama  # Can be any non-empty string

# Ollama Embeddings Configuration
GRAPHITI_EMBEDDER_API_BASE=http://localhost:11434/v1
GRAPHITI_EMBEDDER_MODEL=nomic-embed-text:latest
GRAPHITI_EMBEDDER_API_KEY=ollama  # Can be any non-empty string

# Database Configuration (choose one)
# For Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# For FalkorDB
FALKORDB_HOST=localhost
FALKORDB_PORT=6379
FALKORDB_USERNAME=
FALKORDB_PASSWORD=
```

### Option 2: Direct Python Configuration

```python
import asyncio
from graphiti_core import Graphiti
from graphiti_core.llm_client import OpenAIGenericClient
from graphiti_core.embedder import OpenAIEmbedder
from graphiti_core.driver import Neo4jDriver  # or FalkorDBDriver

async def setup_graphiti_with_ollama():
    # Configure LLM client for Ollama
    llm_client = OpenAIGenericClient(
        api_key="ollama",  # Can be any non-empty string
        api_base="http://localhost:11434/v1",
        model="llama3.1:8b"
    )
    
    # Configure embedder for Ollama
    embedder = OpenAIEmbedder(
        api_key="ollama",  # Can be any non-empty string
        api_base="http://localhost:11434/v1",
        model="nomic-embed-text:latest"
    )
    
    # Configure database driver
    driver = Neo4jDriver(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password"
    )
    
    # Initialize Graphiti with custom clients
    graphiti = Graphiti(
        driver=driver,
        llm_client=llm_client,
        embedder=embedder
    )
    
    return graphiti

# Usage example
async def main():
    graphiti = await setup_graphiti_with_ollama()
    
    # Add some content
    await graphiti.add_episode(
        name="Test Episode",
        episode_body="This is a test episode to verify Ollama integration works.",
        source_description="Test data"
    )
    
    # Search for content
    results = await graphiti.search("test episode")
    print("Search results:", results)

if __name__ == "__main__":
    asyncio.run(main())
```

## Complete Working Example

Here's a complete script based on the quickstart example:

```python
"""
Graphiti + Ollama Integration Example
Run this after setting up Ollama and your database
"""
import asyncio
import os
from graphiti_core import Graphiti
from graphiti_core.llm_client import OpenAIGenericClient
from graphiti_core.embedder import OpenAIEmbedder
from graphiti_core.driver import Neo4jDriver

async def main():
    """Run Graphiti with local Ollama"""
    
    # Configure Ollama LLM Client
    llm_client = OpenAIGenericClient(
        api_key="ollama",
        api_base="http://localhost:11434/v1",
        model="llama3.1:8b",
        temperature=0.1
    )
    
    # Configure Ollama Embedder
    embedder = OpenAIEmbedder(
        api_key="ollama", 
        api_base="http://localhost:11434/v1",
        model="nomic-embed-text:latest"
    )
    
    # Configure Database (Neo4j example)
    driver = Neo4jDriver(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password")
    )
    
    # Initialize Graphiti
    graphiti = Graphiti(
        driver=driver,
        llm_client=llm_client,
        embedder=embedder
    )
    
    print("🚀 Graphiti initialized with Ollama!")
    
    # Test adding content
    print("📝 Adding test episode...")
    await graphiti.add_episode(
        name="Ollama Integration Test",
        episode_body="""
        Today I successfully configured Graphiti to use local Ollama instead of OpenAI.
        This allows me to run everything locally without sending data to external APIs.
        The setup involved configuring both the LLM client and embedder to use Ollama's
        OpenAI-compatible endpoints at localhost:11434.
        """,
        source_description="Local setup test"
    )
    
    # Test search functionality
    print("🔍 Testing search...")
    results = await graphiti.search("Ollama configuration")
    print(f"Found {len(results.edges)} relevant connections!")
    
    # Test node search
    print("🔎 Testing node search...")
    node_results = await graphiti.search_nodes("local setup")
    print(f"Found {len(node_results)} relevant nodes!")
    
    print("✅ Graphiti + Ollama integration working successfully!")

if __name__ == "__main__":
    asyncio.run(main())
```

## Recommended Models

### For LLM (Choose based on your hardware):
- **llama3.1:8b** - Good balance of quality and performance
- **llama3.1:70b** - Best quality, requires powerful hardware
- **codellama:13b** - Specialized for code and technical content
- **mistral:7b** - Fast and efficient alternative

### For Embeddings:
- **nomic-embed-text:latest** - Recommended, optimized for text similarity
- **mxbai-embed-large:latest** - High-quality alternative
- **all-minilm:latest** - Smaller, faster option

## Performance Tips

1. **Model Size**: Start with smaller models (7B-8B) and scale up based on performance needs
2. **Memory**: Ensure sufficient RAM for your chosen models
3. **Concurrent Requests**: Ollama handles concurrent requests well, but monitor resource usage
4. **Embedding Caching**: Graphiti caches embeddings, reducing repeated computation

## Troubleshooting

### Common Issues:

1. **"Connection refused"**: Ensure Ollama is running (`ollama serve`)
2. **Model not found**: Pull the model first (`ollama pull model_name`)
3. **Slow performance**: Try smaller models or increase system resources
4. **API errors**: Verify Ollama's OpenAI compatibility is enabled

### Debug Commands:

```bash
# Check Ollama status
ollama list

# Monitor Ollama logs
ollama logs

# Test API directly
curl http://localhost:11434/v1/models
```

## Security Benefits

Running Graphiti with local Ollama provides:
- **Data Privacy**: All processing happens locally
- **No API Costs**: No charges for LLM or embedding usage
- **Offline Capability**: Works without internet connection
- **Full Control**: Complete control over model versions and configurations

## Next Steps

1. **Scale Up**: Try larger models as your hardware allows
2. **Fine-tune**: Consider fine-tuning models for your specific domain
3. **Optimize**: Profile performance and adjust model choices
4. **Integrate**: Connect with your existing applications and workflows

---

**Note**: This setup requires sufficient local compute resources. Monitor CPU, memory, and disk usage when running large language models locally. 