# Graphiti MCP Server with Gemini and Ollama

This directory contains the updated Graphiti MCP server that uses **Gemini (Vertex AI)** for LLM operations and **Ollama** for embeddings and reranking instead of OpenAI.

## Files Created/Modified

### New Files
1. **`vertex_ai_client.py`** - Vertex AI LLM client with concurrency limiting and retry logic
2. **`ollama_embedder.py`** - Non-batching Ollama embedder that processes embeddings sequentially
3. **`ollama_reranker_client.py`** - Ollama-based cross-encoder for reranking using BGE embeddings
4. **`graphiti_mcp_server_gemini_ollama.py`** - New simplified MCP server using Gemini and Ollama

### Key Changes Made

#### 1. LLM Client (OpenAI → Gemini)
- **Before**: Used OpenAI API with `OpenAIClient`
- **After**: Uses Google Vertex AI with `VertexAIClient`
- **Configuration**: 
  - Model: `gemini-2.5-flash` (default)
  - Temperature: `0.3` (instead of `0.0`)
  - Environment variables: `GOOGLE_CLOUD_PROJECT`, `GCP_LOCATION`

#### 2. Embedder (OpenAI → Ollama)
- **Before**: Used OpenAI text embeddings via `OpenAIEmbedder`
- **After**: Uses Ollama with `NonBatchingOllamaEmbedder`
- **Configuration**:
  - Model: `nomic-embed-text` (default)
  - Base URL: `http://localhost:11434/v1` (default)
  - API Key: `"ollama"` (placeholder)

#### 3. Cross-Encoder (None → Ollama BGE)
- **Before**: Used default cross-encoder or none
- **After**: Uses `OllamaRerankerClient` with BGE-M3 embeddings
- **Features**:
  - Cosine similarity-based reranking
  - Uses BGE-M3 model for better multilingual performance
  - Fallback to neutral scores on errors

#### 4. Configuration Simplification
- **Before**: Complex configuration with OpenAI/Azure branching
- **After**: Simplified `GraphitiConfig` class with Gemini/Ollama settings
- **Environment Variables**:
  ```
  # Gemini/Vertex AI
  GOOGLE_CLOUD_PROJECT=your-project-id
  GCP_LOCATION=global
  MODEL_NAME=gemini-2.5-flash
  LLM_TEMPERATURE=0.3
  
  # Ollama
  OLLAMA_BASE_URL=http://localhost:11434/v1
  EMBEDDER_MODEL=nomic-embed-text
  
  # Neo4j (unchanged)
  NEO4J_URI=bolt://localhost:7687
  NEO4J_USER=neo4j
  NEO4J_PASSWORD=password
  
  # Graphiti
  GROUP_ID=default
  USE_CUSTOM_ENTITIES=false
  ```

## What Stays the Same

All the core Graphiti functionality remains unchanged:

1. **MCP Tools**: `add_memory`, `search_memory_nodes`, `search_memory_facts`, `clear_graph`
2. **Entity Types**: `Requirement`, `Preference`, `Procedure`
3. **Search Configurations**: Still uses hybrid search with BM25 + vector similarity
4. **Neo4j Integration**: Database operations remain identical
5. **API Responses**: Same response formats and error handling
6. **Episode Processing**: Queue-based episode processing unchanged

## Benefits of the Switch

### Gemini Advantages
- **Higher Token Limits**: 65,500 tokens vs 8K-32K for OpenAI
- **Better Structured Output**: Native JSON schema support
- **Cost Efficiency**: Competitive pricing with Google Cloud credits
- **Thinking Budget**: Support for reasoning steps (Gemini 2.5+)

### Ollama Advantages
- **Local Deployment**: No external API calls for embeddings
- **Privacy**: Data stays local
- **Cost**: No per-token embedding costs
- **Flexibility**: Easy model switching

### Combined Benefits
- **Reduced Dependencies**: No more OpenAI API key management
- **Better Performance**: Local embeddings + powerful cloud LLM
- **Cost Control**: Predictable Vertex AI usage + free local embeddings

## Setup Requirements

1. **Google Cloud**: 
   - Project with Vertex AI API enabled
   - Authentication configured (`gcloud auth application-default login`)
   - Set `GOOGLE_CLOUD_PROJECT` environment variable

2. **Ollama**:
   - Install Ollama locally
   - Pull required models:
     ```bash
     ollama pull nomic-embed-text
     ollama pull bge-m3
     ```

3. **Python Dependencies**:
   ```bash
   pip install google-genai aiohttp tenacity
   ```

## Usage

Run the new server:
```bash
cd mcp_server
python graphiti_mcp_server_gemini_ollama.py
```

The server provides the same MCP interface but now uses Gemini and Ollama under the hood.

## Migration Notes

- **Existing Data**: No migration needed - Neo4j data remains compatible
- **API Compatibility**: All MCP tools have the same signatures
- **Performance**: Initial requests may be slower due to Vertex AI cold starts
- **Monitoring**: Check Google Cloud Console for Vertex AI usage and quotas

## Troubleshooting

1. **Vertex AI Errors**: Verify project setup and API enablement
2. **Ollama Connection**: Ensure Ollama is running on the specified port
3. **Model Loading**: Check that required models are pulled in Ollama
4. **Concurrency**: Adjust `VERTEX_AI_MAX_CONCURRENT` if hitting rate limits 