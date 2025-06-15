# Graphiti System Interaction Guide

## Overview
Graphiti provides multiple ways to interact with the knowledge graph system, from programmatic APIs to AI assistant integrations. This guide covers all available interaction methods and their use cases.

## 🎯 System Architecture

Graphiti operates as a **knowledge graph engine** with the following components:
- **Core Library**: Python package (`graphiti-core`) for direct integration
- **REST API Server**: FastAPI-based HTTP service for web applications
- **MCP Server**: Model Context Protocol server for AI assistants (Claude, Cursor, etc.)
- **Neo4j Database**: Graph database backend for data storage

---

## 🔧 Interaction Methods

### 1. 📚 **Direct Python Library Usage**

**Best for**: Custom applications, data science workflows, direct integration

#### Installation
```bash
pip install graphiti-core
```

#### Basic Usage
```python
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from datetime import datetime, timezone

# Initialize connection
graphiti = Graphiti("bolt://localhost:7687", "neo4j", "password")

# Build indices (one-time setup)
await graphiti.build_indices_and_constraints()

# Add data
await graphiti.add_episode(
    name="Meeting Notes",
    episode_body="John discussed the Q4 budget with Sarah",
    source=EpisodeType.text,
    reference_time=datetime.now(timezone.utc)
)

# Search the graph
results = await graphiti.search("Who discussed the budget?")
for result in results:
    print(f"Fact: {result.fact}")

# Close connection
await graphiti.close()
```

#### Key Methods
- `add_episode()`: Ingest text, JSON, or message data
- `search()`: Hybrid semantic + keyword search for relationships
- `_search()`: Advanced search with custom configurations
- `get_entity_edge()`: Retrieve specific relationships
- `delete_entity_edge()`: Remove relationships
- `build_indices_and_constraints()`: Initialize database schema

---

### 2. 🌐 **REST API Server**

**Best for**: Web applications, microservices, language-agnostic integrations

#### Setup & Running
```bash
# Using Docker (Recommended)
docker-compose up

# Or build locally
cd server/
pip install -r requirements.txt
uvicorn graph_service.main:app --host 0.0.0.0 --port 8000
```

#### Base URL
```
http://localhost:8000
```

#### 📋 Available Endpoints

##### **Data Ingestion**
```http
POST /messages
Content-Type: application/json

{
  "group_id": "conversation_123",
  "messages": [
    {
      "uuid": "msg_001",
      "name": "User Message",
      "content": "What's the weather like?",
      "role": "user",
      "role_type": "human",
      "timestamp": "2024-01-15T10:30:00Z",
      "source_description": "chat_app"
    }
  ]
}
```

```http
POST /entity-node
Content-Type: application/json

{
  "uuid": "entity_001",
  "group_id": "project_alpha",
  "name": "John Smith",
  "summary": "Senior Software Engineer working on AI projects"
}
```

##### **Data Retrieval**
```http
POST /search
Content-Type: application/json

{
  "query": "Who is working on AI projects?",
  "group_ids": ["project_alpha"],
  "max_facts": 10
}
```

```http
GET /entity-edge/{uuid}
```

```http
GET /episodes/{group_id}?last_n=50
```

```http
POST /get-memory
Content-Type: application/json

{
  "group_id": "conversation_123",
  "max_facts": 20,
  "messages": [
    {
      "content": "Tell me about our recent discussions",
      "role": "user",
      "role_type": "human"
    }
  ]
}
```

##### **Data Management**
```http
DELETE /entity-edge/{uuid}
```

```http
DELETE /group/{group_id}
```

```http
DELETE /episode/{uuid}
```

```http
POST /clear
```

#### 📖 API Documentation
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

### 3. 🤖 **MCP Server (AI Assistant Integration)**

**Best for**: Claude Desktop, Cursor, and other MCP-compatible AI assistants

#### What is MCP?
Model Context Protocol (MCP) allows AI assistants to interact with external systems. The Graphiti MCP server gives AI assistants **persistent memory** through knowledge graphs.

#### Setup for Claude Desktop

1. **Install Prerequisites**
```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone Graphiti
git clone https://github.com/getzep/graphiti.git
cd graphiti/mcp_server
```

2. **Configure Claude Desktop**
Add to Claude's MCP configuration file:
```json
{
  "mcpServers": {
    "graphiti-memory": {
      "transport": "stdio",
      "command": "/Users/<user>/.local/bin/uv",
      "args": [
        "run",
        "--isolated",
        "--directory",
        "/path/to/graphiti/mcp_server",
        "--project",
        ".",
        "graphiti_mcp_server.py",
        "--model",
        "gpt-4o-mini",
        "--group-id",
        "claude_session"
      ],
      "env": {
        "OPENAI_API_KEY": "your_openai_api_key",
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "password"
      }
    }
  }
}
```

#### Setup for Cursor
Similar configuration in Cursor's MCP settings.

#### MCP Capabilities
Once configured, AI assistants can:
- **Remember conversations** across sessions
- **Learn from interactions** and build knowledge over time
- **Answer questions** using accumulated knowledge
- **Manage data** (add, search, delete episodes and entities)
- **Organize information** by groups/namespaces

#### Example MCP Usage
```
User: "Remember that John Smith is our lead developer on the AI project"
Assistant: I'll remember that information about John Smith.
[Uses MCP to store: "John Smith is the lead developer on the AI project"]

User: "Who is working on AI projects?"
Assistant: Based on what I remember, John Smith is the lead developer on the AI project.
[Uses MCP to search and retrieve the stored information]
```

---

### 4. 🖥️ **Neo4j Browser (Database UI)**

**Best for**: Data exploration, debugging, advanced graph queries

#### Access
```
http://localhost:7474
```

#### Login
- **Username**: `neo4j`
- **Password**: `password` (or your configured password)

#### Sample Queries
```cypher
// View all entities
MATCH (n:Entity) RETURN n LIMIT 25

// View all relationships
MATCH (n)-[r:RELATES_TO]->(m) RETURN n, r, m LIMIT 25

// Search for specific entities
MATCH (n:Entity) WHERE n.name CONTAINS "John" RETURN n

// View episodes
MATCH (e:Episodic) RETURN e ORDER BY e.created_at DESC LIMIT 10

// Complex graph traversal
MATCH path = (start:Entity)-[*1..3]-(end:Entity)
WHERE start.name = "John Smith"
RETURN path LIMIT 10
```

---

## 🎯 **Use Case Examples**

### 1. **Customer Support Knowledge Base**
```python
# Add support tickets
await graphiti.add_episode(
    name="Ticket #1234",
    episode_body="Customer John reported login issues with 2FA",
    source=EpisodeType.text,
    source_description="support_ticket"
)

# Search for similar issues
results = await graphiti.search("login problems two factor authentication")
```

### 2. **Meeting Notes & Action Items**
```python
# Add meeting transcript
await graphiti.add_episode(
    name="Weekly Standup",
    episode_body={
        "attendees": ["Alice", "Bob", "Charlie"],
        "action_items": ["Deploy to staging", "Review PR #456"],
        "blockers": ["Waiting for API keys"]
    },
    source=EpisodeType.json
)

# Find action items
results = await graphiti.search("what are the pending action items?")
```

### 3. **Research Paper Analysis**
```python
# Add research papers
await graphiti.add_episode(
    name="Attention Is All You Need",
    episode_body="Transformer architecture paper by Vaswani et al...",
    source=EpisodeType.text,
    source_description="research_paper"
)

# Find related concepts
results = await graphiti.search("transformer attention mechanism")
```

### 4. **Personal Knowledge Management**
```python
# Add personal notes
await graphiti.add_episode(
    name="Book Notes: Atomic Habits",
    episode_body="Habit stacking: link new habits to existing ones...",
    source=EpisodeType.text
)

# Retrieve insights
results = await graphiti.search("how to build better habits?")
```

---

## 🔐 **Security & Configuration**

### Environment Variables
```bash
# Required
OPENAI_API_KEY=your_openai_api_key

# Neo4j Connection
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_secure_password

# Optional Performance Tuning
SEMAPHORE_LIMIT=20
USE_PARALLEL_RUNTIME=false
MAX_REFLEXION_ITERATIONS=0

# Optional Alternative LLM Providers
ANTHROPIC_API_KEY=your_anthropic_key
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
```

### Production Considerations
- **Database Security**: Use strong Neo4j passwords and network isolation
- **API Keys**: Store securely, rotate regularly
- **Rate Limiting**: Implement rate limiting for public APIs
- **Data Privacy**: Consider data encryption and access controls
- **Monitoring**: Set up logging and monitoring for production deployments

---

## 🚀 **Getting Started Recommendations**

### For Developers
1. **Start with Python Library**: Use the quickstart example to understand core concepts
2. **Explore REST API**: Build a simple web interface
3. **Try MCP Integration**: Experience AI assistant memory capabilities

### For AI Assistant Users
1. **Set up MCP Server**: Configure Claude Desktop or Cursor
2. **Start Simple**: Have conversations and let the assistant remember key facts
3. **Explore Advanced**: Use group IDs to organize different projects/contexts

### For Data Scientists
1. **Use Python Library**: Direct integration for data analysis workflows
2. **Leverage Neo4j Browser**: Explore graph patterns and relationships
3. **Custom Queries**: Write Cypher queries for advanced analytics

---

## 📚 **Additional Resources**

- **Documentation**: [https://help.getzep.com/graphiti](https://help.getzep.com/graphiti)
- **GitHub Repository**: [https://github.com/getzep/graphiti](https://github.com/getzep/graphiti)
- **Discord Community**: [Zep Discord #Graphiti channel](https://discord.com/invite/W8Kw6bsgXQ)
- **Examples**: Check the `examples/` directory in the repository
- **Paper**: [Zep: A Temporal Knowledge Graph Architecture for Agent Memory](https://arxiv.org/abs/2501.13956)

---

## ❓ **Frequently Asked Questions**

**Q: Is there a web UI for Graphiti?**
A: Currently, there's no dedicated web UI. You can use the Neo4j Browser for graph visualization, or build custom UIs using the REST API.

**Q: Can I use other LLM providers besides OpenAI?**
A: Yes! Graphiti supports Anthropic, Google Gemini, Groq, and Azure OpenAI. See the configuration section for details.

**Q: How does Graphiti handle data privacy?**
A: Graphiti processes data locally in your infrastructure. Data is stored in your Neo4j database and only sent to LLM providers for processing (entity extraction, embeddings).

**Q: Can I scale Graphiti for production use?**
A: Yes! Use Neo4j clustering, implement proper caching, and consider horizontal scaling of the API servers for production deployments.

**Q: What's the difference between episodes, entities, and relationships?**
A: **Episodes** are raw data inputs (text, JSON, messages). **Entities** are extracted concepts (people, places, things). **Relationships** are connections between entities with associated facts. 

---

## 📋 **Appendix: Web UI and Visualization Options**

### 🌐 **Current Web UI Status**

**Graphiti does not currently have a dedicated web UI.** However, there are several visualization and interface options available for interacting with Graphiti knowledge graphs.

### 🎯 **Available Visualization Options**

#### 1. **Neo4j Browser (Primary Graph Visualization)**
- **Access**: `http://localhost:7474`
- **Purpose**: Database-level graph exploration and visualization
- **Login**: Username: `neo4j`, Password: `password` (or your configured password)

**Key Capabilities:**
- Interactive graph visualization with drag-and-drop nodes
- Real-time query execution and results display
- Relationship exploration and filtering
- Historical data inspection
- Performance monitoring and query optimization

**Advanced Neo4j Browser Queries:**
```cypher
// Temporal relationship queries
MATCH (n)-[r:RELATES_TO]->(m) 
WHERE r.valid_at <= datetime() AND (r.invalid_at IS NULL OR r.invalid_at > datetime())
RETURN n, r, m

// Entity clustering analysis
MATCH (n:Entity)-[r]-(m:Entity)
WITH n, count(r) as connections
WHERE connections > 5
RETURN n ORDER BY connections DESC

// Time-based graph evolution
MATCH (n)-[r]->(m)
WHERE r.created_at >= datetime('2024-01-01T00:00:00Z')
RETURN n, r, m, r.created_at
ORDER BY r.created_at
```

#### 2. **REST API for Custom Web UIs**
- **Base URL**: `http://localhost:8000`
- **Documentation**: `/docs` (Swagger UI) and `/redoc`
- **Use Case**: Foundation for building custom web interfaces

**Visualization-Specific Endpoints:**
```http
# Get graph structure data
GET /graph/structure?group_id=project_alpha

# Export graph data for visualization libraries
GET /export/json?format=d3&group_id=project_alpha

# Get temporal snapshots
GET /graph/snapshot?timestamp=2024-01-15T10:30:00Z
```

### 🛠️ **Building Custom Web UIs**

#### **Recommended Visualization Libraries**

| Library | Best For | Complexity |
|---------|----------|------------|
| **D3.js** | Custom, complex visualizations | High |
| **Cytoscape.js** | Interactive graph networks | Medium |
| **Vis.js** | Quick network diagrams | Low |
| **React Flow** | Modern React applications | Medium |
| **Sigma.js** | Large graph performance | Medium |

#### **Example: Custom Web UI with D3.js**
```javascript
// Fetch graph data from Graphiti REST API
async function loadGraphData(groupId) {
  const response = await fetch(`http://localhost:8000/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: "*",
      group_ids: [groupId],
      max_facts: 100
    })
  });
  return await response.json();
}

// Render with D3.js
function renderGraph(data) {
  const svg = d3.select("#graph-container")
    .append("svg")
    .attr("width", 800)
    .attr("height", 600);

  // Create force simulation for node positioning
  const simulation = d3.forceSimulation(data.nodes)
    .force("link", d3.forceLink(data.links).id(d => d.id))
    .force("charge", d3.forceManyBody().strength(-300))
    .force("center", d3.forceCenter(400, 300));

  // Add temporal controls
  addTimelineSlider(data.temporal_range);
}
```

#### **Temporal Visualization Features**
```javascript
// Timeline slider for temporal navigation
function addTimelineSlider(timeRange) {
  const slider = d3.slider()
    .domain(timeRange)
    .on("slide", function(evt, value) {
      filterGraphByDate(value);
      updateNodeVisibility();
    });
}

// Dynamic relationship updates
function updateRelationships(timestamp) {
  edges.style("opacity", d => {
    return isActiveAt(d, timestamp) ? 1.0 : 0.2;
  });
}
```

### 🔄 **Integration Patterns**

#### **Web Dashboard Architecture**
```
Frontend (React/Vue/Angular)
├── Graph Visualization Component
├── Search Interface
├── Temporal Controls
├── Entity Management Panel
└── Real-time Updates (WebSocket)

Backend Integration
├── Graphiti REST API
├── Custom Middleware
├── Caching Layer (Redis)
└── WebSocket Server
```

#### **Real-time Updates**
```python
# WebSocket endpoint for live graph updates
from fastapi import WebSocket
import asyncio

@app.websocket("/ws/graph/{group_id}")
async def websocket_endpoint(websocket: WebSocket, group_id: str):
    await websocket.accept()
    
    while True:
        # Listen for graph changes
        changes = await graphiti.get_recent_changes(group_id)
        if changes:
            await websocket.send_json({
                "type": "graph_update",
                "changes": changes
            })
        await asyncio.sleep(1)
```

### 📊 **Alternative Visualization Approaches**

#### **1. Jupyter Notebook Integration**
```python
import networkx as nx
import matplotlib.pyplot as plt
from graphiti_core import Graphiti

# Create NetworkX graph from Graphiti data
def create_networkx_graph(graphiti_results):
    G = nx.Graph()
    for result in graphiti_results:
        G.add_edge(result.source.name, result.target.name, 
                  fact=result.fact, weight=result.score)
    return G

# Visualize in Jupyter
G = create_networkx_graph(search_results)
plt.figure(figsize=(12, 8))
nx.draw(G, with_labels=True, node_color='lightblue', 
        node_size=1500, font_size=10)
plt.show()
```

#### **2. Graph Analytics Dashboards**
```python
# Export to Gephi format
def export_to_gephi(graphiti_instance, filename):
    results = await graphiti_instance.search("*", max_facts=1000)
    
    nodes = []
    edges = []
    
    for result in results:
        nodes.extend([result.source, result.target])
        edges.append({
            'source': result.source.uuid,
            'target': result.target.uuid,
            'weight': result.score,
            'label': result.fact
        })
    
    # Write GEXF format
    write_gexf(nodes, edges, filename)
```

### 🎨 **UI/UX Design Considerations**

#### **Essential Interface Elements**
- **Timeline Scrubber**: Navigate temporal changes
- **Search Bar**: Full-text and semantic search
- **Filter Panel**: Entity types, date ranges, groups
- **Node Inspector**: Detailed entity/relationship views
- **Graph Controls**: Zoom, pan, layout algorithms
- **Export Options**: PNG, SVG, JSON, CSV formats

#### **Responsive Design Patterns**
```css
/* Mobile-first graph container */
.graph-container {
  width: 100%;
  height: 60vh;
  position: relative;
}

@media (min-width: 768px) {
  .graph-container {
    height: 80vh;
  }
  
  .sidebar {
    position: fixed;
    width: 300px;
    right: 0;
  }
}
```

### 🚀 **Development Roadmap**

#### **Community Contributions Welcome**
The Graphiti team welcomes community contributions for web UI development:

1. **Basic Graph Visualizer**: D3.js-based simple interface
2. **Dashboard Template**: React/Vue starter kit
3. **Analytics Interface**: Business intelligence focused
4. **Mobile App**: React Native or Flutter implementation

#### **Integration Examples**
Check the `examples/` directory in the [Graphiti repository](https://github.com/getzep/graphiti) for:
- Basic web UI prototypes
- Visualization library integrations
- Custom dashboard implementations
- Mobile-responsive designs

### 📚 **Additional Visualization Resources**

- **Neo4j Bloom**: Commercial graph visualization tool (compatible with Graphiti's Neo4j backend)
- **Graphistry**: GPU-accelerated graph visualization (enterprise)
- **Linkurious**: Enterprise graph analytics platform
- **yEd**: Desktop graph editing and visualization

### 💡 **Best Practices for Custom UIs**

1. **Performance**: Implement virtual scrolling for large graphs
2. **Caching**: Cache frequent queries and graph layouts
3. **Progressive Loading**: Load graph data incrementally
4. **User Experience**: Provide context menus and keyboard shortcuts
5. **Accessibility**: Ensure screen reader compatibility
6. **Responsive Design**: Support mobile and tablet interfaces

---

*For the latest updates on web UI development and community contributions, visit the [Graphiti GitHub Issues](https://github.com/getzep/graphiti/issues) and [Discord Community](https://discord.com/invite/W8Kw6bsgXQ).* 