# Graphiti Pipeline & Search Integration Points

> **Document Purpose**: Detail how the Episode-to-Graph Pipeline (Document 1) and Search System (Document 2) integrate and depend on each other. This is critical for understanding the full system architecture.

---

## Table of Contents

1. [Integration Overview](#1-integration-overview)
2. [Data Flow Between Systems](#2-data-flow-between-systems)
3. [Node Deduplication Uses Search](#3-node-deduplication-uses-search)
4. [Edge Resolution Uses Search](#4-edge-resolution-uses-search)
5. [Embeddings: Pipeline Generation → Search Consumption](#5-embeddings-pipeline-generation--search-consumption)
6. [Shared Data Models](#6-shared-data-models)
7. [Configuration Flow](#7-configuration-flow)
8. [Temporal Logic Integration](#8-temporal-logic-integration)
9. [Community Operations](#9-community-operations)
10. [Critical Dependencies](#10-critical-dependencies)
11. [DSPy Reimplementation Implications](#11-dspy-reimplementation-implications)

---

## 1. Integration Overview

The Pipeline and Search systems are **tightly coupled** — the pipeline cannot deduplicate or resolve entities/facts without search, and search depends on the graph built by the pipeline.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PIPELINE ←→ SEARCH INTEGRATION                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────┐                      ┌──────────────────────────┐  │
│  │   EPISODE-TO-GRAPH  │                      │      SEARCH SYSTEM       │  │
│  │      PIPELINE       │                      │                          │  │
│  ├─────────────────────┤                      ├──────────────────────────┤  │
│  │                     │  ──── USES ────────► │                          │  │
│  │ 1. Extract Nodes    │                      │  node_similarity_search  │  │
│  │ 2. Dedupe Nodes ────┼──────────────────────┼► NODE_HYBRID_SEARCH_RRF  │  │
│  │ 3. Extract Edges    │                      │                          │  │
│  │ 4. Resolve Edges ───┼──────────────────────┼► EDGE_HYBRID_SEARCH_RRF  │  │
│  │ 5. Invalidation ────┼──────────────────────┼► invalidation_candidates │  │
│  │ 6. Persist          │                      │                          │  │
│  │         │           │                      │                          │  │
│  │         ▼           │  ◄─── QUERIES ────── │  User Search Queries     │  │
│  │    GRAPH DATABASE   │                      │                          │  │
│  │    (Neo4j/etc)      │                      │                          │  │
│  │         │           │                      │                          │  │
│  │         ▼           │                      │                          │  │
│  │  - name_embedding   │  ──── INDEXED ─────► │  Similarity Search       │  │
│  │  - fact_embedding   │                      │  Fulltext Search         │  │
│  │  - fulltext fields  │                      │  BFS Traversal           │  │
│  └─────────────────────┘                      └──────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Integration Points:

| Pipeline Stage | Search Function Used | Purpose |
|----------------|---------------------|---------|
| **Node Deduplication** | `search()` with `NODE_HYBRID_SEARCH_RRF` | Find candidate nodes to match against |
| **Edge Resolution** | `search()` with `EDGE_HYBRID_SEARCH_RRF` | Find related edges for deduplication |
| **Edge Invalidation** | `search()` with `EDGE_HYBRID_SEARCH_RRF` | Find edges that may contradict new facts |
| **Community Building** | Internal graph queries | Find nodes to cluster |

---

## 2. Data Flow Between Systems

### 2.1 Pipeline → Search (Write Path)

```
Episode Input
     │
     ▼
┌────────────────────┐
│  Extract Entities  │  (LLM)
└────────────────────┘
     │
     ▼
┌────────────────────┐     ┌─────────────────────┐
│ Generate Embedding │────►│  Stored in Database  │
│ (name_embedding)   │     │  for Search Index    │
└────────────────────┘     └─────────────────────┘
     │
     ▼
┌────────────────────┐
│  Extract Facts     │  (LLM)
└────────────────────┘
     │
     ▼
┌────────────────────┐     ┌─────────────────────┐
│ Generate Embedding │────►│  Stored in Database  │
│ (fact_embedding)   │     │  for Search Index    │
└────────────────────┘     └─────────────────────┘
     │
     ▼
┌────────────────────┐
│ Persist to Graph   │────► Fulltext indices updated
└────────────────────┘      Vector indices updated
```

### 2.2 Search → Pipeline (Read Path during Ingestion)

```
┌─────────────────────────────────────────────────────────────────┐
│                    DEDUPLICATION FLOW                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Extracted Node: "Barack Obama"                                  │
│       │                                                          │
│       ▼                                                          │
│  ┌──────────────────────────────────────────────────────┐        │
│  │  search(query="Barack Obama", config=NODE_HYBRID)    │        │
│  │                                                       │        │
│  │  Executes:                                            │        │
│  │    - node_fulltext_search() → BM25 on name/summary   │        │
│  │    - node_similarity_search() → cosine on embedding  │        │
│  │    - rrf() → combine results                          │        │
│  └──────────────────────────────────────────────────────┘        │
│       │                                                          │
│       ▼                                                          │
│  Search Results: [                                               │
│    EntityNode("Barack Obama", uuid="abc"),                       │
│    EntityNode("Obama", uuid="def"),                              │
│    EntityNode("Barack H. Obama", uuid="ghi"),                    │
│  ]                                                               │
│       │                                                          │
│       ▼                                                          │
│  ┌──────────────────────────────────────────────────────┐        │
│  │  Deterministic Deduplication (MinHash/LSH)           │        │
│  │  OR                                                   │        │
│  │  LLM Deduplication (if deterministic fails)          │        │
│  └──────────────────────────────────────────────────────┘        │
│       │                                                          │
│       ▼                                                          │
│  Decision: Merge with existing "abc" OR create new               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Node Deduplication Uses Search

### 3.1 Location & Function

```python
# File: graphiti_core/utils/maintenance/node_operations.py

async def _search_existing_nodes(
    clients: GraphitiClients,
    extracted_nodes: list[EntityNode],
    existing_nodes_override: list[EntityNode] | None,
) -> list[EntityNode]:
    """Search per extracted name and return unique candidates with overrides honored in order."""
    
    # For each extracted node, search the graph for similar existing nodes
    search_results: list[SearchResults] = await semaphore_gather(
        *[
            search(
                clients=clients,
                query=node.name,                    # Search by entity name
                group_ids=[node.group_id],          # Within same partition
                search_filter=SearchFilters(),      # No additional filters
                config=NODE_HYBRID_SEARCH_RRF,      # BM25 + Cosine + RRF
            )
            for node in extracted_nodes
        ]
    )
```

### 3.2 Search Config Used

```python
# File: graphiti_core/search/search_config_recipes.py

NODE_HYBRID_SEARCH_RRF = SearchConfig(
    node_config=NodeSearchConfig(
        search_methods=[NodeSearchMethod.bm25, NodeSearchMethod.cosine_similarity],
        reranker=NodeReranker.rrf,
    )
)
```

### 3.3 Flow Diagram

```
resolve_extracted_nodes()
     │
     ├──► _search_existing_nodes()
     │         │
     │         └──► search() × N  (one per extracted node)
     │                   │
     │                   └──► node_fulltext_search() + node_similarity_search()
     │                              │
     │                              └──► RRF reranking
     │
     ├──► _build_candidate_indexes()  (MinHash/LSH for fuzzy matching)
     │
     └──► _resolve_with_similarity()  (Deterministic matching)
                │
                └──► LLM escalation (if needed)
```

---

## 4. Edge Resolution Uses Search

### 4.1 Location & Function

```python
# File: graphiti_core/utils/maintenance/edge_operations.py

async def resolve_extracted_edges(
    clients: GraphitiClients,
    extracted_edges: list[EntityEdge],
    episode: EpisodicNode,
    ...
) -> tuple[list[EntityEdge], list[EntityEdge]]:
    
    # 1. Get edges between the same node pairs (for duplicate detection)
    valid_edges_list: list[list[EntityEdge]] = await semaphore_gather(
        *[
            EntityEdge.get_between_nodes(driver, edge.source_node_uuid, edge.target_node_uuid)
            for edge in extracted_edges
        ]
    )
    
    # 2. Search for semantically related edges
    related_edges_results: list[SearchResults] = await semaphore_gather(
        *[
            search(
                clients,
                extracted_edge.fact,                    # Search by fact content
                group_ids=[extracted_edge.group_id],
                config=EDGE_HYBRID_SEARCH_RRF,          # BM25 + Cosine + RRF
                search_filter=SearchFilters(
                    edge_uuids=[edge.uuid for edge in valid_edges]  # Limit to same node pair
                ),
            )
            for extracted_edge, valid_edges in zip(extracted_edges, valid_edges_list)
        ]
    )
    
    # 3. Search for potential invalidation candidates (broader search)
    edge_invalidation_candidate_results: list[SearchResults] = await semaphore_gather(
        *[
            search(
                clients,
                extracted_edge.fact,
                group_ids=[extracted_edge.group_id],
                config=EDGE_HYBRID_SEARCH_RRF,
                search_filter=SearchFilters(),           # No filter - find ALL similar facts
            )
            for extracted_edge in extracted_edges
        ]
    )
```

### 4.2 Two Different Search Purposes

| Search | Filter | Purpose |
|--------|--------|---------|
| **Related Edges** | `edge_uuids` limited to same node pair | Find duplicates of the same relationship |
| **Invalidation Candidates** | No filter (broad) | Find contradictory facts anywhere in graph |

### 4.3 How Results Are Used

```python
# Related edges: Used for duplicate detection in LLM prompt
related_edges_context = [
    {'idx': i, 'fact': edge.fact} 
    for i, edge in enumerate(related_edges)
]

# Invalidation candidates: Used for contradiction detection
invalidation_edge_candidates_context = [
    {'idx': i, 'fact': existing_edge.fact} 
    for i, existing_edge in enumerate(existing_edges)
]

# Prompt to LLM includes both:
context = {
    'existing_edges': related_edges_context,        # For dedup
    'new_edge': extracted_edge.fact,
    'edge_invalidation_candidates': invalidation_edge_candidates_context,  # For contradictions
    'edge_types': edge_types_context,
}
```

---

## 5. Embeddings: Pipeline Generation → Search Consumption

### 5.1 Embedding Generation (Pipeline)

```python
# File: graphiti_core/nodes.py

class EntityNode:
    async def generate_name_embedding(self, embedder: EmbedderClient):
        text = self.name.replace('\n', ' ')
        self.name_embedding = await embedder.create(input_data=[text])

# File: graphiti_core/edges.py

class EntityEdge:
    async def generate_embedding(self, embedder: EmbedderClient):
        text = self.fact.replace('\n', ' ')
        self.fact_embedding = await embedder.create(input_data=[text])
```

### 5.2 Embedding Usage (Search)

```python
# File: graphiti_core/search/search_utils.py

async def node_similarity_search(driver, search_vector, ...):
    # Query uses stored name_embedding
    query = """
        MATCH (n:Entity)
        WITH n, vector.similarity.cosine(n.name_embedding, $search_vector) AS score
        WHERE score > $min_score
        RETURN n
        ORDER BY score DESC
    """

async def edge_similarity_search(driver, search_vector, ...):
    # Query uses stored fact_embedding
    query = """
        MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)
        WITH e, vector.similarity.cosine(e.fact_embedding, $search_vector) AS score
        WHERE score > $min_score
        RETURN e
        ORDER BY score DESC
    """
```

### 5.3 Embedding Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    EMBEDDING LIFECYCLE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PIPELINE STAGE                    SEARCH STAGE                  │
│  ─────────────                     ────────────                  │
│                                                                  │
│  1. Entity Extraction              5. Query Embedding            │
│     "Barack Obama"                    "Who is Obama?"            │
│          │                                 │                     │
│          ▼                                 ▼                     │
│  2. generate_name_embedding()      6. embedder.create()          │
│     → [0.23, -0.15, ...]              → [0.21, -0.18, ...]      │
│          │                                 │                     │
│          ▼                                 │                     │
│  3. node.save()                            │                     │
│     INSERT name_embedding                  │                     │
│          │                                 │                     │
│          ▼                                 ▼                     │
│  4. Vector Index Updated           7. Cosine Similarity          │
│          │                              score = dot(q, n.emb)    │
│          │                                 │                     │
│          └─────────────────────────────────┘                     │
│                        │                                         │
│                        ▼                                         │
│               Search Results Ranked by Score                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Shared Data Models

### 6.1 Models Used by Both Systems

```python
# EntityNode - Created by pipeline, searched by search system
class EntityNode:
    uuid: str
    name: str
    name_embedding: list[float] | None  # Generated by pipeline, indexed for search
    summary: str                         # Extracted by pipeline, fulltext indexed
    labels: list[str]                    # Used for search filtering
    group_id: str                        # Partition key for both systems

# EntityEdge - Created by pipeline, searched by search system
class EntityEdge:
    uuid: str
    name: str                           # Relation type (e.g., "WORKS_AT")
    fact: str                           # Natural language fact
    fact_embedding: list[float] | None  # Generated by pipeline, indexed for search
    source_node_uuid: str               # Links to EntityNode
    target_node_uuid: str               # Links to EntityNode
    valid_at: datetime | None           # Temporal - set by pipeline, filtered in search
    invalid_at: datetime | None         # Temporal - set by pipeline, filtered in search
    expired_at: datetime | None         # Set by invalidation logic

# EpisodicNode - Input to pipeline, searchable for context
class EpisodicNode:
    uuid: str
    content: str                        # Fulltext indexed for search
    source: EpisodeType
    source_description: str
    valid_at: datetime
```

### 6.2 SearchFilters - Bridge Between Systems

```python
# Used by pipeline during deduplication and resolution
class SearchFilters:
    node_labels: list[str] | None      # Filter by entity type
    edge_types: list[str] | None       # Filter by relation type
    edge_uuids: list[str] | None       # Specific edges (used in edge resolution)
    valid_at: list[list[DateFilter]]   # Temporal filtering
    invalid_at: list[list[DateFilter]] # Temporal filtering
    created_at: list[list[DateFilter]] # Temporal filtering
    expired_at: list[list[DateFilter]] # Temporal filtering
```

---

## 7. Configuration Flow

### 7.1 GraphitiClients - Shared Across Both Systems

```python
# File: graphiti_core/graphiti_types.py

class GraphitiClients(BaseModel):
    driver: GraphDriver        # Database connection (Neo4j, FalkorDB, etc.)
    llm_client: LLMClient      # For extraction & deduplication
    embedder: EmbedderClient   # For generating embeddings
    cross_encoder: CrossEncoderClient  # For search reranking
    tracer: Tracer             # For observability
```

### 7.2 Where Clients Are Used

| Client | Pipeline Usage | Search Usage |
|--------|----------------|--------------|
| `driver` | Save nodes/edges, read existing | Execute search queries |
| `llm_client` | Extract entities, dedupe, resolve | Not used directly |
| `embedder` | Generate node/edge embeddings | Generate query embedding |
| `cross_encoder` | Not used | Rerank search results |

### 7.3 Search Config Used in Pipeline

```python
# Pipeline uses specific search configs during ingestion:

# For node deduplication:
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF

# For edge resolution:
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF
```

---

## 8. Temporal Logic Integration

### 8.1 Timeline

```
Episode Ingestion                    User Search Query
       │                                    │
       ▼                                    ▼
┌─────────────────┐               ┌─────────────────┐
│ Extract Facts   │               │ Apply Temporal  │
│ with valid_at   │               │ Filters         │
└─────────────────┘               └─────────────────┘
       │                                    │
       ▼                                    │
┌─────────────────┐                         │
│ Find Invalidation                         │
│ Candidates      │◄────────────────────────┘
└─────────────────┘     (Search returns candidates)
       │
       ▼
┌─────────────────┐
│ Resolve         │
│ Contradictions  │
│ - Set invalid_at│
│ - Set expired_at│
└─────────────────┘
       │
       ▼
┌─────────────────┐
│ Persist Updated │
│ Edge States     │
└─────────────────┘
       │
       ▼
   Future searches filter by expired_at IS NULL
```

### 8.2 resolve_edge_contradictions()

```python
# File: graphiti_core/utils/maintenance/edge_operations.py

def resolve_edge_contradictions(
    resolved_edge: EntityEdge, 
    invalidation_candidates: list[EntityEdge]  # ← Came from search!
) -> list[EntityEdge]:
    """
    For each candidate found via search:
    1. If candidate.invalid_at <= resolved_edge.valid_at: no conflict
    2. If resolved_edge.invalid_at <= candidate.valid_at: no conflict
    3. Otherwise: invalidate the older one
    """
    invalidated_edges = []
    for candidate in invalidation_candidates:
        if is_contradiction(resolved_edge, candidate):
            if candidate.valid_at < resolved_edge.valid_at:
                # Candidate is older - invalidate it
                candidate.invalid_at = resolved_edge.valid_at
                candidate.expired_at = utc_now()
                invalidated_edges.append(candidate)
    
    return invalidated_edges
```

---

## 9. Community Operations

### 9.1 Community Building Uses Graph Traversal

```python
# File: graphiti_core/utils/maintenance/community_operations.py

async def build_community(driver, embedder, llm_client, nodes):
    """
    1. Run label propagation on graph
    2. Group nodes into clusters
    3. Generate community summary via LLM
    4. Create CommunityNode with embedding
    """
    
    # Uses graph structure (edges built by pipeline)
    clusters = await get_community_clusters(driver, nodes)
    
    for cluster in clusters:
        # Generate summary via LLM
        summary = await summarize_cluster(llm_client, cluster)
        
        community = CommunityNode(
            name=cluster.name,
            summary=summary,
        )
        
        # Generate embedding (for community search)
        await community.generate_name_embedding(embedder)
        await community.save(driver)
```

### 9.2 Community Search

```python
# Communities are searchable entities:
COMMUNITY_HYBRID_SEARCH_RRF = SearchConfig(
    community_config=CommunitySearchConfig(
        search_methods=[CommunitySearchMethod.bm25, CommunitySearchMethod.cosine_similarity],
        reranker=CommunityReranker.rrf,
    )
)
```

---

## 10. Critical Dependencies

### 10.1 Pipeline Depends on Search

| Pipeline Operation | Required Search Function | What Happens Without It |
|-------------------|-------------------------|------------------------|
| Node Deduplication | `search()` with node config | Would create duplicate entities |
| Edge Resolution | `search()` with edge config | Would create duplicate/contradictory facts |
| Invalidation | `search()` for candidates | Contradictory facts would coexist |

### 10.2 Search Depends on Pipeline

| Search Operation | Required Pipeline Output | What Happens Without It |
|-----------------|-------------------------|------------------------|
| Similarity Search | `name_embedding`, `fact_embedding` | No vector search possible |
| Fulltext Search | Indexed text fields | No BM25 search possible |
| BFS Traversal | Graph edges | No graph traversal possible |

### 10.3 Shared Infrastructure

```
┌────────────────────────────────────────────────────────────────┐
│                    SHARED INFRASTRUCTURE                        │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐      ┌─────────────────┐                  │
│  │  GraphDriver    │      │   EmbedderClient │                  │
│  │  (Neo4j, etc)   │      │   (OpenAI, etc)  │                  │
│  └────────┬────────┘      └────────┬─────────┘                  │
│           │                        │                            │
│           │    ┌───────────────────┘                            │
│           │    │                                                │
│           ▼    ▼                                                │
│  ┌────────────────────────────────────────────────────┐        │
│  │               GraphitiClients                       │        │
│  │  (Bundled for passing to both pipeline & search)    │        │
│  └────────────────────────────────────────────────────┘        │
│           │                        │                            │
│           ▼                        ▼                            │
│  ┌─────────────────┐      ┌─────────────────┐                  │
│  │    PIPELINE     │      │     SEARCH      │                  │
│  │  - extract      │      │  - fulltext     │                  │
│  │  - dedupe       │      │  - similarity   │                  │
│  │  - resolve      │      │  - bfs          │                  │
│  │  - persist      │      │  - rerank       │                  │
│  └─────────────────┘      └─────────────────┘                  │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## 11. DSPy Reimplementation Implications

### 11.1 What Must Be Preserved

The integration between pipeline and search must be preserved in any DSPy reimplementation:

```python
# Current pattern that MUST continue to work:

# 1. Search returns candidates
search_results = await search(clients, node.name, config=NODE_HYBRID_SEARCH_RRF)
candidate_nodes = search_results.nodes

# 2. Pipeline uses candidates for deduplication
dedupe_result = await llm_client.generate_response(
    dedupe_prompt(extracted_node, candidate_nodes),
    response_model=NodeDuplicate,
)
```

### 11.2 DSPy Module Design Considerations

```python
# The DSPy modules should accept search results as input:

class DeduplicateEntitiesModule(dspy.Module):
    def __init__(self):
        self.predict = dspy.ChainOfThought(DeduplicateNodesSignature)
    
    def forward(self, extracted_entity: str, candidate_entities: list[str]):
        # candidate_entities comes from search system
        return self.predict(
            extracted_entity=extracted_entity,
            existing_entities=candidate_entities,  # From search!
        )

class ResolveFactsModule(dspy.Module):
    def __init__(self):
        self.predict = dspy.ChainOfThought(ResolveFactSignature)
    
    def forward(self, new_fact: str, related_facts: list[str], invalidation_candidates: list[str]):
        # related_facts and invalidation_candidates come from search!
        return self.predict(
            new_fact=new_fact,
            existing_facts=related_facts,
            contradiction_candidates=invalidation_candidates,
        )
```

### 11.3 Search Integration Points for DSPy

| DSPy Module | Search Input | Purpose |
|-------------|--------------|---------|
| `ExtractEntitiesModule` | Previous episodes (fulltext search) | Context for extraction |
| `DeduplicateNodesModule` | `search(node.name, NODE_HYBRID_SEARCH_RRF)` | Find candidates |
| `ExtractFactsModule` | Previous episodes (fulltext search) | Context for extraction |
| `ResolveFactsModule` | `search(edge.fact, EDGE_HYBRID_SEARCH_RRF)` | Find related + contradictions |

### 11.4 Embedding Generation Must Happen Before Search

```python
# IMPORTANT: Embeddings must be generated BEFORE deduplication search can find them

# Wrong order:
search(node.name)  # Can't find by similarity if embedding not stored yet!
node.save()        # Saves embedding

# Correct order (current implementation):
# During PREVIOUS episode processing:
node.generate_name_embedding()
node.save()  # Embedding now indexed

# During CURRENT episode processing:
search(node.name)  # Can now find similar nodes by embedding!
```

### 11.5 Summary: Integration Contracts

The following contracts MUST be maintained:

1. **Search before dedupe**: Pipeline calls search to get candidates before deduplication
2. **Embeddings before search**: Entities must have embeddings before they can be found by similarity search
3. **Dual search for edges**: Both "related edges" (narrow) and "invalidation candidates" (broad) searches
4. **Temporal resolution after search**: Contradiction detection depends on finding candidates via search
5. **Shared clients**: Both systems use the same `GraphitiClients` bundle

---

## Appendix A: File Cross-References

### Pipeline Files That Import Search

```python
# graphiti_core/utils/maintenance/node_operations.py
from graphiti_core.search.search import search
from graphiti_core.search.search_config import SearchResults
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF
from graphiti_core.search.search_filters import SearchFilters

# graphiti_core/utils/maintenance/edge_operations.py
from graphiti_core.search.search import search
from graphiti_core.search.search_config import SearchResults
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF
from graphiti_core.search.search_filters import SearchFilters
```

### Search Files That Use Pipeline Models

```python
# graphiti_core/search/search_utils.py
from graphiti_core.nodes import EntityNode, EpisodicNode, CommunityNode
from graphiti_core.edges import EntityEdge

# graphiti_core/search/search.py
from graphiti_core.nodes import EntityNode, EpisodicNode, CommunityNode
from graphiti_core.edges import EntityEdge
```

---

## Appendix B: Sequence Diagram

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│ Episode  │     │ Pipeline │     │  Search  │     │ Database │
└────┬─────┘     └────┬─────┘     └────┬─────┘     └────┬─────┘
     │                │                │                │
     │  add_episode   │                │                │
     │───────────────>│                │                │
     │                │                │                │
     │                │ extract_nodes (LLM)             │
     │                │──────────────────────────────>  │
     │                │                │                │
     │                │ search(name)   │                │
     │                │───────────────>│                │
     │                │                │  query nodes   │
     │                │                │───────────────>│
     │                │                │<───────────────│
     │                │  candidates    │                │
     │                │<───────────────│                │
     │                │                │                │
     │                │ dedupe_nodes (LLM, with candidates)
     │                │──────────────────────────────>  │
     │                │                │                │
     │                │ extract_edges (LLM)             │
     │                │──────────────────────────────>  │
     │                │                │                │
     │                │ search(fact)   │                │
     │                │───────────────>│                │
     │                │                │  query edges   │
     │                │                │───────────────>│
     │                │                │<───────────────│
     │                │  related +     │                │
     │                │  invalidation  │                │
     │                │<───────────────│                │
     │                │                │                │
     │                │ resolve_edges (LLM, with candidates)
     │                │──────────────────────────────>  │
     │                │                │                │
     │                │ generate_embeddings             │
     │                │──────────────────────────────>  │
     │                │                │                │
     │                │ save_nodes_edges                │
     │                │────────────────────────────────>│
     │                │                │                │
     │<───────────────│                │                │
     │   AddEpisodeResult              │                │
```

---

*Document generated for Graphiti codebase analysis. Last updated: December 2024*

