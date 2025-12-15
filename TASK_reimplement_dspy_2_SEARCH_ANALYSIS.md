# Graphiti Search System Analysis

> **Document Purpose**: Thorough analysis of the Graphiti search and retrieval system, documenting all concepts, configurations, algorithms, and implementation details for preservation and future enhancement.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Search Configuration System](#3-search-configuration-system)
4. [Search Methods](#4-search-methods)
5. [Reranking Algorithms](#5-reranking-algorithms)
6. [Filter System](#6-filter-system)
7. [Database Provider Abstraction](#7-database-provider-abstraction)
8. [Cross-Encoder Implementations](#8-cross-encoder-implementations)
9. [Embedder System](#9-embedder-system)
10. [Query Generation](#10-query-generation)
11. [Search Flow Diagrams](#11-search-flow-diagrams)
12. [Pre-built Recipe Configurations](#12-pre-built-recipe-configurations)
13. [Performance Constants](#13-performance-constants)
14. [Helper Utilities](#14-helper-utilities)
15. [Extension Points](#15-extension-points)

---

## 1. Executive Summary

Graphiti's search system is a **multi-modal hybrid retrieval pipeline** that combines:
- **Fulltext search** (BM25) for lexical matching
- **Vector similarity search** (cosine) for semantic matching
- **Breadth-first graph traversal** (BFS) for relationship discovery
- **Multiple reranking strategies** (RRF, MMR, cross-encoder, graph-based)

The system searches across four entity types:
1. **EntityNodes** - Named entities extracted from episodes
2. **EntityEdges** - Facts/relationships between entities
3. **EpisodicNodes** - Raw episode content
4. **CommunityNodes** - Aggregated community summaries

### Key Design Principles

```
┌─────────────────────────────────────────────────────────────────┐
│                    SEARCH ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────┤
│  1. Configurable: SearchConfig controls all behavior            │
│  2. Composable: Mix search methods + rerankers per entity type  │
│  3. Extensible: SearchInterface for custom backends             │
│  4. Multi-provider: Neo4j, FalkorDB, Kuzu, Neptune support      │
│  5. Async-first: semaphore_gather for bounded concurrency       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Architecture Overview

### 2.1 Component Hierarchy

```
search()                              # Main entry point
├── edge_search()                     # Search EntityEdges
│   ├── edge_fulltext_search()        # BM25 over facts
│   ├── edge_similarity_search()      # Cosine over embeddings
│   ├── edge_bfs_search()             # Graph traversal
│   └── [reranker]                    # RRF/MMR/cross-encoder/etc
├── node_search()                     # Search EntityNodes
│   ├── node_fulltext_search()        # BM25 over names/summaries
│   ├── node_similarity_search()      # Cosine over name embeddings
│   ├── node_bfs_search()             # Graph traversal
│   └── [reranker]                    # RRF/MMR/cross-encoder/etc
├── episode_search()                  # Search EpisodicNodes
│   └── episode_fulltext_search()     # BM25 over content
│   └── [reranker]                    # RRF/cross-encoder
└── community_search()                # Search CommunityNodes
    ├── community_fulltext_search()   # BM25 over names
    ├── community_similarity_search() # Cosine over name embeddings
    └── [reranker]                    # RRF/MMR/cross-encoder
```

### 2.2 File Structure

```
graphiti_core/search/
├── __init__.py                  # Public exports
├── search.py                    # Main search orchestration
├── search_config.py             # Configuration models & enums
├── search_config_recipes.py     # Pre-built SearchConfig instances
├── search_filters.py            # Filter models & query builders
├── search_helpers.py            # Result formatting utilities
└── search_utils.py              # Low-level search implementations
```

### 2.3 Key Dependencies

| Module | Purpose |
|--------|---------|
| `graphiti_core.driver.driver.GraphDriver` | Database abstraction |
| `graphiti_core.cross_encoder.client.CrossEncoderClient` | Reranking |
| `graphiti_core.embedder.client.EmbedderClient` | Query embedding |
| `graphiti_core.helpers.semaphore_gather` | Bounded concurrency |
| `graphiti_core.graph_queries` | DB-specific query generation |

---

## 3. Search Configuration System

### 3.1 Core Configuration Models

```python
# File: graphiti_core/search/search_config.py

class SearchConfig(BaseModel):
    """Master configuration for a search operation."""
    edge_config: EdgeSearchConfig | None = None      # Configure edge search
    node_config: NodeSearchConfig | None = None      # Configure node search
    episode_config: EpisodeSearchConfig | None = None  # Configure episode search
    community_config: CommunitySearchConfig | None = None  # Configure community search
    limit: int = 10                                  # Max results per entity type
    reranker_min_score: float = 0                    # Minimum reranker score threshold
```

### 3.2 Entity-Specific Configurations

Each entity type has its own configuration with:
- `search_methods`: List of methods to execute
- `reranker`: How to combine/rerank results
- `sim_min_score`: Minimum similarity score threshold
- `mmr_lambda`: Diversity parameter for MMR
- `bfs_max_depth`: Max traversal depth for BFS

```python
class EdgeSearchConfig(BaseModel):
    search_methods: list[EdgeSearchMethod]
    reranker: EdgeReranker = EdgeReranker.rrf
    sim_min_score: float = 0.6
    mmr_lambda: float = 0.5
    bfs_max_depth: int = 3

class NodeSearchConfig(BaseModel):
    search_methods: list[NodeSearchMethod]
    reranker: NodeReranker = NodeReranker.rrf
    sim_min_score: float = 0.6
    mmr_lambda: float = 0.5
    bfs_max_depth: int = 3

class EpisodeSearchConfig(BaseModel):
    search_methods: list[EpisodeSearchMethod]
    reranker: EpisodeReranker = EpisodeReranker.rrf
    sim_min_score: float = 0.6
    mmr_lambda: float = 0.5
    bfs_max_depth: int = 3

class CommunitySearchConfig(BaseModel):
    search_methods: list[CommunitySearchMethod]
    reranker: CommunityReranker = CommunityReranker.rrf
    sim_min_score: float = 0.6
    mmr_lambda: float = 0.5
    bfs_max_depth: int = 3
```

### 3.3 Search Method Enums

```python
class EdgeSearchMethod(Enum):
    cosine_similarity = 'cosine_similarity'
    bm25 = 'bm25'
    bfs = 'breadth_first_search'

class NodeSearchMethod(Enum):
    cosine_similarity = 'cosine_similarity'
    bm25 = 'bm25'
    bfs = 'breadth_first_search'

class EpisodeSearchMethod(Enum):
    bm25 = 'bm25'  # Only BM25 for episodes

class CommunitySearchMethod(Enum):
    cosine_similarity = 'cosine_similarity'
    bm25 = 'bm25'
```

### 3.4 Reranker Enums

```python
class EdgeReranker(Enum):
    rrf = 'reciprocal_rank_fusion'
    node_distance = 'node_distance'
    episode_mentions = 'episode_mentions'
    mmr = 'mmr'
    cross_encoder = 'cross_encoder'

class NodeReranker(Enum):
    rrf = 'reciprocal_rank_fusion'
    node_distance = 'node_distance'
    episode_mentions = 'episode_mentions'
    mmr = 'mmr'
    cross_encoder = 'cross_encoder'

class EpisodeReranker(Enum):
    rrf = 'reciprocal_rank_fusion'
    cross_encoder = 'cross_encoder'

class CommunityReranker(Enum):
    rrf = 'reciprocal_rank_fusion'
    mmr = 'mmr'
    cross_encoder = 'cross_encoder'
```

### 3.5 SearchResults Model

```python
class SearchResults(BaseModel):
    """Container for all search results with reranker scores."""
    edges: list[EntityEdge] = []
    edge_reranker_scores: list[float] = []
    nodes: list[EntityNode] = []
    node_reranker_scores: list[float] = []
    episodes: list[EpisodicNode] = []
    episode_reranker_scores: list[float] = []
    communities: list[CommunityNode] = []
    community_reranker_scores: list[float] = []

    @classmethod
    def merge(cls, results_list: list['SearchResults']) -> 'SearchResults':
        """Merge multiple SearchResults into one (simple concatenation)."""
        # Extends all lists from each result
```

---

## 4. Search Methods

### 4.1 BM25 Fulltext Search

**Purpose**: Lexical/keyword matching using inverted index

**Implementation Details**:

```python
async def edge_fulltext_search(
    driver: GraphDriver,
    query: str,
    search_filter: SearchFilters,
    group_ids: list[str] | None = None,
    limit: int = 10,
) -> list[EntityEdge]:
```

**Query Generation**:

```python
def fulltext_query(query: str, group_ids: list[str] | None, driver: GraphDriver) -> str:
    # Neo4j: Uses Lucene query syntax with group_id filtering
    # FalkorDB: Custom fulltext query builder
    # Kuzu: Simple query (no Lucene special chars)
    
    # Applies lucene_sanitize() to escape special characters:
    # + - && || ! ( ) { } [ ] ^ " ~ * ? : \ /
    
    # Max query length: MAX_QUERY_LENGTH = 128 words
```

**Indexed Fields by Entity Type**:

| Entity Type | Indexed Fields |
|-------------|----------------|
| EntityNode | `name`, `summary`, `group_id` |
| EntityEdge | `name`, `fact`, `group_id` |
| EpisodicNode | `content`, `source`, `source_description`, `group_id` |
| CommunityNode | `name`, `group_id` |

**Database-Specific Queries**:

```python
# Neo4j
'CALL db.index.fulltext.queryNodes("node_name_and_summary", $query, {limit: $limit})'

# FalkorDB
"CALL db.idx.fulltext.queryNodes('Entity', $query)"

# Kuzu
"CALL QUERY_FTS_INDEX('Entity', 'node_name_and_summary', $query, TOP := $limit)"
```

### 4.2 Vector Similarity Search

**Purpose**: Semantic matching using embedding cosine similarity

**Implementation**:

```python
async def edge_similarity_search(
    driver: GraphDriver,
    search_vector: list[float],
    source_node_uuid: str | None,  # Optional: filter by source
    target_node_uuid: str | None,  # Optional: filter by target
    search_filter: SearchFilters,
    group_ids: list[str] | None = None,
    limit: int = 10,
    min_score: float = 0.6,  # DEFAULT_MIN_SCORE
) -> list[EntityEdge]:
```

**Embedding Fields**:

| Entity Type | Embedding Field | Dimension |
|-------------|-----------------|-----------|
| EntityNode | `name_embedding` | `EMBEDDING_DIM` (default 1024) |
| EntityEdge | `fact_embedding` | `EMBEDDING_DIM` (default 1024) |
| CommunityNode | `name_embedding` | `EMBEDDING_DIM` (default 1024) |

**Cosine Similarity Functions by Provider**:

```python
def get_vector_cosine_func_query(vec1, vec2, provider: GraphProvider) -> str:
    # Neo4j
    return f'vector.similarity.cosine({vec1}, {vec2})'
    
    # FalkorDB (rescaled to [0,1])
    return f'(2 - vec.cosineDistance({vec1}, vecf32({vec2})))/2'
    
    # Kuzu
    return f'array_cosine_similarity({vec1}, {vec2})'
```

**Client-side Cosine Calculation (for Neptune)**:

```python
def calculate_cosine_similarity(vector1: list[float], vector2: list[float]) -> float:
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    return dot_product / (norm_vector1 * norm_vector2)
```

### 4.3 Breadth-First Graph Search (BFS)

**Purpose**: Discover related entities through graph traversal

**Implementation**:

```python
async def edge_bfs_search(
    driver: GraphDriver,
    bfs_origin_node_uuids: list[str] | None,  # Starting nodes
    bfs_max_depth: int,                        # MAX_SEARCH_DEPTH = 3
    search_filter: SearchFilters,
    group_ids: list[str] | None = None,
    limit: int = 10,
) -> list[EntityEdge]:
```

**Traversal Pattern**:

```cypher
-- Neo4j pattern
UNWIND $bfs_origin_node_uuids AS origin_uuid
MATCH path = (origin {uuid: origin_uuid})-[:RELATES_TO|MENTIONS*1..{bfs_max_depth}]->(:Entity)
UNWIND relationships(path) AS rel
MATCH (n:Entity)-[e:RELATES_TO {uuid: rel.uuid}]-(m:Entity)
RETURN DISTINCT ...
```

**BFS Behavior**:
- Can start from EntityNodes OR EpisodicNodes
- Traverses both `RELATES_TO` and `MENTIONS` edges
- Kuzu requires special handling (edges stored as intermediate nodes)
- If `bfs_origin_node_uuids` is None, uses source nodes from other search results

---

## 5. Reranking Algorithms

### 5.1 Reciprocal Rank Fusion (RRF)

**Purpose**: Combine rankings from multiple search methods

**Algorithm**:

```python
def rrf(
    results: list[list[str]],  # List of ranked UUID lists
    rank_const: int = 1,       # Smoothing constant (k)
    min_score: float = 0,
) -> tuple[list[str], list[float]]:
    scores: dict[str, float] = defaultdict(float)
    
    for result in results:
        for i, uuid in enumerate(result):
            # RRF formula: score += 1 / (rank + k)
            scores[uuid] += 1 / (i + rank_const)
    
    # Sort by total score descending
    scored_uuids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    return [uuid for uuid, score in scored_uuids if score >= min_score], \
           [score for uuid, score in scored_uuids if score >= min_score]
```

**Properties**:
- Fast O(n) computation
- No embeddings needed at rerank time
- Naturally handles different scoring scales

### 5.2 Maximal Marginal Relevance (MMR)

**Purpose**: Balance relevance with diversity

**Algorithm**:

```python
def maximal_marginal_relevance(
    query_vector: list[float],
    candidates: dict[str, list[float]],  # uuid -> embedding
    mmr_lambda: float = 0.5,             # DEFAULT_MMR_LAMBDA
    min_score: float = -2.0,
) -> tuple[list[str], list[float]]:
    
    query_array = np.array(query_vector)
    
    # Normalize all embeddings
    candidate_arrays = {uuid: normalize_l2(emb) for uuid, emb in candidates.items()}
    
    # Build similarity matrix between candidates
    similarity_matrix = np.zeros((len(uuids), len(uuids)))
    for i, uuid_1 in enumerate(uuids):
        for j, uuid_2 in enumerate(uuids[:i]):
            similarity = np.dot(candidate_arrays[uuid_1], candidate_arrays[uuid_2])
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
    
    # Calculate MMR score
    mmr_scores = {}
    for i, uuid in enumerate(uuids):
        max_sim = np.max(similarity_matrix[i, :])
        relevance = np.dot(query_array, candidate_arrays[uuid])
        # MMR = λ * relevance + (λ - 1) * max_similarity_to_selected
        mmr = mmr_lambda * relevance + (mmr_lambda - 1) * max_sim
        mmr_scores[uuid] = mmr
    
    return sorted_by_mmr_score, scores
```

**Parameters**:
- `mmr_lambda = 1.0`: Pure relevance (no diversity)
- `mmr_lambda = 0.5`: Balanced
- `mmr_lambda = 0.0`: Pure diversity

**Performance Note**: Requires loading embeddings from database

### 5.3 Cross-Encoder Reranking

**Purpose**: Use LLM/transformer for high-quality relevance scoring

**Implementation Flow**:

```python
# In edge_search():
if config.reranker == EdgeReranker.cross_encoder:
    # Build fact -> UUID mapping
    fact_to_uuid_map = {edge.fact: edge.uuid for edge in edges[:limit]}
    
    # Call cross-encoder
    reranked_facts = await cross_encoder.rank(query, list(fact_to_uuid_map.keys()))
    
    # Map back to UUIDs with score filtering
    reranked_uuids = [
        fact_to_uuid_map[fact] 
        for fact, score in reranked_facts 
        if score >= reranker_min_score
    ]
```

**Reranked Content by Entity Type**:

| Entity Type | Content Passed to Cross-Encoder |
|-------------|--------------------------------|
| EntityEdge | `edge.fact` |
| EntityNode | `node.name` |
| EpisodicNode | `episode.content` |
| CommunityNode | `community.name` |

### 5.4 Node Distance Reranking

**Purpose**: Rank by graph proximity to a center node

**Implementation**:

```python
async def node_distance_reranker(
    driver: GraphDriver,
    node_uuids: list[str],
    center_node_uuid: str,
    min_score: float = 0,
) -> tuple[list[str], list[float]]:
    
    # Query for direct connections (1-hop)
    query = """
    UNWIND $node_uuids AS node_uuid
    MATCH (center:Entity {uuid: $center_uuid})-[:RELATES_TO]-(n:Entity {uuid: node_uuid})
    RETURN 1 AS score, node_uuid AS uuid
    """
    
    results = await driver.execute_query(query, ...)
    
    # Score: 1 for connected, infinity for disconnected
    scores = {uuid: 1 if connected else float('inf') for ...}
    
    # Sort by distance (lower is better)
    sorted_uuids = sorted(uuids, key=lambda u: scores[u])
    
    # Return 1/distance as score (connected = 1.0, disconnected = 0)
    return [uuid for uuid in sorted_uuids if (1 / scores[uuid]) >= min_score], ...
```

**Requirements**: `center_node_uuid` must be provided, else raises `SearchRerankerError`

### 5.5 Episode Mentions Reranking

**Purpose**: Rank by how frequently entities are mentioned across episodes

**Implementation**:

```python
async def episode_mentions_reranker(
    driver: GraphDriver, 
    node_uuids: list[list[str]], 
    min_score: float = 0
) -> tuple[list[str], list[float]]:
    
    # Use RRF as preliminary ranking
    sorted_uuids, _ = rrf(node_uuids)
    
    # Count episode mentions
    results = await driver.execute_query("""
        UNWIND $node_uuids AS node_uuid
        MATCH (episode:Episodic)-[r:MENTIONS]->(n:Entity {uuid: node_uuid})
        RETURN count(*) AS score, n.uuid AS uuid
    """)
    
    # Sort by mention count descending
    sorted_uuids.sort(key=lambda uuid: scores[uuid], reverse=True)
    
    return filtered_by_min_score
```

---

## 6. Filter System

### 6.1 SearchFilters Model

```python
class SearchFilters(BaseModel):
    node_labels: list[str] | None = None      # Filter by node label
    edge_types: list[str] | None = None       # Filter by edge type
    valid_at: list[list[DateFilter]] | None = None     # Temporal: when true
    invalid_at: list[list[DateFilter]] | None = None   # Temporal: when ended
    created_at: list[list[DateFilter]] | None = None   # Creation time
    expired_at: list[list[DateFilter]] | None = None   # Expiration time
    edge_uuids: list[str] | None = None       # Specific edge UUIDs
    property_filters: list[PropertyFilter] | None = None  # Custom properties
```

### 6.2 Comparison Operators

```python
class ComparisonOperator(Enum):
    equals = '='
    not_equals = '<>'
    greater_than = '>'
    less_than = '<'
    greater_than_equal = '>='
    less_than_equal = '<='
    is_null = 'IS NULL'
    is_not_null = 'IS NOT NULL'
```

### 6.3 Date Filters

```python
class DateFilter(BaseModel):
    date: datetime | None
    comparison_operator: ComparisonOperator

# Usage: Filter for facts valid on a specific date
filters = SearchFilters(
    valid_at=[[
        DateFilter(date=datetime(2024, 1, 1), comparison_operator=ComparisonOperator.less_than_equal),
        DateFilter(comparison_operator=ComparisonOperator.is_not_null),
    ]],
    invalid_at=[[
        DateFilter(date=datetime(2024, 1, 1), comparison_operator=ComparisonOperator.greater_than),
    ], [
        DateFilter(comparison_operator=ComparisonOperator.is_null),  # OR still valid
    ]],
)
```

**Date Filter Logic**:
- Inner lists are AND-ed together
- Outer lists are OR-ed together
- Example: `[[A, B], [C]]` = `(A AND B) OR (C)`

### 6.4 Property Filters

```python
class PropertyFilter(BaseModel):
    property_name: str
    property_value: str | int | float | None
    comparison_operator: ComparisonOperator
```

### 6.5 Filter Query Construction

```python
def edge_search_filter_query_constructor(
    filters: SearchFilters,
    provider: GraphProvider,
) -> tuple[list[str], dict[str, Any]]:
    """Generate WHERE clauses and parameters for edge filtering."""
    
    filter_queries: list[str] = []
    filter_params: dict[str, Any] = {}
    
    # Edge types
    if filters.edge_types:
        filter_queries.append('e.name in $edge_types')
        filter_params['edge_types'] = filters.edge_types
    
    # Edge UUIDs
    if filters.edge_uuids:
        filter_queries.append('e.uuid in $edge_uuids')
        filter_params['edge_uuids'] = filters.edge_uuids
    
    # Node labels (provider-specific)
    if filters.node_labels:
        if provider == GraphProvider.KUZU:
            filter_queries.append('list_has_all(n.labels, $labels)')
        else:
            labels = '|'.join(filters.node_labels)
            filter_queries.append(f'n:{labels} AND m:{labels}')
    
    # Temporal filters (complex AND/OR logic)
    # ... date filter construction ...
    
    return filter_queries, filter_params
```

---

## 7. Database Provider Abstraction

### 7.1 Supported Providers

```python
class GraphProvider(Enum):
    NEO4J = 'neo4j'
    FALKORDB = 'falkordb'
    KUZU = 'kuzu'
    NEPTUNE = 'neptune'
```

### 7.2 Provider-Specific Behaviors

| Feature | Neo4j | FalkorDB | Kuzu | Neptune |
|---------|-------|----------|------|---------|
| Fulltext Index | Lucene | Redis FT | FTS extension | OpenSearch |
| Vector Search | Native | vecf32 | array_cosine | Client-side |
| Edge as Node | No | No | Yes (RelatesToNode_) | No |
| BFS Pattern | Standard | Standard | Double depth | Standard |

### 7.3 Kuzu Special Handling

Kuzu stores edges as intermediate nodes, requiring adjusted queries:

```python
# Standard edge match
MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)

# Kuzu edge match (edges are nodes)
MATCH (n:Entity)-[:RELATES_TO]->(e:RelatesToNode_)-[:RELATES_TO]->(m:Entity)
```

**BFS Depth Adjustment**:
```python
if driver.provider == GraphProvider.KUZU:
    # Kuzu needs 2x depth because edges are nodes
    depth = bfs_max_depth * 2 - 1
```

### 7.4 SearchInterface Extension

```python
class SearchInterface(BaseModel):
    """Interface for custom search implementations."""
    
    async def edge_fulltext_search(self, driver, query, search_filter, group_ids, limit):
        raise NotImplementedError
    
    async def edge_similarity_search(self, driver, search_vector, source_node_uuid, 
                                      target_node_uuid, search_filter, group_ids, 
                                      limit, min_score):
        raise NotImplementedError
    
    async def node_fulltext_search(self, driver, query, search_filter, group_ids, limit):
        raise NotImplementedError
    
    async def node_similarity_search(self, driver, search_vector, search_filter, 
                                      group_ids, limit, min_score):
        raise NotImplementedError
    
    async def episode_fulltext_search(self, driver, query, search_filter, group_ids, limit):
        raise NotImplementedError
```

**Usage**: If `driver.search_interface` is set, it's called instead of default implementations.

---

## 8. Cross-Encoder Implementations

### 8.1 Abstract Interface

```python
class CrossEncoderClient(ABC):
    @abstractmethod
    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        """
        Rank passages by relevance to query.
        Returns: List of (passage, score) tuples, sorted descending by score.
        """
        pass
```

### 8.2 OpenAI Reranker

**Approach**: Binary classification with logprobs

```python
class OpenAIRerankerClient(CrossEncoderClient):
    DEFAULT_MODEL = 'gpt-4.1-nano'
    
    async def rank(self, query: str, passages: list[str]):
        # Concurrent API calls for each passage
        responses = await semaphore_gather(*[
            self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "Determine if passage is relevant to query"},
                    {"role": "user", "content": f"""
                        Respond with "True" if PASSAGE is relevant to QUERY and "False" otherwise.
                        <PASSAGE>{passage}</PASSAGE>
                        <QUERY>{query}</QUERY>
                    """},
                ],
                temperature=0,
                max_tokens=1,
                logit_bias={'6432': 1, '7983': 1},  # Bias for True/False tokens
                logprobs=True,
                top_logprobs=2,
            )
            for passage in passages
        ])
        
        # Extract probability from logprobs
        for top_logprobs in responses:
            norm_logprobs = np.exp(top_logprobs[0].logprob)
            if top_logprobs[0].token.lower() == 'true':
                score = norm_logprobs
            else:
                score = 1 - norm_logprobs
```

**Characteristics**:
- Fast (single token output)
- Uses log-probability for score
- O(n) API calls per rerank

### 8.3 Gemini Reranker

**Approach**: Direct 0-100 scoring

```python
class GeminiRerankerClient(CrossEncoderClient):
    DEFAULT_MODEL = 'gemini-2.5-flash-lite'
    
    async def rank(self, query: str, passages: list[str]):
        # Generate scoring prompt for each passage
        prompts = [f"""
            Rate how well this passage answers or relates to the query. 
            Use a scale from 0 to 100.
            Query: {query}
            Passage: {passage}
            Provide only a number between 0 and 100:
        """ for passage in passages]
        
        # Concurrent scoring
        responses = await semaphore_gather(*[
            self.client.aio.models.generate_content(
                model=self.config.model,
                contents=prompt,
                config=GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=3,
                ),
            )
            for prompt in prompts
        ])
        
        # Extract numeric score, normalize to [0, 1]
        for response in responses:
            score_match = re.search(r'\b(\d{1,3})\b', response.text)
            normalized_score = float(score_match.group(1)) / 100.0
```

**Characteristics**:
- No logprob support in Gemini API
- Direct numeric scoring
- O(n) API calls per rerank

### 8.4 BGE Reranker

**Approach**: Local transformer model

```python
class BGERerankerClient(CrossEncoderClient):
    def __init__(self):
        self.model = CrossEncoder('BAAI/bge-reranker-v2-m3')
    
    async def rank(self, query: str, passages: list[str]):
        input_pairs = [[query, passage] for passage in passages]
        
        # Run in executor (sync model)
        loop = asyncio.get_running_loop()
        scores = await loop.run_in_executor(None, self.model.predict, input_pairs)
        
        return sorted(
            [(passage, float(score)) for passage, score in zip(passages, scores)],
            key=lambda x: x[1],
            reverse=True,
        )
```

**Characteristics**:
- Runs locally (no API calls)
- Requires `sentence-transformers` package
- Single batch inference

---

## 9. Embedder System

### 9.1 Abstract Interface

```python
class EmbedderClient(ABC):
    @abstractmethod
    async def create(
        self, 
        input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        """Create embedding for single input."""
        pass
    
    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """Create embeddings for batch (optional, default not implemented)."""
        raise NotImplementedError()
```

### 9.2 Configuration

```python
EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', 1024))

class EmbedderConfig(BaseModel):
    embedding_dim: int = Field(default=EMBEDDING_DIM, frozen=True)
```

### 9.3 Query Embedding in Search

```python
# In search():
if needs_embedding:  # If cosine_similarity or MMR
    search_vector = (
        query_vector                          # If provided
        if query_vector is not None
        else await embedder.create(           # Else generate
            input_data=[query.replace('\n', ' ')]
        )
    )
else:
    search_vector = [0.0] * EMBEDDING_DIM     # Placeholder
```

---

## 10. Query Generation

### 10.1 Fulltext Index Queries

```python
def get_nodes_query(name: str, query: str, limit: int, provider: GraphProvider) -> str:
    if provider == GraphProvider.FALKORDB:
        label = NEO4J_TO_FALKORDB_MAPPING[name]
        return f"CALL db.idx.fulltext.queryNodes('{label}', {query})"
    
    if provider == GraphProvider.KUZU:
        label = INDEX_TO_LABEL_KUZU_MAPPING[name]
        return f"CALL QUERY_FTS_INDEX('{label}', '{name}', {query}, TOP := $limit)"
    
    # Neo4j
    return f'CALL db.index.fulltext.queryNodes("{name}", {query}, {{limit: $limit}})'

def get_relationships_query(name: str, limit: int, provider: GraphProvider) -> str:
    if provider == GraphProvider.FALKORDB:
        label = NEO4J_TO_FALKORDB_MAPPING[name]
        return f"CALL db.idx.fulltext.queryRelationships('{label}', $query)"
    
    if provider == GraphProvider.KUZU:
        label = INDEX_TO_LABEL_KUZU_MAPPING[name]
        return f"CALL QUERY_FTS_INDEX('{label}', '{name}', cast($query AS STRING), TOP := $limit)"
    
    # Neo4j
    return f'CALL db.index.fulltext.queryRelationships("{name}", $query, {{limit: $limit}})'
```

### 10.2 Index Name Mappings

```python
NEO4J_TO_FALKORDB_MAPPING = {
    'node_name_and_summary': 'Entity',
    'community_name': 'Community',
    'episode_content': 'Episodic',
    'edge_name_and_fact': 'RELATES_TO',
}

INDEX_TO_LABEL_KUZU_MAPPING = {
    'node_name_and_summary': 'Entity',
    'community_name': 'Community',
    'episode_content': 'Episodic',
    'edge_name_and_fact': 'RelatesToNode_',
}
```

### 10.3 Lucene Query Sanitization

```python
def lucene_sanitize(query: str) -> str:
    """Escape Lucene special characters."""
    escape_map = str.maketrans({
        '+': r'\+', '-': r'\-', '&': r'\&', '|': r'\|',
        '!': r'\!', '(': r'\(', ')': r'\)', '{': r'\{',
        '}': r'\}', '[': r'\[', ']': r'\]', '^': r'\^',
        '"': r'\"', '~': r'\~', '*': r'\*', '?': r'\?',
        ':': r'\:', '\\': r'\\', '/': r'\/',
        'O': r'\O', 'R': r'\R', 'N': r'\N', 'T': r'\T',
        'A': r'\A', 'D': r'\D',  # Reserved words
    })
    return query.translate(escape_map)
```

---

## 11. Search Flow Diagrams

### 11.1 Main Search Flow

```
search(query, group_ids, config, search_filter, ...)
│
├─ Create query embedding (if needed)
│   └─ embedder.create(query)
│
└─ semaphore_gather (parallel):
    ├─ edge_search(...)
    ├─ node_search(...)
    ├─ episode_search(...)
    └─ community_search(...)
        │
        └─ Returns SearchResults(
             edges, edge_reranker_scores,
             nodes, node_reranker_scores,
             episodes, episode_reranker_scores,
             communities, community_reranker_scores
           )
```

### 11.2 Entity Search Flow (edge_search example)

```
edge_search(query, query_vector, group_ids, config, ...)
│
├─ Build search tasks based on config.search_methods:
│   ├─ EdgeSearchMethod.bm25 → edge_fulltext_search()
│   ├─ EdgeSearchMethod.cosine_similarity → edge_similarity_search()
│   └─ EdgeSearchMethod.bfs → edge_bfs_search()
│
├─ semaphore_gather(*search_tasks)
│   └─ Returns: list[list[EntityEdge]]
│
├─ Build edge_uuid_map (deduplicate)
│
├─ Apply reranker:
│   ├─ EdgeReranker.rrf → rrf(search_result_uuids)
│   ├─ EdgeReranker.mmr → maximal_marginal_relevance(query_vector, embeddings)
│   ├─ EdgeReranker.cross_encoder → cross_encoder.rank(query, facts)
│   ├─ EdgeReranker.node_distance → node_distance_reranker(driver, uuids, center)
│   └─ EdgeReranker.episode_mentions → sort by len(edge.episodes)
│
└─ Return (reranked_edges[:limit], scores[:limit])
```

### 11.3 Hybrid Search Example

```
hybrid_node_search(queries, embeddings, driver, search_filter, group_ids, limit)
│
├─ semaphore_gather:
│   ├─ [node_fulltext_search(q, ...) for q in queries]
│   └─ [node_similarity_search(e, ...) for e in embeddings]
│
├─ Build node_uuid_map from all results
│
├─ rrf(result_uuids) → ranked_uuids
│
└─ Return [node_uuid_map[uuid] for uuid in ranked_uuids]
```

---

## 12. Pre-built Recipe Configurations

### 12.1 Combined Hybrid Searches

```python
# RRF reranking across all entity types
COMBINED_HYBRID_SEARCH_RRF = SearchConfig(
    edge_config=EdgeSearchConfig(
        search_methods=[EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
        reranker=EdgeReranker.rrf,
    ),
    node_config=NodeSearchConfig(
        search_methods=[NodeSearchMethod.bm25, NodeSearchMethod.cosine_similarity],
        reranker=NodeReranker.rrf,
    ),
    episode_config=EpisodeSearchConfig(
        search_methods=[EpisodeSearchMethod.bm25],
        reranker=EpisodeReranker.rrf,
    ),
    community_config=CommunitySearchConfig(
        search_methods=[CommunitySearchMethod.bm25, CommunitySearchMethod.cosine_similarity],
        reranker=CommunityReranker.rrf,
    ),
)

# MMR reranking for diversity
COMBINED_HYBRID_SEARCH_MMR = SearchConfig(
    edge_config=EdgeSearchConfig(
        search_methods=[EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
        reranker=EdgeReranker.mmr,
        mmr_lambda=1,  # Pure relevance
    ),
    node_config=NodeSearchConfig(
        search_methods=[NodeSearchMethod.bm25, NodeSearchMethod.cosine_similarity],
        reranker=NodeReranker.mmr,
        mmr_lambda=1,
    ),
    # ... similar for episode and community
)

# Cross-encoder with BFS for maximum quality
COMBINED_HYBRID_SEARCH_CROSS_ENCODER = SearchConfig(
    edge_config=EdgeSearchConfig(
        search_methods=[
            EdgeSearchMethod.bm25,
            EdgeSearchMethod.cosine_similarity,
            EdgeSearchMethod.bfs,
        ],
        reranker=EdgeReranker.cross_encoder,
    ),
    node_config=NodeSearchConfig(
        search_methods=[
            NodeSearchMethod.bm25,
            NodeSearchMethod.cosine_similarity,
            NodeSearchMethod.bfs,
        ],
        reranker=NodeReranker.cross_encoder,
    ),
    # ... similar for episode and community
)
```

### 12.2 Single-Entity Searches

```python
# Edge-only searches
EDGE_HYBRID_SEARCH_RRF = SearchConfig(
    edge_config=EdgeSearchConfig(
        search_methods=[EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
        reranker=EdgeReranker.rrf,
    )
)

EDGE_HYBRID_SEARCH_NODE_DISTANCE = SearchConfig(
    edge_config=EdgeSearchConfig(
        search_methods=[EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
        reranker=EdgeReranker.node_distance,
    ),
)

EDGE_HYBRID_SEARCH_CROSS_ENCODER = SearchConfig(
    edge_config=EdgeSearchConfig(
        search_methods=[
            EdgeSearchMethod.bm25,
            EdgeSearchMethod.cosine_similarity,
            EdgeSearchMethod.bfs,
        ],
        reranker=EdgeReranker.cross_encoder,
    ),
    limit=10,
)

# Node-only searches
NODE_HYBRID_SEARCH_RRF = SearchConfig(
    node_config=NodeSearchConfig(
        search_methods=[NodeSearchMethod.bm25, NodeSearchMethod.cosine_similarity],
        reranker=NodeReranker.rrf,
    )
)

# Community-only searches
COMMUNITY_HYBRID_SEARCH_RRF = SearchConfig(
    community_config=CommunitySearchConfig(
        search_methods=[CommunitySearchMethod.bm25, CommunitySearchMethod.cosine_similarity],
        reranker=CommunityReranker.rrf,
    )
)

COMMUNITY_HYBRID_SEARCH_CROSS_ENCODER = SearchConfig(
    community_config=CommunitySearchConfig(
        search_methods=[CommunitySearchMethod.bm25, CommunitySearchMethod.cosine_similarity],
        reranker=CommunityReranker.cross_encoder,
    ),
    limit=3,  # Communities are typically fewer, more coarse
)
```

---

## 13. Performance Constants

### 13.1 Default Values

```python
# search_utils.py
RELEVANT_SCHEMA_LIMIT = 10      # Default result limit per search method
DEFAULT_MIN_SCORE = 0.6         # Minimum similarity score threshold
DEFAULT_MMR_LAMBDA = 0.5        # Default MMR diversity parameter
MAX_SEARCH_DEPTH = 3            # Maximum BFS traversal depth
MAX_QUERY_LENGTH = 128          # Max words in fulltext query

# search_config.py
DEFAULT_SEARCH_LIMIT = 10       # Default limit in SearchConfig

# helpers.py
SEMAPHORE_LIMIT = int(os.getenv('SEMAPHORE_LIMIT', 20))  # Max concurrent tasks
```

### 13.2 Over-fetching Strategy

Search methods fetch `2 * limit` results, then reranker reduces to `limit`:

```python
# In edge_search():
search_tasks.append(
    edge_fulltext_search(driver, query, search_filter, group_ids, 2 * limit)
)
search_tasks.append(
    edge_similarity_search(driver, query_vector, ..., 2 * limit, ...)
)
# After reranking:
return reranked_edges[:limit], edge_scores[:limit]
```

---

## 14. Helper Utilities

### 14.1 Semaphore Gather

```python
async def semaphore_gather(
    *coroutines: Coroutine,
    max_coroutines: int | None = None,
) -> list[Any]:
    """Bounded concurrent execution of coroutines."""
    semaphore = asyncio.Semaphore(max_coroutines or SEMAPHORE_LIMIT)
    
    async def _wrap_coroutine(coroutine):
        async with semaphore:
            return await coroutine
    
    return await asyncio.gather(*(_wrap_coroutine(c) for c in coroutines))
```

### 14.2 L2 Normalization

```python
def normalize_l2(embedding: list[float]) -> NDArray:
    """Normalize embedding to unit length for cosine similarity."""
    embedding_array = np.array(embedding)
    norm = np.linalg.norm(embedding_array, 2, axis=0, keepdims=True)
    return np.where(norm == 0, embedding_array, embedding_array / norm)
```

### 14.3 Search Results to Context String

```python
def search_results_to_context_string(search_results: SearchResults) -> str:
    """Format SearchResults as LLM context."""
    return f"""
    FACTS and ENTITIES represent relevant context to the current conversation.
    COMMUNITIES represent a cluster of closely related entities.

    <FACTS>
        {to_prompt_json([{
            'fact': edge.fact,
            'valid_at': str(edge.valid_at),
            'invalid_at': str(edge.invalid_at or 'Present'),
        } for edge in search_results.edges])}
    </FACTS>
    <ENTITIES>
        {to_prompt_json([{
            'entity_name': node.name, 
            'summary': node.summary
        } for node in search_results.nodes])}
    </ENTITIES>
    <EPISODES>
        {to_prompt_json([{
            'source_description': episode.source_description,
            'content': episode.content,
        } for episode in search_results.episodes])}
    </EPISODES>
    <COMMUNITIES>
        {to_prompt_json([{
            'community_name': community.name, 
            'summary': community.summary
        } for community in search_results.communities])}
    </COMMUNITIES>
    """
```

---

## 15. Extension Points

### 15.1 Custom Search Implementation

Implement `SearchInterface` for custom search backends:

```python
from graphiti_core.driver.search_interface.search_interface import SearchInterface

class MyCustomSearch(SearchInterface):
    async def edge_fulltext_search(self, driver, query, search_filter, group_ids, limit):
        # Custom implementation (e.g., external search service)
        return await my_external_search_api.search_edges(query, limit)
    
    async def node_similarity_search(self, driver, search_vector, search_filter, 
                                      group_ids, limit, min_score):
        # Custom vector search (e.g., Pinecone, Weaviate)
        return await my_vector_db.search(search_vector, limit)

# Attach to driver
driver.search_interface = MyCustomSearch()
```

### 15.2 Custom Cross-Encoder

```python
from graphiti_core.cross_encoder.client import CrossEncoderClient

class MyReranker(CrossEncoderClient):
    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        # Custom reranking logic
        scores = await my_rerank_api.score(query, passages)
        return sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
```

### 15.3 Custom SearchConfig

```python
# Create domain-specific search config
CUSTOMER_SUPPORT_SEARCH = SearchConfig(
    edge_config=EdgeSearchConfig(
        search_methods=[EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
        reranker=EdgeReranker.cross_encoder,
        sim_min_score=0.7,  # Higher threshold for precision
    ),
    node_config=NodeSearchConfig(
        search_methods=[NodeSearchMethod.bm25],  # Skip similarity for speed
        reranker=NodeReranker.rrf,
    ),
    limit=5,
    reranker_min_score=0.3,
)
```

---

## Appendix A: Database Index Definitions

### A.1 Range Indexes

```python
# Neo4j
CREATE INDEX entity_uuid IF NOT EXISTS FOR (n:Entity) ON (n.uuid)
CREATE INDEX entity_group_id IF NOT EXISTS FOR (n:Entity) ON (n.group_id)
CREATE INDEX name_entity_index IF NOT EXISTS FOR (n:Entity) ON (n.name)
CREATE INDEX created_at_entity_index IF NOT EXISTS FOR (n:Entity) ON (n.created_at)
# ... similar for Episodic, Community, edges

# FalkorDB (composite indexes)
CREATE INDEX FOR (n:Entity) ON (n.uuid, n.group_id, n.name, n.created_at)
CREATE INDEX FOR ()-[e:RELATES_TO]-() ON (e.uuid, e.group_id, e.name, e.created_at, e.expired_at, e.valid_at, e.invalid_at)
```

### A.2 Fulltext Indexes

```python
# Neo4j
CREATE FULLTEXT INDEX node_name_and_summary IF NOT EXISTS
FOR (n:Entity) ON EACH [n.name, n.summary, n.group_id]

CREATE FULLTEXT INDEX edge_name_and_fact IF NOT EXISTS
FOR ()-[e:RELATES_TO]-() ON EACH [e.name, e.fact, e.group_id]

# FalkorDB
CALL db.idx.fulltext.createNodeIndex({label: 'Entity', stopwords: [...]}, 'name', 'summary', 'group_id')

# Kuzu
CALL CREATE_FTS_INDEX('Entity', 'node_name_and_summary', ['name', 'summary']);
```

---

## Appendix B: Quick Reference

### Search Method → Best Use Case

| Method | Best For |
|--------|----------|
| BM25 | Exact keyword matches, proper nouns |
| Cosine Similarity | Semantic meaning, paraphrases |
| BFS | Related entities, graph exploration |

### Reranker → Best Use Case

| Reranker | Best For | Trade-off |
|----------|----------|-----------|
| RRF | Fast fusion of multiple methods | No query awareness |
| MMR | Diverse results | Requires embeddings |
| Cross-Encoder | Highest accuracy | O(n) API calls |
| Node Distance | Graph proximity | Requires center node |
| Episode Mentions | Frequency importance | Simple heuristic |

### Provider Compatibility Matrix

| Feature | Neo4j | FalkorDB | Kuzu | Neptune |
|---------|:-----:|:--------:|:----:|:-------:|
| Fulltext Search | ✅ | ✅ | ✅ | ✅ (OpenSearch) |
| Vector Search | ✅ | ✅ | ✅ | ⚠️ (Client-side) |
| BFS | ✅ | ✅ | ⚠️ (2x depth) | ✅ |
| Temporal Filters | ✅ | ✅ | ✅ | ✅ |
| Cross-Encoder | ✅ | ✅ | ✅ | ✅ |

---

## Appendix C: Public Search API

### C.1 Graphiti.search() - Simple Search

```python
# File: graphiti_core/graphiti.py

@handle_multiple_group_ids
async def search(
    self,
    query: str,
    center_node_uuid: str | None = None,
    group_ids: list[str] | None = None,
    num_results: int = DEFAULT_SEARCH_LIMIT,
    search_filter: SearchFilters | None = None,
    driver: GraphDriver | None = None,
) -> list[EntityEdge]:
    """
    Basic out-of-the-box search returning edges only.
    
    - If center_node_uuid is provided: Uses EDGE_HYBRID_SEARCH_NODE_DISTANCE
    - Otherwise: Uses EDGE_HYBRID_SEARCH_RRF
    """
    search_config = (
        EDGE_HYBRID_SEARCH_RRF if center_node_uuid is None 
        else EDGE_HYBRID_SEARCH_NODE_DISTANCE
    )
    search_config.limit = num_results
    
    edges = (await search(self.clients, query, group_ids, search_config, ...)).edges
    return edges
```

### C.2 Graphiti.search_() - Advanced Search

```python
@handle_multiple_group_ids
async def search_(
    self,
    query: str,
    config: SearchConfig = COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
    group_ids: list[str] | None = None,
    center_node_uuid: str | None = None,
    bfs_origin_node_uuids: list[str] | None = None,
    search_filter: SearchFilters | None = None,
    driver: GraphDriver | None = None,
) -> SearchResults:
    """
    Advanced search returning full SearchResults (nodes, edges, episodes, communities).
    
    - Full control over search configuration
    - Access to all entity types
    - Custom filters and rerankers
    """
    return await search(
        self.clients, query, group_ids, config, 
        search_filter or SearchFilters(),
        center_node_uuid=center_node_uuid,
        bfs_origin_node_uuids=bfs_origin_node_uuids,
        driver=driver,
    )
```

---

## Appendix D: Internal Pipeline Search Functions

These functions are used internally during the episode-to-graph pipeline for deduplication and resolution.

### D.1 get_relevant_nodes()

```python
async def get_relevant_nodes(
    driver: GraphDriver,
    nodes: list[EntityNode],
    search_filter: SearchFilters,
    min_score: float = 0.6,
    limit: int = 10,
) -> list[list[EntityNode]]:
    """
    For each node, find similar existing nodes in the database.
    Used for node deduplication during ingestion.
    
    Returns: List of candidate matches per input node.
    
    Combines:
    - Vector similarity on name_embedding
    - Fulltext search on name
    - Deduplicates and limits results
    """
```

### D.2 get_relevant_edges()

```python
async def get_relevant_edges(
    driver: GraphDriver,
    edges: list[EntityEdge],
    search_filter: SearchFilters,
    min_score: float = 0.6,
    limit: int = 10,
) -> list[list[EntityEdge]]:
    """
    For each edge, find similar existing edges between the same node pair.
    Used for edge deduplication during ingestion.
    
    Returns: List of candidate matches per input edge.
    
    Uses:
    - Matches on source_node_uuid and target_node_uuid
    - Vector similarity on fact_embedding
    """
```

### D.3 get_edge_invalidation_candidates()

```python
async def get_edge_invalidation_candidates(
    driver: GraphDriver,
    edges: list[EntityEdge],
    search_filter: SearchFilters,
    min_score: float = 0.6,
    limit: int = 10,
) -> list[list[EntityEdge]]:
    """
    Find existing edges that might be invalidated by new edges.
    Used for temporal contradiction detection.
    
    Returns: List of potential invalidation candidates per input edge.
    
    Matches edges where either:
    - source_node_uuid matches source OR target of new edge
    - target_node_uuid matches source OR target of new edge
    
    Then filters by fact_embedding similarity.
    """
```

---

## Appendix E: GraphOperationsInterface

The `GraphOperationsInterface` provides hooks for custom graph mutation and embedding operations. When set on the driver, it overrides default database operations.

### E.1 Interface Definition

```python
# File: graphiti_core/driver/graph_operations/graph_operations.py

class GraphOperationsInterface(BaseModel):
    """Interface for updating graph mutation behavior."""
    
    # Node operations
    async def node_save(self, node: Any, driver: Any) -> None: ...
    async def node_delete(self, node: Any, driver: Any) -> None: ...
    async def node_save_bulk(self, _cls, driver, transaction, nodes, batch_size=100) -> None: ...
    async def node_delete_by_group_id(self, _cls, driver, group_id, batch_size=100) -> None: ...
    async def node_delete_by_uuids(self, _cls, driver, uuids, group_id=None, batch_size=100) -> None: ...
    
    # Embedding operations (used in search)
    async def node_load_embeddings(self, node: Any, driver: Any) -> None: ...
    async def node_load_embeddings_bulk(self, driver, nodes, batch_size=100) -> dict[str, list[float]]: ...
    
    # Episodic node operations
    async def episodic_node_save(self, node: Any, driver: Any) -> None: ...
    async def episodic_node_delete(self, node: Any, driver: Any) -> None: ...
    async def episodic_node_save_bulk(self, _cls, driver, transaction, nodes, batch_size=100) -> None: ...
    async def episodic_edge_save_bulk(self, _cls, driver, transaction, episodic_edges, batch_size=100) -> None: ...
    
    # Edge operations
    async def edge_save(self, edge: Any, driver: Any) -> None: ...
    async def edge_delete(self, edge: Any, driver: Any) -> None: ...
    async def edge_save_bulk(self, _cls, driver, transaction, edges, batch_size=100) -> None: ...
    async def edge_delete_by_uuids(self, _cls, driver, uuids) -> None: ...
    async def edge_load_embeddings(self, edge: Any, driver: Any) -> None: ...
    async def edge_load_embeddings_bulk(self, driver, edges, batch_size=100) -> dict[str, list[float]]: ...
```

### E.2 Search Integration

The `GraphOperationsInterface` is checked in search utilities:

```python
# In search_utils.py

async def get_embeddings_for_nodes(driver, nodes):
    if driver.graph_operations_interface:
        # Use custom implementation
        return await driver.graph_operations_interface.node_load_embeddings_bulk(driver, nodes)
    # ... default implementation

async def get_embeddings_for_edges(driver, edges):
    if driver.graph_operations_interface:
        return await driver.graph_operations_interface.edge_load_embeddings_bulk(driver, edges)
    # ... default implementation
```

---

## Appendix F: FalkorDB-Specific Details

### F.1 STOPWORDS List

FalkorDB uses RedisSearch syntax and filters stopwords from queries:

```python
# File: graphiti_core/driver/falkordb_driver.py

STOPWORDS = [
    'a', 'is', 'the', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by',
    'for', 'if', 'in', 'into', 'it', 'no', 'not', 'of', 'on', 'or', 'such',
    'that', 'their', 'then', 'there', 'these', 'they', 'this', 'to', 'was',
    'will', 'with',
]
```

### F.2 Fulltext Query Building

```python
def build_fulltext_query(self, query: str, group_ids: list[str] | None, max_query_length: int = 128) -> str:
    """
    Build FalkorDB-specific fulltext query using RedisSearch syntax.
    
    - Field queries use @ prefix: @field:value
    - Multiple values: (@group_id:value1|value2)
    - AND is implicit with space
    - OR uses pipe within parentheses
    """
    # Build group filter
    if group_ids:
        group_values = '|'.join(group_ids)
        group_filter = f'(@group_id:{group_values})'
    else:
        group_filter = ''
    
    # Sanitize and filter stopwords
    sanitized_query = self.sanitize(query)
    query_words = sanitized_query.split()
    filtered_words = [w for w in query_words if w.lower() not in STOPWORDS]
    sanitized_query = ' | '.join(filtered_words)
    
    # Combine
    return group_filter + ' (' + sanitized_query + ')'
```

### F.3 Default Group ID

FalkorDB requires an escaped underscore as the default group ID:

```python
class FalkorDriver(GraphDriver):
    default_group_id: str = '\\_'
    fulltext_syntax: str = '@'
```

---

## Appendix G: Environment Variables

### G.1 Index Names (Neptune/OpenSearch)

```python
# File: graphiti_core/driver/driver.py

ENTITY_INDEX_NAME = os.environ.get('ENTITY_INDEX_NAME', 'entities')
EPISODE_INDEX_NAME = os.environ.get('EPISODE_INDEX_NAME', 'episodes')
COMMUNITY_INDEX_NAME = os.environ.get('COMMUNITY_INDEX_NAME', 'communities')
ENTITY_EDGE_INDEX_NAME = os.environ.get('ENTITY_EDGE_INDEX_NAME', 'entity_edges')
```

### G.2 Search-Related Environment Variables

```python
# File: graphiti_core/helpers.py

SEMAPHORE_LIMIT = int(os.getenv('SEMAPHORE_LIMIT', 20))
MAX_REFLEXION_ITERATIONS = int(os.getenv('MAX_REFLEXION_ITERATIONS', 0))

# File: graphiti_core/embedder/client.py

EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', 1024))
```

---

## Appendix H: Error Handling & Decorators

### H.1 SearchRerankerError

```python
# File: graphiti_core/errors.py

class SearchRerankerError(GraphitiError):
    """Raised when reranker requirements are not met."""
    
    def __init__(self, text: str):
        self.message = text
        super().__init__(self.message)

# Usage: Raised when node_distance reranker used without center_node_uuid
if config.reranker == EdgeReranker.node_distance:
    if center_node_uuid is None:
        raise SearchRerankerError('No center node provided for Node Distance reranker')
```

### H.2 @handle_multiple_group_ids Decorator

FalkorDB stores each group_id as a separate database. This decorator handles multi-group searches:

```python
# File: graphiti_core/decorators.py

def handle_multiple_group_ids(func: F) -> F:
    """
    Decorator for FalkorDB methods that need to handle multiple group_ids.
    Runs the function for each group_id separately and merges results.
    """
    async def wrapper(self, *args, **kwargs):
        # Only activate for FalkorDB with multiple group_ids
        if (
            self.clients.driver.provider == GraphProvider.FALKORDB
            and group_ids
            and len(group_ids) > 1
        ):
            # Execute for each group_id concurrently
            results = await semaphore_gather(*[
                execute_for_group(gid) for gid in group_ids
            ])
            
            # Merge results based on type
            if isinstance(results[0], SearchResults):
                return SearchResults.merge(results)
            elif isinstance(results[0], list):
                return [item for result in results for item in result]
            elif isinstance(results[0], tuple):
                # Handle tuple outputs (like build_communities returning (nodes, edges))
                return merge_tuples(results)
            else:
                return results
        
        # Normal execution for other providers
        return await func(self, *args, **kwargs)
```

**Applied to**:
- `Graphiti.search()`
- `Graphiti.search_()`
- `Graphiti.retrieve_episodes()`
- `Graphiti.build_communities()`

---

## Appendix I: hybrid_node_search Utility

A convenience function for quick hybrid node searches:

```python
async def hybrid_node_search(
    queries: list[str],
    embeddings: list[list[float]],
    driver: GraphDriver,
    search_filter: SearchFilters,
    group_ids: list[str] | None = None,
    limit: int = 10,
) -> list[EntityNode]:
    """
    Perform hybrid search combining multiple queries and embeddings.
    
    1. Runs fulltext search for each query
    2. Runs similarity search for each embedding
    3. Combines with RRF reranking
    
    Returns: Deduplicated, ranked list of EntityNodes
    """
    results: list[list[EntityNode]] = await semaphore_gather(
        *[node_fulltext_search(driver, q, search_filter, group_ids, 2 * limit) for q in queries],
        *[node_similarity_search(driver, e, search_filter, group_ids, 2 * limit) for e in embeddings],
    )
    
    node_uuid_map = {node.uuid: node for result in results for node in result}
    result_uuids = [[node.uuid for node in result] for result in results]
    
    ranked_uuids, _ = rrf(result_uuids)
    return [node_uuid_map[uuid] for uuid in ranked_uuids]
```

---

*Document generated for Graphiti codebase analysis. Last updated: December 2024*

