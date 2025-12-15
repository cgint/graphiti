# Graphiti Episode-to-Graph Pipeline: Research & DSPy Reimplementation Plan

## Executive Summary

This document provides a thorough analysis of Graphiti's core episode-to-graph pipeline and a detailed plan for reimplementing it using DSPy. The goal is to leverage DSPy's modular, optimizable approach to create a more stable, maintainable, and performant knowledge graph construction system.

---

## Part 1: Graphiti Pipeline Deep Dive

### 1.1 High-Level Architecture

Graphiti processes **Episodes** (raw text, messages, or JSON) and transforms them into a **Temporal Knowledge Graph** consisting of:

- **EpisodicNodes**: Raw episode storage with metadata
- **EntityNodes**: Extracted entities with names, summaries, attributes, and embeddings
- **EntityEdges**: Relationships (facts) between entities with temporal validity
- **EpisodicEdges**: Links from episodes to the entities they mention
- **CommunityNodes/Edges**: Clustered entity summaries (optional)

### 1.2 Core Pipeline Stages

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EPISODE INPUT                                        │
│  (name, content, source_type, reference_time, source_description)           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: CONTEXT RETRIEVAL                                                  │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • Retrieve previous episodes (last N by valid_at) for context               │
│  • Create EpisodicNode for current episode                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: ENTITY EXTRACTION (extract_nodes)                                  │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • LLM call with episode content + previous episodes + entity types          │
│  • Reflexion loop: check for missed entities, re-extract if needed           │
│  • Output: list[ExtractedEntity] with name + entity_type_id                  │
│  • Convert to EntityNode objects (no embeddings yet)                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: NODE DEDUPLICATION (resolve_extracted_nodes)                       │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • Search existing graph for candidate matches (semantic + BM25)             │
│  • Deterministic resolution: exact name match, MinHash similarity            │
│  • LLM escalation for ambiguous cases                                        │
│  • Output: resolved nodes, uuid_map (extracted→canonical), duplicate_pairs   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 4: EDGE EXTRACTION (extract_edges)                                    │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • LLM call with episode + extracted nodes + previous episodes               │
│  • Extract fact triples: source_entity, relation, target_entity, fact text   │
│  • Extract temporal info: valid_at, invalid_at dates                         │
│  • Reflexion loop for missed facts                                           │
│  • Output: list[EntityEdge] with embeddings                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 5: EDGE RESOLUTION (resolve_extracted_edges)                          │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • Find existing edges between same node pairs                               │
│  • Search for related edges (semantic similarity)                            │
│  • LLM determines: duplicates, contradictions, fact type                     │
│  • Handle temporal invalidation logic                                        │
│  • Output: resolved edges, invalidated edges                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 6: ATTRIBUTE EXTRACTION (extract_attributes_from_nodes)               │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • For each node, extract structured attributes based on entity type schema  │
│  • Update node summaries                                                     │
│  • Generate embeddings for nodes and edges                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 7: PERSISTENCE                                                        │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • Build episodic edges (episode → mentioned entities)                       │
│  • Bulk save: episodes, nodes, episodic edges, entity edges                  │
│  • Optional: update communities                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 1.3 Detailed Component Analysis

#### 1.3.1 Entity Extraction (`extract_nodes`)

**Location**: `graphiti_core/utils/maintenance/node_operations.py:88-208`

**Current Implementation**:
```python
# Prompts in graphiti_core/prompts/extract_nodes.py
class ExtractedEntity(BaseModel):
    name: str
    entity_type_id: int

class ExtractedEntities(BaseModel):
    extracted_entities: list[ExtractedEntity]
```

**LLM Flow**:
1. Build context with entity types, previous episodes, current content
2. Call LLM with `extract_message`, `extract_text`, or `extract_json` prompt based on source type
3. Reflexion: Call LLM again to check for missed entities
4. Repeat up to `MAX_REFLEXION_ITERATIONS` (typically 2)

**Issues Identified**:
- Reflexion loop adds latency without guaranteed improvement
- Entity type classification is done inline with extraction (coupled concerns)
- No optimization mechanism for prompt quality

#### 1.3.2 Node Deduplication (`resolve_extracted_nodes`)

**Location**: `graphiti_core/utils/maintenance/node_operations.py:395-450`

**Two-Phase Resolution**:
1. **Deterministic**: Exact name match, MinHash similarity threshold
2. **LLM Escalation**: For unresolved nodes, ask LLM to match against candidates

**Response Model**:
```python
class NodeDuplicate(BaseModel):
    id: int
    duplicate_idx: int  # -1 if no match
    name: str
    duplicates: list[int]

class NodeResolutions(BaseModel):
    entity_resolutions: list[NodeDuplicate]
```

**Issues Identified**:
- LLM often returns invalid indices outside valid range
- Complex prompt with positional indexing prone to errors
- No retry mechanism for malformed responses

#### 1.3.3 Edge Extraction (`extract_edges`)

**Location**: `graphiti_core/utils/maintenance/edge_operations.py:89-238`

**Response Model**:
```python
class Edge(BaseModel):
    relation_type: str  # SCREAMING_SNAKE_CASE
    source_entity_id: int
    target_entity_id: int
    fact: str
    valid_at: str | None
    invalid_at: str | None

class ExtractedEdges(BaseModel):
    edges: list[Edge]
```

**Issues Identified**:
- Entity ID validation happens post-hoc (many edges rejected)
- Date parsing brittle with various ISO formats
- Reflexion adds significant latency

#### 1.3.4 Edge Resolution (`resolve_extracted_edge`)

**Location**: `graphiti_core/utils/maintenance/edge_operations.py:444-647`

**Response Model**:
```python
class EdgeDuplicate(BaseModel):
    duplicate_facts: list[int]  # indices into EXISTING FACTS
    contradicted_facts: list[int]  # indices into INVALIDATION CANDIDATES
    fact_type: str  # one of edge types or DEFAULT
```

**Complex Logic**:
1. Fast-path: exact fact text match
2. LLM resolution for semantic duplicates
3. Fact type classification
4. Contradiction detection
5. Temporal invalidation logic

**Issues Identified**:
- Two separate index spaces (existing vs invalidation) cause confusion
- LLM often returns indices from wrong list
- Complex temporal logic mixed with LLM interpretation

---

### 1.4 Deterministic Deduplication System (Pre-LLM)

**Location**: `graphiti_core/utils/maintenance/dedup_helpers.py`

A critical component that runs **BEFORE** LLM escalation to reduce costs and latency.

#### 1.4.1 Two-Phase Resolution Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  EXTRACTED ENTITIES                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: ENTROPY FILTERING                                                  │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • Short names (< 6 chars) or low-entropy names → defer to LLM              │
│  • Shannon entropy threshold: 1.5                                            │
│  • Purpose: Avoid false matches on names like "AI", "Bob"                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: EXACT MATCH                                                        │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • Normalize: lowercase, collapse whitespace                                 │
│  • Hash lookup in existing entities                                          │
│  • Single match → resolved immediately                                       │
│  • Multiple matches → defer to LLM                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 3: FUZZY MATCHING (MinHash + LSH)                                     │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • Generate 3-gram shingles from normalized name                             │
│  • Compute MinHash signature (32 permutations)                               │
│  • LSH banding (4-element bands) for candidate retrieval                     │
│  • Jaccard similarity ≥ 0.9 → resolved                                       │
│  • Otherwise → defer to LLM                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 4: LLM ESCALATION                                                     │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • Only unresolved entities sent to LLM                                      │
│  • Reduces LLM calls significantly                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 1.4.2 Key Constants

```python
_NAME_ENTROPY_THRESHOLD = 1.5      # Minimum Shannon entropy
_MIN_NAME_LENGTH = 6               # Minimum chars for fuzzy matching
_MIN_TOKEN_COUNT = 2               # Minimum words for short names
_FUZZY_JACCARD_THRESHOLD = 0.9     # Similarity threshold
_MINHASH_PERMUTATIONS = 32         # Hash functions for MinHash
_MINHASH_BAND_SIZE = 4             # LSH band width
```

#### 1.4.3 Data Structures

```python
@dataclass
class DedupCandidateIndexes:
    """Precomputed lookup structures for deduplication."""
    existing_nodes: list[EntityNode]
    nodes_by_uuid: dict[str, EntityNode]
    normalized_existing: defaultdict[str, list[EntityNode]]  # exact match index
    shingles_by_candidate: dict[str, set[str]]               # for Jaccard
    lsh_buckets: defaultdict[tuple[int, tuple[int, ...]], list[str]]  # LSH index

@dataclass
class DedupResolutionState:
    """Mutable state during resolution."""
    resolved_nodes: list[EntityNode | None]
    uuid_map: dict[str, str]
    unresolved_indices: list[int]
    duplicate_pairs: list[tuple[EntityNode, EntityNode]]
```

#### 1.4.4 DSPy Consideration

The deterministic deduplication should be **preserved** in the DSPy reimplementation:
- It significantly reduces LLM calls (cost savings)
- Provides consistent, reproducible results
- Only ambiguous cases need LLM judgment

---

### 1.5 Community Operations

**Location**: `graphiti_core/utils/maintenance/community_operations.py`

Communities are **optional** higher-level summaries of entity clusters.

#### 1.5.1 Label Propagation Algorithm

```python
def label_propagation(projection: dict[str, list[Neighbor]]) -> list[list[str]]:
    """
    1. Each node starts in its own community
    2. Iterate: each node adopts the plurality community of neighbors
    3. Ties broken by largest community
    4. Continue until convergence
    """
```

#### 1.5.2 Hierarchical Summary Merging

```python
async def build_community(llm_client, community_cluster: list[EntityNode]):
    """
    1. Collect all entity summaries
    2. Pairwise merge summaries using LLM (divide and conquer)
    3. Generate description for final summary
    4. Create CommunityNode with edges to member entities
    """
```

**Prompts Used**:
- `summarize_nodes.summarize_pair`: Merge two summaries into one
- `summarize_nodes.summary_description`: One-sentence description of summary

---

### 1.6 Configuration & Environment Variables

**Location**: `graphiti_core/helpers.py`

| Variable | Default | Description |
|----------|---------|-------------|
| `SEMAPHORE_LIMIT` | 20 | Max concurrent coroutines |
| `MAX_REFLEXION_ITERATIONS` | 0 | Reflexion loop iterations (0 = disabled) |
| `USE_PARALLEL_RUNTIME` | False | Enable parallel processing |

**Key Utility**:
```python
async def semaphore_gather(*coroutines, max_coroutines=None):
    """Bounded asyncio.gather to prevent overwhelming LLM APIs."""
```

---

### 1.7 Search & Retrieval for Candidates

**Location**: `graphiti_core/search/`

Before deduplication, the system searches the graph for candidate matches.

#### 1.7.1 Search Methods

| Method | Used For | Implementation |
|--------|----------|----------------|
| `node_similarity_search` | Find similar entities by embedding | Cosine similarity on `name_embedding` |
| `node_fulltext_search` | Find entities by name keywords | BM25 on entity names |
| `edge_similarity_search` | Find similar facts | Cosine similarity on `fact_embedding` |
| `edge_fulltext_search` | Find facts by keywords | BM25 on fact text |

#### 1.7.2 Reranking Strategies

| Reranker | Description |
|----------|-------------|
| `RRF` | Reciprocal Rank Fusion across search methods |
| `MMR` | Maximal Marginal Relevance for diversity |
| `cross_encoder` | Neural reranking with cross-encoder model |
| `node_distance` | Graph distance from center node |

---

### 1.8 Message Format & Client Architecture

#### Prompt Message Structure

**Location**: `graphiti_core/prompts/models.py`

```python
class Message(BaseModel):
    role: str      # "system" or "user"
    content: str   # The prompt text

# All prompts return list[Message] - typically [system_message, user_message]
PromptFunction = Callable[[dict[str, Any]], list[Message]]
```

#### Client Bundle

**Location**: `graphiti_core/graphiti_types.py`

```python
class GraphitiClients(BaseModel):
    driver: GraphDriver        # Database connection
    llm_client: LLMClient      # Text generation
    embedder: EmbedderClient   # Vector embeddings
    cross_encoder: CrossEncoderClient  # Reranking
    tracer: Tracer             # OpenTelemetry tracing
```

**DSPy Consideration**: DSPy replaces `llm_client` with `dspy.LM`. Other clients remain.

---

### 1.9 Data Models

#### Nodes (`graphiti_core/nodes.py`)

```python
class Node(BaseModel, ABC):
    uuid: str                    # Auto-generated UUID
    name: str                    # Display name
    group_id: str                # Graph partition
    labels: list[str]            # Neo4j labels (e.g., ["Entity", "Person"])
    created_at: datetime         # Creation timestamp

class EpisodicNode(Node):
    source: EpisodeType          # message, json, text
    source_description: str      # Description of data source
    content: str                 # Raw episode content
    valid_at: datetime           # When episode occurred
    entity_edges: list[str]      # UUIDs of edges this episode created

class EntityNode(Node):
    name_embedding: list[float] | None  # Vector embedding of name
    summary: str                        # Generated summary (< 250 chars)
    attributes: dict[str, Any]          # Schema depends on labels/entity_type

class CommunityNode(Node):
    name_embedding: list[float] | None  # Vector embedding
    summary: str                        # Aggregated summary of members
```

#### Edges (`graphiti_core/edges.py`)

```python
class Edge(BaseModel, ABC):
    uuid: str                    # Auto-generated UUID
    group_id: str                # Graph partition
    source_node_uuid: str        # Source entity UUID
    target_node_uuid: str        # Target entity UUID
    created_at: datetime         # Creation timestamp

class EpisodicEdge(Edge):
    # Links episode to entities it mentions (MENTIONS relationship)
    pass

class EntityEdge(Edge):
    name: str                    # Relation type (e.g., WORKS_AT)
    fact: str                    # Natural language fact description
    fact_embedding: list[float] | None  # Vector embedding of fact
    episodes: list[str]          # Episode UUIDs that reference this edge
    expired_at: datetime | None  # When edge was superseded
    valid_at: datetime | None    # When fact became true
    invalid_at: datetime | None  # When fact stopped being true
    attributes: dict[str, Any]   # Typed attributes based on edge type

class CommunityEdge(Edge):
    # Links community to member entities (HAS_MEMBER relationship)
    pass
```

---

### 1.10 Current LLM Client Architecture

**Location**: `graphiti_core/llm_client/client.py`

```python
class LLMClient(ABC):
    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None,
        prompt_name: str | None = None,
    ) -> dict[str, Any]:
```

**Key Issues**:
- Response model schema is appended to prompt as raw JSON schema
- No structured output enforcement at API level
- Validation happens after response (can fail)
- Caching is basic MD5 hash of messages

---

## Part 2: DSPy Reimplementation Strategy

### 2.1 Why DSPy?

| Current Graphiti Approach | DSPy Approach |
|---------------------------|---------------|
| Manual prompt engineering | Declarative Signatures |
| Raw JSON schema in prompts | Native Pydantic integration |
| Ad-hoc reflexion loops | Built-in `ChainOfThought`, optimization |
| No prompt optimization | Automatic few-shot learning, `MIPROv2` |
| Brittle index-based references | Structured field mapping |
| Provider-specific clients | Unified LM interface |

### 2.2 Core DSPy Concepts Mapping

| Graphiti Component | DSPy Equivalent |
|-------------------|-----------------|
| `LLMClient.generate_response()` | `dspy.Predict(Signature)` |
| Prompt templates (extract_nodes.py) | `dspy.Signature` classes |
| Pydantic response models | `dspy.OutputField` with Pydantic types |
| Reflexion loops | `dspy.ChainOfThought` or custom Modules |
| Error handling/retry | DSPy's built-in retries + Assertions |

---

### 2.3 Proposed DSPy Signatures

#### 2.3.1 Entity Extraction Signature

```python
import dspy
from pydantic import BaseModel, Field
from typing import List, Optional

class ExtractedEntity(BaseModel):
    """An entity extracted from the episode content."""
    name: str = Field(description="Name of the entity (full, unambiguous)")
    entity_type: str = Field(description="Type of entity from provided types")
    
class EntityExtractionOutput(BaseModel):
    """Structured output for entity extraction."""
    entities: List[ExtractedEntity] = Field(
        description="List of distinct entities mentioned in the current content"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief explanation of extraction decisions"
    )

class ExtractEntitiesSignature(dspy.Signature):
    """Extract all significant entities from the current content.
    
    Only extract entities explicitly or implicitly mentioned in CURRENT CONTENT.
    Use PREVIOUS CONTEXT only for disambiguation.
    Do NOT extract dates, times, or relationship words.
    """
    
    entity_types: str = dspy.InputField(desc="Available entity types with descriptions")
    previous_context: str = dspy.InputField(desc="Previous episodes for context")
    current_content: str = dspy.InputField(desc="Current episode content to extract from")
    source_type: str = dspy.InputField(desc="Type of source: message, text, or json")
    
    extraction: EntityExtractionOutput = dspy.OutputField()
```

#### 2.3.2 Node Deduplication Signature

```python
class NodeMatch(BaseModel):
    """Resolution result for a single extracted entity."""
    extracted_name: str = Field(description="Name of the extracted entity")
    matched_candidate_name: Optional[str] = Field(
        default=None,
        description="Name of matching existing entity, or null if new"
    )
    is_duplicate: bool = Field(description="Whether this is a duplicate of an existing entity")
    canonical_name: str = Field(description="Best canonical name for this entity")

class NodeDeduplicationOutput(BaseModel):
    """Deduplication results for all extracted entities."""
    resolutions: List[NodeMatch] = Field(
        description="Resolution for each extracted entity in order"
    )

class DeduplicateNodesSignature(dspy.Signature):
    """Determine which extracted entities match existing entities in the graph.
    
    Match entities only if they refer to the SAME real-world object/concept.
    Do NOT match entities that are merely related or similar.
    """
    
    extracted_entities: str = dspy.InputField(desc="Newly extracted entity names with types")
    existing_entities: str = dspy.InputField(desc="Existing entities with names and summaries")
    episode_context: str = dspy.InputField(desc="Context from current and previous episodes")
    
    deduplication: NodeDeduplicationOutput = dspy.OutputField()
```

#### 2.3.3 Edge Extraction Signature

```python
class ExtractedFact(BaseModel):
    """A factual relationship between two entities."""
    source_entity: str = Field(description="Name of source entity (must exist in entities list)")
    target_entity: str = Field(description="Name of target entity (must exist in entities list)")
    relation_type: str = Field(description="Relationship type in SCREAMING_SNAKE_CASE")
    fact: str = Field(description="Natural language description of the relationship")
    valid_at: Optional[str] = Field(default=None, description="ISO 8601 datetime when fact became true")
    invalid_at: Optional[str] = Field(default=None, description="ISO 8601 datetime when fact stopped being true")

class FactExtractionOutput(BaseModel):
    """Extracted facts from the episode."""
    facts: List[ExtractedFact] = Field(description="All factual relationships found")

class ExtractFactsSignature(dspy.Signature):
    """Extract factual relationships between entities from the content.
    
    Facts must involve TWO DISTINCT entities from the provided entity list.
    Use REFERENCE_TIME to resolve relative temporal expressions.
    The fact field should paraphrase the original text, not quote verbatim.
    """
    
    entities: str = dspy.InputField(desc="List of entity names available for facts")
    previous_context: str = dspy.InputField(desc="Previous episodes for context")
    current_content: str = dspy.InputField(desc="Current episode content")
    reference_time: str = dspy.InputField(desc="ISO 8601 timestamp for resolving relative dates")
    fact_types: str = dspy.InputField(desc="Important fact types to look for")
    
    extraction: FactExtractionOutput = dspy.OutputField()
```

#### 2.3.4 Edge Resolution Signature

```python
class FactResolution(BaseModel):
    """Resolution of a new fact against existing facts."""
    is_duplicate: bool = Field(description="Whether this fact duplicates an existing fact")
    duplicate_of: Optional[str] = Field(default=None, description="Fact text of the duplicate, if any")
    contradicts: List[str] = Field(default_factory=list, description="Fact texts this contradicts")
    fact_type: str = Field(description="Classified fact type or DEFAULT")

class ResolveFactSignature(dspy.Signature):
    """Determine if a new fact duplicates or contradicts existing facts.
    
    Duplicates: Facts expressing the same information (not just similar)
    Contradictions: Facts that cannot both be true simultaneously
    """
    
    new_fact: str = dspy.InputField(desc="The new fact to resolve")
    existing_facts: str = dspy.InputField(desc="Existing facts between the same entities")
    all_facts: str = dspy.InputField(desc="All potentially related facts for contradiction check")
    fact_types: str = dspy.InputField(desc="Available fact type classifications")
    
    resolution: FactResolution = dspy.OutputField()
```

---

### 2.4 DSPy Module Architecture

```python
class GraphitiDSPyPipeline(dspy.Module):
    """Complete episode-to-graph pipeline using DSPy."""
    
    def __init__(self, entity_types: dict, fact_types: dict):
        super().__init__()
        self.entity_types = entity_types
        self.fact_types = fact_types
        
        # Core extraction modules
        self.extract_entities = dspy.ChainOfThought(ExtractEntitiesSignature)
        self.dedupe_nodes = dspy.Predict(DeduplicateNodesSignature)
        self.extract_facts = dspy.ChainOfThought(ExtractFactsSignature)
        self.resolve_fact = dspy.Predict(ResolveFactSignature)
        
        # Optional: attribute extraction
        self.extract_attributes = dspy.Predict(ExtractAttributesSignature)
    
    def forward(self, episode_content: str, source_type: str, 
                reference_time: str, previous_episodes: list[str],
                existing_entities: list[dict], existing_facts: list[dict]):
        
        # Stage 1: Extract entities
        entity_result = self.extract_entities(
            entity_types=self._format_entity_types(),
            previous_context=self._format_episodes(previous_episodes),
            current_content=episode_content,
            source_type=source_type
        )
        
        # Stage 2: Deduplicate entities
        dedupe_result = self.dedupe_nodes(
            extracted_entities=self._format_extracted(entity_result.extraction.entities),
            existing_entities=self._format_existing_entities(existing_entities),
            episode_context=episode_content
        )
        
        # Build resolved entity list
        resolved_entities = self._resolve_entities(
            entity_result.extraction.entities,
            dedupe_result.deduplication.resolutions,
            existing_entities
        )
        
        # Stage 3: Extract facts
        facts_result = self.extract_facts(
            entities=self._format_entity_names(resolved_entities),
            previous_context=self._format_episodes(previous_episodes),
            current_content=episode_content,
            reference_time=reference_time,
            fact_types=self._format_fact_types()
        )
        
        # Stage 4: Resolve each fact
        resolved_facts = []
        for fact in facts_result.extraction.facts:
            resolution = self.resolve_fact(
                new_fact=fact.fact,
                existing_facts=self._get_related_facts(fact, existing_facts),
                all_facts=self._format_all_facts(existing_facts),
                fact_types=self._format_fact_types()
            )
            resolved_facts.append((fact, resolution.resolution))
        
        return {
            'entities': resolved_entities,
            'facts': resolved_facts
        }
```

---

### 2.5 Key Improvements Over Current Implementation

#### 2.5.1 Name-Based Entity References (Not Indices)

**Current Problem**: LLM often returns invalid entity indices
```python
# Current: "source_entity_id": 3, "target_entity_id": 1
# Often fails when entity list changes
```

**DSPy Solution**: Reference by name, validate against entity list
```python
class ExtractedFact(BaseModel):
    source_entity: str  # Name, not index
    target_entity: str  # Name, not index
```

#### 2.5.2 Explicit Duplicate Detection (Not Index Matching)

**Current Problem**: Complex index-based matching across multiple lists
```python
# Current: duplicate_idx: 2 (which list? existing? candidates?)
```

**DSPy Solution**: Reference by name/content
```python
class FactResolution(BaseModel):
    duplicate_of: Optional[str]  # Actual fact text, not index
    contradicts: List[str]  # List of fact texts
```

#### 2.5.3 ChainOfThought for Complex Reasoning

**Current Problem**: Flat prompts for complex multi-step reasoning
**DSPy Solution**: Use `dspy.ChainOfThought` for extraction tasks
- Forces step-by-step reasoning
- Improves accuracy on edge cases
- Provides explainability

#### 2.5.4 Optimization Pipeline

**Current Problem**: No mechanism to improve prompts over time
**DSPy Solution**: Use `MIPROv2` optimizer with labeled examples
```python
from dspy.teleprompt import MIPROv2

optimizer = MIPROv2(
    metric=extraction_accuracy_metric,
    num_candidates=7,
    init_temperature=1.0
)

optimized_pipeline = optimizer.compile(
    pipeline,
    trainset=labeled_episodes
)
```

---

### 2.6 Implementation Phases

#### Phase 1: Core Signatures & Basic Pipeline
- [ ] Define all Pydantic models for structured outputs
- [ ] Create DSPy Signatures for each extraction task
- [ ] Implement basic `dspy.Predict` modules
- [ ] Create pipeline orchestrator

#### Phase 2: Enhanced Reasoning
- [ ] Add `ChainOfThought` for entity extraction
- [ ] Add `ChainOfThought` for fact extraction
- [ ] Implement entity name validation
- [ ] Add temporal parsing utilities

#### Phase 3: Deduplication & Resolution
- [ ] Implement name-based entity matching
- [ ] Create fact deduplication module
- [ ] Add contradiction detection
- [ ] Implement temporal invalidation logic

#### Phase 4: Optimization
- [ ] Create labeled dataset from existing graph
- [ ] Define accuracy metrics
- [ ] Run `MIPROv2` optimization
- [ ] Evaluate and iterate

#### Phase 5: Integration
- [ ] Adapt to existing Graphiti driver interface
- [ ] Implement embedding generation
- [ ] Add persistence layer integration
- [ ] Create migration utilities

---

### 2.7 File Structure Proposal

```
graphiti_core/
├── dspy_pipeline/
│   ├── __init__.py
│   ├── signatures/
│   │   ├── __init__.py
│   │   ├── entity_extraction.py
│   │   ├── node_deduplication.py
│   │   ├── fact_extraction.py
│   │   ├── fact_resolution.py
│   │   └── attribute_extraction.py
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── entity_extractor.py
│   │   ├── node_resolver.py
│   │   ├── fact_extractor.py
│   │   ├── fact_resolver.py
│   │   └── pipeline.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── entities.py
│   │   └── facts.py
│   ├── optimizers/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── training.py
│   └── utils/
│       ├── __init__.py
│       ├── formatters.py
│       └── validators.py
```

---

## Part 3: Risk Assessment & Mitigation

### 3.1 Potential Challenges

| Challenge | Mitigation |
|-----------|------------|
| DSPy learning curve | Start with simple `dspy.Predict`, add complexity gradually |
| Name-based matching slower than index | Pre-compute name→UUID maps, fuzzy matching |
| Optimization requires labeled data | Generate from existing successful extractions |
| Provider compatibility | Use DSPy's unified interface, test across providers |
| Batch processing performance | Implement async DSPy calls, parallel execution |

### 3.2 Backward Compatibility

- Keep existing Graphiti interface (`add_episode`, `add_episode_bulk`)
- DSPy pipeline as internal implementation detail
- Gradual migration: flag to switch between old/new pipeline
- Same output data models (EntityNode, EntityEdge, etc.)

---

## Part 4: Success Metrics

1. **Accuracy**: % of correctly extracted entities and facts vs. manual annotation
2. **Stability**: Reduction in LLM hallucination (invalid indices, malformed output)
3. **Latency**: Processing time per episode (target: <5s for typical message)
4. **Optimization Lift**: Improvement after MIPRO optimization vs. baseline

---

## Appendix A: Key Graphiti Code Locations

### Core Pipeline

| Component | File | Function/Class |
|-----------|------|----------------|
| Main Pipeline | `graphiti_core/graphiti.py` | `Graphiti.add_episode()` |
| Bulk Pipeline | `graphiti_core/graphiti.py` | `Graphiti.add_episode_bulk()` |
| Entity Extraction | `graphiti_core/utils/maintenance/node_operations.py` | `extract_nodes()` |
| Node Resolution | `graphiti_core/utils/maintenance/node_operations.py` | `resolve_extracted_nodes()` |
| Attribute Extraction | `graphiti_core/utils/maintenance/node_operations.py` | `extract_attributes_from_nodes()` |
| Edge Extraction | `graphiti_core/utils/maintenance/edge_operations.py` | `extract_edges()` |
| Edge Resolution | `graphiti_core/utils/maintenance/edge_operations.py` | `resolve_extracted_edge()` |
| Bulk Utils | `graphiti_core/utils/bulk_utils.py` | `extract_nodes_and_edges_bulk()`, `dedupe_nodes_bulk()` |

### Deduplication System

| Component | File | Function/Class |
|-----------|------|----------------|
| Deterministic Dedup | `graphiti_core/utils/maintenance/dedup_helpers.py` | `_resolve_with_similarity()` |
| MinHash/LSH | `graphiti_core/utils/maintenance/dedup_helpers.py` | `_minhash_signature()`, `_lsh_bands()` |
| Entropy Filter | `graphiti_core/utils/maintenance/dedup_helpers.py` | `_has_high_entropy()` |
| Index Builder | `graphiti_core/utils/maintenance/dedup_helpers.py` | `_build_candidate_indexes()` |

### Prompts

| Prompt File | Purpose | Key Functions |
|-------------|---------|---------------|
| `extract_nodes.py` | Entity extraction | `extract_message()`, `extract_text()`, `extract_json()`, `reflexion()` |
| `extract_edges.py` | Fact extraction | `edge()`, `reflexion()`, `extract_attributes()` |
| `dedupe_nodes.py` | Node deduplication | `node()`, `nodes()`, `node_list()` |
| `dedupe_edges.py` | Edge deduplication | `edge()`, `edge_list()`, `resolve_edge()` |
| `summarize_nodes.py` | Summary generation | `summarize_pair()`, `summarize_context()`, `summary_description()` |
| `invalidate_edges.py` | Contradiction detection | `v1()`, `v2()` |

### Data Models

| Component | File | Class |
|-----------|------|-------|
| Episode Node | `graphiti_core/nodes.py` | `EpisodicNode` |
| Entity Node | `graphiti_core/nodes.py` | `EntityNode` |
| Community Node | `graphiti_core/nodes.py` | `CommunityNode` |
| Entity Edge | `graphiti_core/edges.py` | `EntityEdge` |
| Episodic Edge | `graphiti_core/edges.py` | `EpisodicEdge` |
| Community Edge | `graphiti_core/edges.py` | `CommunityEdge` |

### Search & Retrieval

| Component | File | Function |
|-----------|------|----------|
| Main Search | `graphiti_core/search/search.py` | `search()` |
| Edge Search | `graphiti_core/search/search.py` | `edge_search()` |
| Node Search | `graphiti_core/search/search.py` | `node_search()` |
| Search Utils | `graphiti_core/search/search_utils.py` | `node_similarity_search()`, `edge_fulltext_search()` |
| RRF Reranker | `graphiti_core/search/search_utils.py` | `rrf()` |
| MMR Reranker | `graphiti_core/search/search_utils.py` | `maximal_marginal_relevance()` |

### Community & Summarization

| Component | File | Function |
|-----------|------|----------|
| Build Communities | `graphiti_core/utils/maintenance/community_operations.py` | `build_communities()` |
| Label Propagation | `graphiti_core/utils/maintenance/community_operations.py` | `label_propagation()` |
| Summarize Pair | `graphiti_core/utils/maintenance/community_operations.py` | `summarize_pair()` |
| Update Community | `graphiti_core/utils/maintenance/community_operations.py` | `update_community()` |

### Utilities

| Component | File | Function/Class |
|-----------|------|----------------|
| Semaphore Gather | `graphiti_core/helpers.py` | `semaphore_gather()` |
| LLM Client | `graphiti_core/llm_client/client.py` | `LLMClient` |
| Embedder | `graphiti_core/embedder/client.py` | `EmbedderClient` |
| Cross Encoder | `graphiti_core/cross_encoder/client.py` | `CrossEncoderClient` |

---

## Appendix B: Complete Prompt & Response Model Inventory

### Entity Extraction Prompts

| Prompt | Input Context | Response Model | Purpose |
|--------|---------------|----------------|---------|
| `extract_nodes.extract_message` | episode_content, previous_episodes, entity_types, custom_prompt | `ExtractedEntities` | Extract from conversational messages |
| `extract_nodes.extract_text` | episode_content, entity_types, custom_prompt | `ExtractedEntities` | Extract from plain text |
| `extract_nodes.extract_json` | episode_content, source_description, entity_types, custom_prompt | `ExtractedEntities` | Extract from JSON data |
| `extract_nodes.reflexion` | episode_content, previous_episodes, extracted_entities | `MissedEntities` | Check for missed entities |
| `extract_nodes.extract_attributes` | node, episode_content, previous_episodes | (Custom Pydantic) | Extract typed attributes |
| `extract_nodes.extract_summary` | node, episode_content, previous_episodes | `EntitySummary` | Generate/update entity summary |

### Edge Extraction Prompts

| Prompt | Input Context | Response Model | Purpose |
|--------|---------------|----------------|---------|
| `extract_edges.edge` | episode_content, nodes, previous_episodes, reference_time, edge_types | `ExtractedEdges` | Extract facts between entities |
| `extract_edges.reflexion` | episode_content, nodes, extracted_facts | `MissingFacts` | Check for missed facts |
| `extract_edges.extract_attributes` | episode_content, reference_time, fact | (Custom Pydantic) | Extract typed edge attributes |

### Deduplication Prompts

| Prompt | Input Context | Response Model | Purpose |
|--------|---------------|----------------|---------|
| `dedupe_nodes.node` | extracted_node, existing_nodes, episode_content, entity_type_description | `NodeResolutions` | Single node dedup |
| `dedupe_nodes.nodes` | extracted_nodes, existing_nodes, episode_content | `NodeResolutions` | Batch node dedup |
| `dedupe_nodes.node_list` | nodes (list) | Custom | Merge duplicate node lists |
| `dedupe_edges.edge` | extracted_edges, related_edges | `EdgeDuplicate` | Single edge dedup |
| `dedupe_edges.resolve_edge` | new_edge, existing_edges, edge_invalidation_candidates, edge_types | `EdgeDuplicate` | Full edge resolution |

### Summarization Prompts

| Prompt | Input Context | Response Model | Purpose |
|--------|---------------|----------------|---------|
| `summarize_nodes.summarize_pair` | node_summaries (pair) | `Summary` | Merge two summaries |
| `summarize_nodes.summarize_context` | previous_episodes, episode_content, node_name, node_summary, attributes | `Summary` | Summary from context |
| `summarize_nodes.summary_description` | summary | `SummaryDescription` | One-sentence description |

### Invalidation Prompts

| Prompt | Input Context | Response Model | Purpose |
|--------|---------------|----------------|---------|
| `invalidate_edges.v1` | previous_episodes, current_episode, existing_edges, new_edges | `InvalidatedEdges` | Detect contradictions (batch) |
| `invalidate_edges.v2` | existing_edges, new_edge | `InvalidatedEdges` | Detect contradictions (single) |

### Response Model Definitions

```python
# Entity Extraction
class ExtractedEntity(BaseModel):
    name: str
    entity_type_id: int

class ExtractedEntities(BaseModel):
    extracted_entities: list[ExtractedEntity]

class MissedEntities(BaseModel):
    missed_entities: list[str]

class EntitySummary(BaseModel):
    summary: str  # < 250 chars

# Edge Extraction
class Edge(BaseModel):
    relation_type: str      # SCREAMING_SNAKE_CASE
    source_entity_id: int   # Index into entities list
    target_entity_id: int   # Index into entities list
    fact: str               # Natural language description
    valid_at: str | None    # ISO 8601
    invalid_at: str | None  # ISO 8601

class ExtractedEdges(BaseModel):
    edges: list[Edge]

class MissingFacts(BaseModel):
    missing_facts: list[str]

# Node Deduplication
class NodeDuplicate(BaseModel):
    id: int
    duplicate_idx: int      # -1 if no match
    name: str               # Canonical name
    duplicates: list[int]   # All matching indices

class NodeResolutions(BaseModel):
    entity_resolutions: list[NodeDuplicate]

# Edge Deduplication
class EdgeDuplicate(BaseModel):
    duplicate_facts: list[int]     # Indices into EXISTING FACTS
    contradicted_facts: list[int]  # Indices into INVALIDATION CANDIDATES
    fact_type: str                 # Classified type or "DEFAULT"

# Summarization
class Summary(BaseModel):
    summary: str  # < 250 chars

class SummaryDescription(BaseModel):
    description: str  # One sentence

# Invalidation
class InvalidatedEdges(BaseModel):
    contradicted_facts: list[int]
```

---

## Appendix C: Temporal Logic Details

### Edge Temporal Fields

| Field | Type | Description |
|-------|------|-------------|
| `valid_at` | datetime | When the fact became true |
| `invalid_at` | datetime | When the fact stopped being true |
| `expired_at` | datetime | When the edge was marked as superseded |
| `created_at` | datetime | When the edge was created in the graph |

### Invalidation Logic

```python
def resolve_edge_contradictions(resolved_edge, invalidation_candidates):
    """
    For each candidate edge that contradicts the new edge:
    1. If candidate.invalid_at <= resolved_edge.valid_at: no conflict (already expired)
    2. If resolved_edge.invalid_at <= candidate.valid_at: no conflict (new edge expired first)
    3. If candidate.valid_at < resolved_edge.valid_at: invalidate candidate
       - Set candidate.invalid_at = resolved_edge.valid_at
       - Set candidate.expired_at = now
    """
```

### DSPy Consideration

Temporal logic should be handled **outside** of DSPy modules:
- Parse dates after LLM extraction
- Apply invalidation rules in Python
- Keep LLM focused on semantic understanding

---

## Appendix D: DSPy Quick Reference

```python
# Basic Signature
class MySignature(dspy.Signature):
    """Task description for the LLM."""
    input_field: str = dspy.InputField(desc="Description")
    output_field: MyPydanticModel = dspy.OutputField()

# Basic Module
class MyModule(dspy.Module):
    def __init__(self):
        self.predictor = dspy.Predict(MySignature)
    
    def forward(self, input_field):
        return self.predictor(input_field=input_field)

# ChainOfThought (adds reasoning)
self.predictor = dspy.ChainOfThought(MySignature)

# Configure LM
lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv('OPENAI_API_KEY'))
dspy.configure(lm=lm)

# Optimization
from dspy.teleprompt import MIPROv2
optimizer = MIPROv2(metric=my_metric)
optimized = optimizer.compile(my_module, trainset=examples)
```

---

## Appendix E: Known Issues & Stability Problems

### 1. Index Out of Range Errors

**Problem**: LLM returns entity/fact indices outside valid range
**Location**: `node_operations.py:355-386`, `edge_operations.py:541-573`
**Frequency**: Common with complex entity lists

**Current Mitigation**:
```python
if relative_id not in valid_relative_range:
    logger.warning('Skipping invalid LLM dedupe id %d', relative_id)
    continue
```

### 2. Missing Resolutions

**Problem**: LLM doesn't return resolution for all entities
**Location**: `node_operations.py:344-346`
**Impact**: Entities treated as new (duplicates in graph)

### 3. Index Space Confusion

**Problem**: `resolve_edge` uses TWO separate lists with independent indices
- `existing_edges` (for duplicates)
- `edge_invalidation_candidates` (for contradictions)

LLM frequently returns indices from wrong list.

### 4. Date Parsing Failures

**Problem**: LLM returns dates in various formats
**Location**: `edge_operations.py:205-220`
**Examples**: 
- `"2024-01-15"` (missing time)
- `"January 15, 2024"` (natural language)
- `"2024-01-15T10:30:00"` (missing timezone)

### 5. Empty/Whitespace Facts

**Problem**: LLM sometimes returns empty fact strings
**Location**: `edge_operations.py:185-186`
**Current Mitigation**: `if not edge_data.fact.strip(): continue`

### 6. Reflexion Latency

**Problem**: Each reflexion iteration adds full LLM call latency
**Impact**: 2-3x processing time with `MAX_REFLEXION_ITERATIONS > 0`
**Trade-off**: Sometimes catches missed entities, often finds nothing

---

## Appendix F: DSPy Migration Checklist

### Phase 1: Foundation
- [ ] Set up DSPy project structure
- [ ] Create base Pydantic models matching existing ones
- [ ] Configure DSPy with target LLM provider
- [ ] Create test harness with sample episodes

### Phase 2: Core Signatures
- [ ] `ExtractEntitiesSignature` - entity extraction
- [ ] `DeduplicateNodesSignature` - node dedup (LLM portion only)
- [ ] `ExtractFactsSignature` - fact extraction
- [ ] `ResolveFactSignature` - fact dedup & contradiction
- [ ] `ExtractAttributesSignature` - typed attribute extraction
- [ ] `SummarizeEntitySignature` - summary generation

### Phase 3: Modules
- [ ] `EntityExtractor` module with ChainOfThought
- [ ] `NodeResolver` module (deterministic + LLM)
- [ ] `FactExtractor` module with ChainOfThought
- [ ] `FactResolver` module
- [ ] `GraphitiDSPyPipeline` orchestrator

### Phase 4: Integration
- [ ] Preserve deterministic dedup logic
- [ ] Integrate with existing graph drivers
- [ ] Integrate with embedder clients
- [ ] Implement temporal logic
- [ ] Add persistence layer

### Phase 5: Testing & Optimization
- [ ] Unit tests for each signature
- [ ] Integration tests with real episodes
- [ ] Create labeled training dataset
- [ ] Run MIPROv2 optimization
- [ ] A/B test against current implementation
- [ ] Performance benchmarking

---

*Document created: 2025-01-15*
*Last updated: 2025-01-15*
*Graphiti Version: Latest (main branch)*
*DSPy Version: Compatible with 2.5+*

