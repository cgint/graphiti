# Graphiti DSPy Reimplementation Plan

> **Document Purpose**: Thorough plan for reimplementing Graphiti's LLM-based pipeline using DSPy. Focuses on core DSPy concepts and key implementation patterns.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [DSPy Architecture Overview](#2-dspy-architecture-overview)
3. [Signature Definitions](#3-signature-definitions)
4. [Module Architecture](#4-module-architecture)
5. [Optimization Strategy](#5-optimization-strategy)
6. [Evaluation Metrics](#6-evaluation-metrics)
7. [Data Preparation](#7-data-preparation)
8. [Migration Path](#8-migration-path)
9. [Key Design Decisions](#9-key-design-decisions)

---

## 1. Executive Summary

### What We're Replacing

| Current Graphiti | DSPy Replacement |
|------------------|------------------|
| Ad-hoc prompt templates | Declarative Signatures |
| Manual retry loops (reflexion) | `dspy.Refine` / automatic backtracking |
| Index-based entity references | Name-based typed outputs |
| No systematic optimization | `MIPROv2` optimizer |
| Pydantic for structured output | DSPy native Pydantic support |

### What We're Keeping

- Search system (unchanged)
- Database drivers (unchanged)
- Embedding generation (unchanged)
- Temporal logic (unchanged)
- `GraphitiClients` bundle (unchanged)

### DSPy Benefits for This Use Case

1. **Declarative Signatures** → Clear input/output contracts
2. **Typed Pydantic Outputs** → Automatic JSON schema enforcement
3. **ChainOfThought** → Better reasoning for complex deduplication
4. **MIPROv2 Optimizer** → Automatic prompt tuning on labeled data
5. **Modular Design** → Each pipeline stage is an independent module

---

## 2. DSPy Architecture Overview

### 2.1 Core Concepts Applied

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DSPy ARCHITECTURE FOR GRAPHITI                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   SIGNATURES    │    │    MODULES      │    │   OPTIMIZERS    │         │
│  │                 │    │                 │    │                 │         │
│  │ Define WHAT     │───►│ Define HOW      │───►│ Tune prompts    │         │
│  │ inputs/outputs  │    │ to execute      │    │ automatically   │         │
│  │                 │    │                 │    │                 │         │
│  │ • ExtractEntities│    │ • dspy.Predict  │    │ • MIPROv2       │         │
│  │ • DeduplicateNode│    │ • ChainOfThought│    │ • BootstrapFS   │         │
│  │ • ExtractFacts  │    │ • Custom logic  │    │                 │         │
│  │ • ResolveFact   │    │                 │    │                 │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│           │                      │                      │                   │
│           ▼                      ▼                      ▼                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    GraphitiDSPyPipeline                              │   │
│  │                                                                      │   │
│  │   Episode ──► Extract ──► Dedupe ──► Extract ──► Resolve ──► Save   │   │
│  │              Entities    Nodes     Facts       Facts               │   │
│  │                 │           │         │           │                  │   │
│  │                 ▼           ▼         ▼           ▼                  │   │
│  │              [Module]   [Module]   [Module]   [Module]               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Pipeline → DSPy Mapping

| Pipeline Stage | DSPy Signature | DSPy Module | Notes |
|----------------|----------------|-------------|-------|
| Entity Extraction | `ExtractEntitiesSignature` | `dspy.ChainOfThought` | Reasoning helps with complex text |
| Node Deduplication | `DeduplicateNodesSignature` | `dspy.ChainOfThought` | Search provides candidates |
| Fact Extraction | `ExtractFactsSignature` | `dspy.ChainOfThought` | Needs extracted entities as input |
| Fact Resolution | `ResolveFactSignature` | `dspy.ChainOfThought` | Search provides existing facts |
| Summarization | `SummarizeEntitySignature` | `dspy.Predict` | Simple transformation |

---

## 3. Signature Definitions

### 3.1 Entity Extraction Signature

```python
import dspy
from pydantic import BaseModel, Field
from typing import List, Optional

# Pydantic model for structured output
class ExtractedEntity(BaseModel):
    """A single extracted entity."""
    name: str = Field(description="Canonical name of the entity")
    entity_type: str = Field(description="Type classification (Person, Organization, etc.)")

class ExtractedEntitiesResult(BaseModel):
    """Collection of extracted entities."""
    entities: List[ExtractedEntity] = Field(
        description="List of entities extracted from the episode"
    )

# DSPy Signature
class ExtractEntitiesSignature(dspy.Signature):
    """
    Extract named entities from episode content.
    Identify all significant entities mentioned: people, organizations, 
    places, concepts, products, events, etc.
    Return canonical names (e.g., "Barack Obama" not "Obama" or "the president").
    """
    
    episode_content: str = dspy.InputField(
        desc="The episode text content to extract entities from"
    )
    previous_context: str = dspy.InputField(
        desc="Summary of previous episodes for context"
    )
    entity_types: List[str] = dspy.InputField(
        desc="Available entity type classifications"
    )
    
    extracted_entities: ExtractedEntitiesResult = dspy.OutputField(
        desc="Structured list of extracted entities with types"
    )
```

**Key Design Decisions:**
- Uses Pydantic `BaseModel` for automatic JSON schema enforcement
- `entity_types` as input allows dynamic type systems
- No index-based references — uses entity names directly
- `previous_context` enables cross-episode consistency

### 3.2 Node Deduplication Signature

```python
class DeduplicationDecision(BaseModel):
    """Decision for a single entity deduplication."""
    extracted_entity_name: str = Field(
        description="Name of the extracted entity being resolved"
    )
    is_duplicate: bool = Field(
        description="True if this entity matches an existing one"
    )
    matched_existing_name: Optional[str] = Field(
        default=None,
        description="Name of the matched existing entity, or None if new"
    )
    canonical_name: str = Field(
        description="The best canonical name to use (existing or new)"
    )

class DeduplicationResult(BaseModel):
    """Results for all entity deduplication decisions."""
    decisions: List[DeduplicationDecision] = Field(
        description="Deduplication decision for each extracted entity"
    )

class DeduplicateNodesSignature(dspy.Signature):
    """
    Determine which extracted entities match existing entities in the knowledge graph.
    
    For each extracted entity:
    1. Check if it refers to the same real-world entity as any existing entity
    2. Account for aliases, nicknames, abbreviations, and partial names
    3. If a match is found, select the most complete canonical name
    4. If no match, mark as new entity
    
    Be conservative: only match if confident they refer to the same entity.
    """
    
    extracted_entities: List[str] = dspy.InputField(
        desc="Names of newly extracted entities to resolve"
    )
    existing_entities: List[str] = dspy.InputField(
        desc="Names of candidate existing entities from search"
    )
    
    deduplication: DeduplicationResult = dspy.OutputField(
        desc="Deduplication decision for each extracted entity"
    )
```

**Key Design Decisions:**
- **Name-based references** instead of fragile indices
- `existing_entities` comes from search (external input)
- `is_duplicate` is explicit boolean, not implicit index check
- `canonical_name` allows LLM to pick best representation

### 3.3 Fact Extraction Signature

```python
class ExtractedFact(BaseModel):
    """A single extracted fact/relationship."""
    source_entity: str = Field(
        description="Name of the source entity"
    )
    relation_type: str = Field(
        description="Relationship predicate in SCREAMING_SNAKE_CASE"
    )
    target_entity: str = Field(
        description="Name of the target entity"
    )
    fact_statement: str = Field(
        description="Natural language description of the relationship"
    )
    valid_at: Optional[str] = Field(
        default=None,
        description="ISO 8601 datetime when the fact became true"
    )
    invalid_at: Optional[str] = Field(
        default=None,
        description="ISO 8601 datetime when the fact stopped being true"
    )

class ExtractedFactsResult(BaseModel):
    """Collection of extracted facts."""
    facts: List[ExtractedFact] = Field(
        description="List of facts extracted from the episode"
    )

class ExtractFactsSignature(dspy.Signature):
    """
    Extract factual relationships between entities from episode content.
    
    For each fact:
    1. Identify source and target entities (must be from the entities list)
    2. Determine the relationship type
    3. Write a clear natural language fact statement
    4. Extract temporal bounds if mentioned (valid_at, invalid_at)
    
    Only extract facts explicitly stated or strongly implied in the content.
    """
    
    episode_content: str = dspy.InputField(
        desc="The episode text content"
    )
    entities: List[str] = dspy.InputField(
        desc="Entities available for fact extraction (from deduplication)"
    )
    previous_context: str = dspy.InputField(
        desc="Summary of previous episodes for context"
    )
    current_time: str = dspy.InputField(
        desc="Reference time for temporal reasoning"
    )
    
    extracted_facts: ExtractedFactsResult = dspy.OutputField(
        desc="Structured list of extracted facts with temporal bounds"
    )
```

**Key Design Decisions:**
- `entities` list constrains valid source/target names
- Temporal fields are optional strings (ISO format)
- `fact_statement` is human-readable text

### 3.4 Fact Resolution Signature

```python
class FactResolutionDecision(BaseModel):
    """Resolution decision for a single fact."""
    new_fact: str = Field(
        description="The new fact being resolved"
    )
    is_duplicate: bool = Field(
        description="True if this fact duplicates an existing one"
    )
    duplicate_of: Optional[str] = Field(
        default=None,
        description="The existing fact this duplicates, if any"
    )
    contradicts: List[str] = Field(
        default_factory=list,
        description="List of existing facts this contradicts"
    )
    fact_type: str = Field(
        default="RELATES_TO",
        description="Classified fact type"
    )

class FactResolutionResult(BaseModel):
    """Results for fact resolution."""
    decisions: List[FactResolutionDecision]

class ResolveFactSignature(dspy.Signature):
    """
    Resolve a new fact against existing facts in the knowledge graph.
    
    For each new fact:
    1. Check if it duplicates any existing fact (same meaning)
    2. Check if it contradicts any existing fact (opposite meaning)
    3. Classify the fact type from available types
    
    Duplicates: Same entities, same relationship, same meaning.
    Contradictions: Same entities, opposing statements.
    """
    
    new_fact: str = dspy.InputField(
        desc="The new fact statement to resolve"
    )
    related_facts: List[str] = dspy.InputField(
        desc="Existing facts between the same entities (from search)"
    )
    invalidation_candidates: List[str] = dspy.InputField(
        desc="Broader set of potentially contradictory facts (from search)"
    )
    available_fact_types: List[str] = dspy.InputField(
        desc="Valid fact type classifications"
    )
    
    resolution: FactResolutionDecision = dspy.OutputField(
        desc="Resolution decision for the new fact"
    )
```

**Key Design Decisions:**
- Two search inputs: `related_facts` (narrow) and `invalidation_candidates` (broad)
- Explicit `contradicts` list for temporal invalidation
- `fact_type` classification integrated into resolution

### 3.5 Entity Summary Signature

```python
class EntitySummary(BaseModel):
    """Summary for an entity."""
    summary: str = Field(description="Concise summary of the entity")

class SummarizeEntitySignature(dspy.Signature):
    """
    Generate a concise summary for an entity based on its relationships.
    The summary should capture the entity's key characteristics and 
    important relationships in 2-3 sentences.
    """
    
    entity_name: str = dspy.InputField(desc="Name of the entity")
    entity_type: str = dspy.InputField(desc="Type of the entity")
    related_facts: List[str] = dspy.InputField(
        desc="Facts involving this entity"
    )
    
    summary: EntitySummary = dspy.OutputField(
        desc="Concise summary of the entity"
    )
```

---

## 4. Module Architecture

### 4.1 Module Design Principles

```python
import dspy

class GraphitiModule(dspy.Module):
    """Base pattern for Graphiti DSPy modules."""
    
    def __init__(self):
        super().__init__()
        # Modules use ChainOfThought for complex reasoning
        self.predictor = dspy.ChainOfThought(SomeSignature)
    
    def forward(self, **inputs):
        # 1. Prepare inputs (from search, etc.)
        # 2. Call predictor
        # 3. Validate/transform outputs
        return self.predictor(**inputs)
```

### 4.2 Entity Extraction Module

```python
class EntityExtractionModule(dspy.Module):
    """Extract entities from episode content."""
    
    def __init__(self, entity_types: List[str]):
        super().__init__()
        self.entity_types = entity_types
        self.extractor = dspy.ChainOfThought(ExtractEntitiesSignature)
    
    def forward(
        self, 
        episode_content: str,
        previous_context: str = "",
    ) -> ExtractedEntitiesResult:
        
        result = self.extractor(
            episode_content=episode_content,
            previous_context=previous_context,
            entity_types=self.entity_types,
        )
        
        return result.extracted_entities
```

**Why ChainOfThought:**
- Entity extraction benefits from reasoning about context
- LLM can explain why an entity was classified a certain type
- Traces are valuable for debugging and optimization

### 4.3 Node Deduplication Module

```python
class NodeDeduplicationModule(dspy.Module):
    """
    Deduplicate extracted entities against existing graph entities.
    Uses search results as input for candidate matching.
    """
    
    def __init__(self):
        super().__init__()
        self.deduplicator = dspy.ChainOfThought(DeduplicateNodesSignature)
    
    def forward(
        self,
        extracted_entities: List[str],
        existing_entities: List[str],  # From search!
    ) -> DeduplicationResult:
        
        result = self.deduplicator(
            extracted_entities=extracted_entities,
            existing_entities=existing_entities,
        )
        
        return result.deduplication
```

**Integration with Search:**
```python
# How this module is called in the pipeline:

# 1. Search for candidates (unchanged from current implementation)
search_results = await search(
    clients=clients,
    query=entity.name,
    config=NODE_HYBRID_SEARCH_RRF,
)
existing_names = [node.name for node in search_results.nodes]

# 2. Call DSPy module with search results
dedup_result = deduplication_module(
    extracted_entities=[entity.name for entity in extracted],
    existing_entities=existing_names,
)
```

### 4.4 Fact Extraction Module

```python
class FactExtractionModule(dspy.Module):
    """Extract facts/relationships between entities."""
    
    def __init__(self):
        super().__init__()
        self.extractor = dspy.ChainOfThought(ExtractFactsSignature)
    
    def forward(
        self,
        episode_content: str,
        entities: List[str],  # From deduplication
        previous_context: str = "",
        current_time: str = "",
    ) -> ExtractedFactsResult:
        
        result = self.extractor(
            episode_content=episode_content,
            entities=entities,
            previous_context=previous_context,
            current_time=current_time,
        )
        
        return result.extracted_facts
```

### 4.5 Fact Resolution Module

```python
class FactResolutionModule(dspy.Module):
    """
    Resolve new facts against existing graph facts.
    Handles deduplication and contradiction detection.
    """
    
    def __init__(self):
        super().__init__()
        self.resolver = dspy.ChainOfThought(ResolveFactSignature)
    
    def forward(
        self,
        new_fact: str,
        related_facts: List[str],         # From narrow search
        invalidation_candidates: List[str],  # From broad search
        available_fact_types: List[str] = ["RELATES_TO"],
    ) -> FactResolutionDecision:
        
        result = self.resolver(
            new_fact=new_fact,
            related_facts=related_facts,
            invalidation_candidates=invalidation_candidates,
            available_fact_types=available_fact_types,
        )
        
        return result.resolution
```

### 4.6 Complete Pipeline Module

```python
class GraphitiDSPyPipeline(dspy.Module):
    """
    Complete episode-to-graph pipeline using DSPy.
    Orchestrates all sub-modules with search integration.
    """
    
    def __init__(self, entity_types: List[str], fact_types: List[str]):
        super().__init__()
        
        # Initialize sub-modules
        self.entity_extractor = EntityExtractionModule(entity_types)
        self.node_deduplicator = NodeDeduplicationModule()
        self.fact_extractor = FactExtractionModule()
        self.fact_resolver = FactResolutionModule()
        self.summarizer = dspy.Predict(SummarizeEntitySignature)
        
        self.fact_types = fact_types
    
    def forward(
        self,
        episode_content: str,
        previous_context: str,
        current_time: str,
        # Search functions passed in (not LLM concerns)
        search_nodes_fn,
        search_edges_fn,
    ):
        # 1. Extract entities
        entities_result = self.entity_extractor(
            episode_content=episode_content,
            previous_context=previous_context,
        )
        extracted_names = [e.name for e in entities_result.entities]
        
        # 2. Search for existing entities (NOT an LLM call)
        existing_nodes = search_nodes_fn(extracted_names)
        
        # 3. Deduplicate entities
        dedup_result = self.node_deduplicator(
            extracted_entities=extracted_names,
            existing_entities=existing_nodes,
        )
        
        # 4. Extract facts using resolved entities
        resolved_names = [d.canonical_name for d in dedup_result.decisions]
        facts_result = self.fact_extractor(
            episode_content=episode_content,
            entities=resolved_names,
            previous_context=previous_context,
            current_time=current_time,
        )
        
        # 5. Resolve each fact (with search for candidates)
        resolved_facts = []
        for fact in facts_result.facts:
            # Search for related facts (NOT an LLM call)
            related, candidates = search_edges_fn(fact)
            
            resolution = self.fact_resolver(
                new_fact=fact.fact_statement,
                related_facts=related,
                invalidation_candidates=candidates,
                available_fact_types=self.fact_types,
            )
            resolved_facts.append(resolution)
        
        return {
            "entities": dedup_result,
            "facts": resolved_facts,
        }
```

---

## 5. Optimization Strategy

### 5.1 MIPROv2 Configuration

```python
from dspy.teleprompt import MIPROv2

# Configure optimizer
optimizer = MIPROv2(
    metric=graphiti_evaluation_metric,  # Custom metric (see section 6)
    auto="medium",                       # Optimization budget
    prompt_model=dspy.LM("openai/gpt-4o"),  # Strong model for proposals
    num_threads=4,                       # Parallel evaluation
)

# Compile the pipeline
optimized_pipeline = optimizer.compile(
    GraphitiDSPyPipeline(entity_types, fact_types),
    trainset=training_examples,
    valset=validation_examples,
)
```

### 5.2 What MIPROv2 Optimizes

| Component | What Gets Optimized |
|-----------|---------------------|
| Instructions | Natural language task descriptions |
| Few-shot examples | Automatically selected demonstrations |
| Module prompts | Each module's prompt separately |

### 5.3 Optimization Process

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       MIPROv2 OPTIMIZATION FLOW                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. BOOTSTRAPPING                                                           │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │ Run pipeline on training examples                                │    │
│     │ Collect execution traces for each module                         │    │
│     │ Filter: Keep only traces where metric > threshold                │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│  2. GROUNDED PROPOSAL                                                       │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │ Analyze: program code + data + high-scoring traces               │    │
│     │ Generate: multiple candidate instructions per module             │    │
│     │ Example: "Extract all named entities..." vs "Identify people..." │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│  3. DISCRETE SEARCH                                                         │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │ Sample combinations of instructions + few-shot examples          │    │
│     │ Evaluate each combination using metric                           │    │
│     │ Use Bayesian Optimization to guide search                        │    │
│     │ Output: Best-performing prompt configuration                     │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.4 Per-Module Optimization

Each module can be optimized independently:

```python
# Optimize just the entity extraction module
entity_optimizer = MIPROv2(metric=entity_extraction_metric, auto="light")
optimized_extractor = entity_optimizer.compile(
    EntityExtractionModule(entity_types),
    trainset=entity_extraction_examples,
)

# Optimize just the deduplication module
dedup_optimizer = MIPROv2(metric=dedup_accuracy_metric, auto="light")
optimized_deduplicator = dedup_optimizer.compile(
    NodeDeduplicationModule(),
    trainset=dedup_examples,
)
```

---

## 6. Evaluation Metrics

### 6.1 Entity Extraction Metric

```python
def entity_extraction_metric(
    example: dspy.Example, 
    pred: dspy.Prediction, 
    trace=None
) -> float:
    """
    Evaluate entity extraction quality.
    
    Scores:
    - Precision: What fraction of predicted entities are correct?
    - Recall: What fraction of gold entities were found?
    - F1: Harmonic mean
    """
    gold_entities = set(example.gold_entities)
    pred_entities = set(e.name for e in pred.extracted_entities.entities)
    
    if not pred_entities:
        return 0.0
    
    true_positives = len(gold_entities & pred_entities)
    precision = true_positives / len(pred_entities)
    recall = true_positives / len(gold_entities) if gold_entities else 1.0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
```

### 6.2 Deduplication Metric

```python
def deduplication_metric(
    example: dspy.Example,
    pred: dspy.Prediction,
    trace=None
) -> float:
    """
    Evaluate deduplication accuracy.
    
    Correct if:
    - is_duplicate matches gold
    - If duplicate, matched_existing_name is correct
    """
    gold = example.gold_deduplication
    pred_decisions = pred.deduplication.decisions
    
    correct = 0
    total = len(gold)
    
    for gold_dec, pred_dec in zip(gold, pred_decisions):
        if gold_dec["is_duplicate"] == pred_dec.is_duplicate:
            if not gold_dec["is_duplicate"]:
                correct += 1  # Correctly identified as new
            elif gold_dec["matched_name"] == pred_dec.matched_existing_name:
                correct += 1  # Correctly matched
    
    return correct / total if total > 0 else 0.0
```

### 6.3 Fact Resolution Metric

```python
def fact_resolution_metric(
    example: dspy.Example,
    pred: dspy.Prediction,
    trace=None
) -> float:
    """
    Evaluate fact resolution quality.
    
    Components:
    - Duplicate detection accuracy
    - Contradiction detection accuracy
    - Fact type classification accuracy
    """
    gold = example.gold_resolution
    pred_res = pred.resolution
    
    score = 0.0
    
    # Duplicate detection (40% weight)
    if gold["is_duplicate"] == pred_res.is_duplicate:
        score += 0.4
        if gold["is_duplicate"] and gold["duplicate_of"] == pred_res.duplicate_of:
            score += 0.1  # Bonus for correct match
    
    # Contradiction detection (40% weight)
    gold_contradicts = set(gold["contradicts"])
    pred_contradicts = set(pred_res.contradicts)
    if gold_contradicts == pred_contradicts:
        score += 0.4
    elif gold_contradicts & pred_contradicts:
        # Partial credit
        score += 0.2
    
    # Fact type classification (20% weight)
    if gold["fact_type"] == pred_res.fact_type:
        score += 0.2
    
    return score
```

### 6.4 End-to-End Pipeline Metric

```python
def pipeline_metric(
    example: dspy.Example,
    pred: dspy.Prediction,
    trace=None
) -> float:
    """
    End-to-end pipeline evaluation.
    Combines entity and fact quality scores.
    """
    entity_score = entity_extraction_metric(example, pred, trace)
    dedup_score = deduplication_metric(example, pred, trace)
    fact_score = fact_resolution_metric(example, pred, trace)
    
    # Weighted combination
    return 0.3 * entity_score + 0.3 * dedup_score + 0.4 * fact_score
```

---

## 7. Data Preparation

### 7.1 Training Example Format

```python
# Entity extraction example
entity_example = dspy.Example(
    episode_content="Barack Obama was the 44th president of the United States...",
    previous_context="",
    gold_entities=["Barack Obama", "United States", "President"],
).with_inputs("episode_content", "previous_context")

# Deduplication example
dedup_example = dspy.Example(
    extracted_entities=["Obama", "USA"],
    existing_entities=["Barack Obama", "United States of America", "Joe Biden"],
    gold_deduplication=[
        {"name": "Obama", "is_duplicate": True, "matched_name": "Barack Obama"},
        {"name": "USA", "is_duplicate": True, "matched_name": "United States of America"},
    ],
).with_inputs("extracted_entities", "existing_entities")

# Fact resolution example
resolution_example = dspy.Example(
    new_fact="Barack Obama served as the 44th President",
    related_facts=["Obama was the 44th US President"],
    invalidation_candidates=["George Bush was the 43rd President", "Obama was a senator"],
    gold_resolution={
        "is_duplicate": True,
        "duplicate_of": "Obama was the 44th US President",
        "contradicts": [],
        "fact_type": "SERVED_AS",
    },
).with_inputs("new_fact", "related_facts", "invalidation_candidates")
```

### 7.2 Generating Training Data

```python
def generate_training_examples_from_existing():
    """
    Generate training examples from existing Graphiti runs.
    Captures successful extractions and resolutions.
    """
    examples = []
    
    # Query existing graph for episodes with good results
    episodes = get_successful_episodes()
    
    for episode in episodes:
        # Get the entities that were extracted and persisted
        entities = get_entities_from_episode(episode)
        
        # Create entity extraction example
        example = dspy.Example(
            episode_content=episode.content,
            previous_context=get_context(episode),
            gold_entities=[e.name for e in entities],
        ).with_inputs("episode_content", "previous_context")
        
        examples.append(example)
    
    return examples
```

### 7.3 Recommended Training Set Sizes

| Module | Minimum Examples | Recommended |
|--------|-----------------|-------------|
| Entity Extraction | 50 | 200+ |
| Node Deduplication | 100 | 500+ |
| Fact Extraction | 50 | 200+ |
| Fact Resolution | 100 | 500+ |
| End-to-End Pipeline | 200 | 1000+ |

---

## 8. Migration Path

### 8.1 Phase 1: Parallel Implementation

```
Week 1-2:
├── Create graphiti_core/dspy/ directory
├── Implement Signatures (Section 3)
├── Implement Modules (Section 4)
└── Unit tests for each module
```

### 8.2 Phase 2: Integration

```
Week 3-4:
├── Wire modules into existing pipeline
├── Replace LLM calls one at a time:
│   ├── extract_nodes.py → EntityExtractionModule
│   ├── node dedup in node_operations.py → NodeDeduplicationModule
│   ├── extract_edges.py → FactExtractionModule
│   └── edge resolution in edge_operations.py → FactResolutionModule
└── Verify functional equivalence
```

### 8.3 Phase 3: Optimization

```
Week 5-6:
├── Generate training data from successful runs
├── Define evaluation metrics
├── Run MIPROv2 optimization
├── Benchmark optimized vs unoptimized
└── Tune hyperparameters
```

### 8.4 Phase 4: Cleanup

```
Week 7-8:
├── Remove legacy prompt code
├── Update documentation
├── Performance tuning
└── Production deployment
```

---

## 9. Key Design Decisions

### 9.1 Name-Based vs Index-Based References

**Current (Problematic):**
```python
# LLM returns: entity_id: 2
# But which list? extracted? existing? 0-indexed?
```

**DSPy (Clear):**
```python
class DeduplicationDecision(BaseModel):
    extracted_entity_name: str  # "Obama"
    matched_existing_name: str  # "Barack Obama"
    # No ambiguous indices!
```

### 9.2 Explicit Duplicate Detection

**Current (Implicit):**
```python
# duplicate_idx: -1 means no match (magic number)
```

**DSPy (Explicit):**
```python
class DeduplicationDecision(BaseModel):
    is_duplicate: bool  # Clear boolean
    matched_existing_name: Optional[str]  # None if not duplicate
```

### 9.3 Search as External Input

**Principle:** Search is NOT an LLM concern. DSPy modules receive search results as inputs.

```python
# DON'T do this:
class BadModule(dspy.Module):
    def forward(self, entity_name):
        # Wrong: Module calling search internally
        candidates = search(entity_name)
        return self.predict(entity_name, candidates)

# DO this:
class GoodModule(dspy.Module):
    def forward(self, entity_name, candidates):  # Candidates passed in
        return self.predict(entity_name, candidates)
```

### 9.4 ChainOfThought for Complex Reasoning

Use `dspy.ChainOfThought` (not `dspy.Predict`) for:
- Entity extraction (context reasoning)
- Deduplication (similarity reasoning)
- Fact resolution (contradiction reasoning)

Use `dspy.Predict` for:
- Simple transformations
- Summarization
- Type classification

### 9.5 Pydantic Output Validation

All outputs use Pydantic models for:
- Automatic JSON schema enforcement
- Type validation
- Default values
- Field descriptions (used in prompts)

```python
class ExtractedEntity(BaseModel):
    name: str = Field(description="...")  # Description helps LLM
    entity_type: str = Field(default="Entity", description="...")  # Default for robustness
```

---

## Appendix A: File Structure

```
graphiti_core/
├── dspy/                          # NEW: DSPy implementation
│   ├── __init__.py
│   ├── signatures/
│   │   ├── __init__.py
│   │   ├── entity_extraction.py   # ExtractEntitiesSignature
│   │   ├── node_deduplication.py  # DeduplicateNodesSignature
│   │   ├── fact_extraction.py     # ExtractFactsSignature
│   │   ├── fact_resolution.py     # ResolveFactSignature
│   │   └── summarization.py       # SummarizeEntitySignature
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── entity_extraction.py   # EntityExtractionModule
│   │   ├── node_deduplication.py  # NodeDeduplicationModule
│   │   ├── fact_extraction.py     # FactExtractionModule
│   │   ├── fact_resolution.py     # FactResolutionModule
│   │   └── pipeline.py            # GraphitiDSPyPipeline
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── entity_metrics.py
│   │   ├── dedup_metrics.py
│   │   ├── fact_metrics.py
│   │   └── pipeline_metrics.py
│   └── optimization/
│       ├── __init__.py
│       ├── training_data.py       # Data generation utilities
│       └── optimize.py            # MIPROv2 optimization scripts
├── utils/maintenance/
│   ├── node_operations.py         # MODIFIED: Uses DSPy modules
│   └── edge_operations.py         # MODIFIED: Uses DSPy modules
└── search/                         # UNCHANGED
```

---

## Appendix B: Quick Reference

### DSPy Concepts Used

| Concept | Usage in Graphiti |
|---------|-------------------|
| `dspy.Signature` | Define I/O schema for each pipeline stage |
| `dspy.InputField` | Episode content, entities, search results |
| `dspy.OutputField` | Extracted entities, dedup decisions, facts |
| `dspy.ChainOfThought` | Complex reasoning (extraction, dedup) |
| `dspy.Predict` | Simple transformations (summarization) |
| `MIPROv2` | End-to-end prompt optimization |
| `Pydantic BaseModel` | Structured typed outputs |

### Key Differences from Current Implementation

| Aspect | Current | DSPy |
|--------|---------|------|
| Prompt definition | String templates | Declarative Signatures |
| Output parsing | Manual JSON extraction | Automatic Pydantic |
| Retry logic | Reflexion loops | Automatic backtracking |
| Optimization | Manual prompt tuning | MIPROv2 automated |
| Error handling | Try/except + retry | dspy.Assert/Refine |
| Entity references | Fragile indices | Robust names |

---

*Document generated for Graphiti DSPy reimplementation. Last updated: December 2024*

