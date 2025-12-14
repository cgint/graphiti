# Request (20251114_203536)

```
what is the heart of this - where is the core logic of how the graph is built from add_episode - not only within add_episode but also in the called methods - please do a deep search in the codebase
```

# Response


Of course. Let's break down the core logic of how the graph is built, starting from `add_episode` and tracing the process through the codebase.

The heart of the system is a multi-stage, LLM-driven pipeline that transforms unstructured text from an "episode" into a structured, deduplicated knowledge graph. The entire process is orchestrated by the `Graphiti.add_episode` method in `graphiti_core/graphiti.py`.

Here is a deep dive into the core logic, broken down into four main phases:

### Phase 1: Extraction (Unstructured Text -> Structured Candidates)

This phase is the initial conversion of raw text into potential nodes and edges.

1.  **Node Extraction**:
    *   **Entry Point**: `add_episode` calls `extract_nodes` in `graphiti_core/utils/maintenance/node_operations.py`.
    *   **Core Logic**: This function makes the first key LLM call. Based on the `EpisodeType` (e.g., `message`, `text`), it uses a specific prompt from `graphiti_core/prompts/extract_nodes.py` (like `extract_message`).
    *   **LLM Task**: The prompt instructs the LLM to read the current message (with previous messages as context) and a list of defined `ENTITY TYPES`, and then extract all significant entities. The LLM returns a list of entity names and their classified type.
    *   **Reflexion Loop**: The system has a self-correction mechanism. It may call `extract_nodes_reflexion` to ask the LLM if it missed any entities. If so, it re-runs the extraction with an additional instruction to include the missed ones.
    *   **Output**: A list of `EntityNode` objects, each with a temporary UUID.

2.  **Edge Extraction**:
    *   **Entry Point**: `add_episode` calls `extract_edges` in `graphiti_core/utils/maintenance/edge_operations.py`.
    *   **Core Logic**: This is the second key LLM call. It uses the `prompt_library.extract_edges.edge` prompt from `graphiti_core/prompts/extract_edges.py`.
    *   **LLM Task**: The prompt provides the LLM with the message content, the list of nodes extracted in the previous step (with temporary IDs), and a schema for desired `FACT TYPES`. The LLM's job is to identify relationships between the provided entities and format them as triplets: `(source_entity_id, relation_type, target_entity_id)`, along with a natural language `fact` describing the relationship.
    *   **Output**: A list of `EntityEdge` objects, linking the temporary `EntityNode` UUIDs.

### Phase 2: Resolution & Deduplication (Ensuring Graph Consistency)

This is the most critical phase. It prevents the graph from filling up with duplicate information by intelligently merging new data with existing data.

1.  **Node Resolution (Deduplication)**:
    *   **Entry Point**: `add_episode` calls `resolve_extracted_nodes` in `graphiti_core/utils/maintenance/node_operations.py`.
    *   **Core Logic**: This function's goal is to determine if the newly extracted nodes are actually new or if they refer to nodes already in the graph. It's a hybrid approach:
        *   **Search**: First, it performs a hybrid (keyword + vector) search (`NODE_HYBRID_SEARCH_RRF`) to find existing nodes in the database that are potential candidates for duplication.
        *   **Deterministic Heuristics**: It uses fast, non-LLM methods in `graphiti_core/utils/maintenance/dedup_helpers.py` (`_resolve_with_similarity`) to handle obvious cases. This includes exact name matches and fuzzy string matching (using MinHash and LSH) for names that are distinct and specific enough ("high entropy").
        *   **LLM Escalation**: For ambiguous cases that heuristics can't solve, it escalates to an LLM call using the `prompt_library.dedupe_nodes.nodes` prompt. The LLM is given the new entities and the list of potential existing duplicates and is asked to make the final judgment, returning which new entities map to which existing ones.
    *   **Output**: A final list of `EntityNode` objects (some new, some pre-existing) and a `uuid_map` that maps all temporary new node UUIDs to their final, canonical UUIDs.

2.  **Edge Resolution (Deduplication & Invalidation)**:
    *   **Entry Point**: `add_episode` calls `resolve_extracted_edges` in `graphiti_core/utils/maintenance/edge_operations.py`.
    *   **Core Logic**: After updating the new edges with the canonical node UUIDs from the previous step, this function resolves the edges.
        *   It calls `resolve_extracted_edge` for each new edge.
        *   This function first searches for existing edges that could be duplicates or contradictions.
        *   It then makes a crucial LLM call using the `prompt_library.dedupe_edges.resolve_edge` prompt. This prompt is a multi-task powerhouse:
            1.  **Duplicate Detection**: It asks the LLM if the new fact is semantically identical to any existing facts between the same two nodes. If so, the existing edge is re-used and updated instead of creating a new one.
            2.  **Fact Type Classification**: It asks the LLM to classify the new fact into one of the predefined `FACT TYPES` (if any apply).
            3.  **Contradiction Detection**: It asks the LLM if the new fact contradicts or invalidates any other existing facts.
    *   **Output**: A list of `resolved_edges` (new or updated edges to be saved) and a list of `invalidated_edges` (existing edges that are now considered outdated and will be marked as such).

### Phase 3: Hydration & Enrichment (Adding Depth)

With the canonical nodes and edges identified, this phase enriches them with more detail.

1.  **Attribute & Summary Extraction**:
    *   **Entry Point**: `add_episode` calls `extract_attributes_from_nodes` in `graphiti_core/utils/maintenance/node_operations.py`.
    *   **Core Logic**: This function iterates through the resolved nodes.
        *   **Attributes**: If an entity has a custom type defined by a Pydantic model, an LLM is called with the `prompt_library.extract_nodes.extract_attributes` prompt to populate the fields of that model from the text.
        *   **Summaries**: An LLM is called with the `prompt_library.extract_nodes.extract_summary` prompt. It's given the new text and the node's existing summary, and it generates a new, updated summary.
2.  **Embedding Generation**:
    *   Throughout the process (e.g., in `add_nodes_and_edges_bulk`), any new nodes or edges that don't have embeddings will have them generated via the configured `EmbedderClient`. This is done for entity names (`create_entity_node_embeddings`) and edge facts (`create_entity_edge_embeddings`).

### Phase 4: Persistence (Saving to the Graph)

This is the final step where all the processed data is written to the database.

1.  **Build Final Object List**:
    *   `add_episode` calls `_process_episode_data`.
    *   `build_episodic_edges`: This creates the simple `MENTIONS` relationships connecting the `EpisodicNode` to the `EntityNode`s that were mentioned in it.
2.  **Commit to Database**:
    *   **Entry Point**: `add_nodes_and_edges_bulk` in `graphiti_core/utils/bulk_utils.py`.
    *   **Core Logic**: This function orchestrates a single database transaction (`add_nodes_and_edges_bulk_tx`) to save all the artifacts created in the previous steps. It uses database-specific bulk `MERGE` queries defined in `graphiti_core/models/nodes/node_db_queries.py` and `graphiti_core/models/edges/edge_db_queries.py`.
    *   **What's Saved**:
        *   The new `EpisodicNode`.
        *   The new `EpisodicEdge`s (`MENTIONS`).
        *   All new and updated `EntityNode`s.
        *   All new, updated, and invalidated `EntityEdge`s.

In summary, the core logic is a sophisticated assembly line that uses LLMs as specialized workers at key stages—extraction, classification, deduplication, and summarization—while using deterministic code and database queries to structure the workflow and persist the results.

## Token Usage

🔢 **Model**: gemini-2.5-pro

📊 Token Usage
  ├─ Prompt:    151364
  ├─ Response:  1936
  ├─ Thoughts:  2314
  └─ Total:     155614

## Generated Files

* Context: .codegiant/20251114_203536_codegiant_context.md
* Raw Output: .codegiant/20251114_203536_codegiant_llm_raw_output.json
* Response: .codegiant/20251114_203536_codegiant_llm_response.md

