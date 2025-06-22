# Directory Structure
_Includes files where the actual content might be omitted. This way the LLM can still use the file structure to understand the project._
```
.
├── app.py
├── diagnose_neo4j.py
├── download_models.py
├── example_logging_client.py
├── local_graphiti_utils
│   ├── __init__.py
│   └── utils
│       ├── __init__.py
│       ├── maintenance
│       │   ├── __init__.py
│       │   ├── graph_data_operations.py
│       │   └── node_operations.py
│       └── text_splitters.py
├── logging_llm_client.py
├── ollama_embedder.py
├── ollama_reranker_client.py
├── runpod_openai_client.py
├── runpod_openai_embedder.py
├── session_log_file_writer.py
├── standalone_tests
│   ├── __init__.py
│   ├── standalone_batteries_test.py
│   └── test_all_readonly_endpoints.py
├── test_concurrency_demo.py
├── test_max_tokens_failure.py
├── test_runpod_real.py
├── test_runpod_wrappers.py
├── test_search_power.py
├── test_vertex_improvements.py
└── vertex_ai_client.py
```

# File Contents

## File: `app.py`
```
import os
import asyncio
from typing import Any, List, Optional, cast, AsyncGenerator
from typing_extensions import TypedDict, LiteralString
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from time import time

from graphiti_core.graphiti import Graphiti, AddEpisodeResults
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.embedder.openai import OpenAIEmbedderConfig
from ollama_embedder import NonBatchingOllamaEmbedder
from dotenv import load_dotenv
import traceback
from vertex_ai_client import VertexAIClient
from graphiti_core.driver.driver import GraphDriver
from graphiti_core.graph_queries import get_fulltext_indices, get_range_indices
from graphiti_core.helpers import DEFAULT_DATABASE
from session_log_file_writer import SessionLogFileWriter

# Load environment variables
load_dotenv()

def filter_embedding_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    """
    Filter out embedding attributes from a dictionary to prevent exposing
    large embedding vectors in API responses.
    
    Args:
        attributes: Dictionary potentially containing embedding data
        
    Returns:
        Filtered dictionary with all keys ending in '_embedding' removed
    """
    if not attributes:
        return {}
    
    return {
        key: value for key, value in attributes.items() 
        if not key.endswith('_embedding')
    }

# Neo4j configuration from environment
neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:8745")
neo4j_user = os.getenv("NEO4J_USER", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD", "demodemo")



def create_graphiti_client(session_log_writer: Optional[SessionLogFileWriter] = None) -> Graphiti:
    # Use Vertex AI Client
    llm_client = VertexAIClient(
        config=LLMConfig(
            # Use a valid Vertex AI model name
            model="gemini-2.5-flash-lite-preview-06-17",
            temperature=0.3,
            max_tokens=65500,
        ),
        session_log_writer=session_log_writer
    )

    # Using Ollama for embeddings as before.
    # For a full GCP setup, you might switch to a Vertex AI embedder.
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    print(f"🔗 Using Ollama embedder at: {ollama_base_url}")
    
    embedder = NonBatchingOllamaEmbedder(
        config=OpenAIEmbedderConfig(
            api_key="ollama",
            base_url=ollama_base_url,
            embedding_model="nomic-embed-text"
        )
    )

    # Use Ollama cross-encoder for faster startup (no model download needed)
    from ollama_reranker_client import OllamaRerankerClient
    cross_encoder = OllamaRerankerClient(
        base_url=ollama_base_url.replace('/v1', ''),
        model_name="bge-m3"  # Use BGE-M3 for better embeddings
    )

    # Initialize Graphiti with Vertex AI client and Ollama cross-encoder
    print(f"Initializing Graphiti with Neo4j URI: {neo4j_uri}, Vertex AI Client, and Ollama Cross-Encoder")
    return Graphiti(
        neo4j_uri,
        neo4j_user,
        neo4j_password,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder
    )

async def build_indices_sequentially(driver: GraphDriver, delete_existing: bool = False) -> None:
    """
    Builds indices sequentially to avoid deadlocks.
    """
    if delete_existing:
        try:
            # Note: SHOW INDEXES is a Neo4j-specific command. This may need adjustment for other DBs.
            records, _, _ = await driver.execute_query(
                "SHOW INDEXES YIELD name",
                database_=DEFAULT_DATABASE,
            )
            index_names = [record['name'] for record in records if not record['name'].startswith('constraint_')]
            print(f"Dropping existing indexes: {index_names}")
            for name in index_names:
                print(f"Dropping index: {name}")
                # Use DROP INDEX ... IF EXISTS to avoid errors if index is already gone
                await driver.execute_query(
                    f"DROP INDEX `{name}` IF EXISTS",
                    database_=DEFAULT_DATABASE,
                )
        except Exception as e:
            # Log the error but continue, as this might be expected on a clean DB or other backends
            print(f"Could not drop indexes (this might be expected on some backends): {e}")

    range_indices: list[LiteralString] = get_range_indices(driver.provider)
    fulltext_indices: list[LiteralString] = get_fulltext_indices(driver.provider)
    index_queries: list[LiteralString] = range_indices + fulltext_indices

    print("Building indices sequentially...")
    for query in index_queries:
        try:
            print(f"Running: {query}")
            await driver.execute_query(
                query,
                database_=DEFAULT_DATABASE,
            )
        except Exception as e:
            # It's common for index creation to fail if it already exists. We can safely ignore this.
            print(f"Could not create index (may already exist): {query}. Error: {e}")
    print("Finished building indices.")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Build indices and constraints for Neo4j sequentially
    # Add timeout and proper error handling to prevent hanging on Ctrl-C
    try:
        # Set a reasonable timeout for the index building
        await asyncio.wait_for(
            build_indices_sequentially(create_graphiti_client().driver, delete_existing=True), 
            timeout=10.0
        )
        print("Graphiti indices and constraints built successfully on startup.")
    except asyncio.TimeoutError:
        print("⚠️  Timeout building indices (likely Neo4j not available) - continuing without indices...")
    except Exception as e:
        print(f"⚠️  Failed to build indices (likely Neo4j not available): {e} - continuing without indices...")
    
    yield
    
    # Shutdown
    # Add any cleanup code here if needed
    pass

app = FastAPI(lifespan=lifespan)

# --- Request and Response Models (from example_graphiti_mcp_server_simple.py) ---

class Requirement(BaseModel):
    project_name: str = Field(
        ...,
        description="The name of the project to which the requirement belongs."
    )
    description: str = Field(
        ...,
        description="Description of the requirement. Only use information mentioned in the context to write this description."
    )

class Preference(BaseModel):
    category: str = Field(
        ...,
        description="The category of the preference. (e.g., 'Brands', 'Food', 'Music')"
    )
    description: str = Field(
        ...,
        description="Brief description of the preference. Only use information mentioned in the context to write this description."
    )

class Procedure(BaseModel):
    description: str = Field(
        ...,
        description="Brief description of the procedure. Only use information mentioned in the context to write this description."
    )

ENTITY_TYPES = {
    'Requirement': Requirement,
    'Preference': Preference,
    'Procedure': Procedure,
}

# Type definitions for API responses
class ErrorResponse(TypedDict):
    error: str

class SuccessAddMemoryResponse(TypedDict):
    message: str
    nodes_added: int
    edges_added: int

class SuccessResponse(TypedDict):
    message: str

class NodeResult(TypedDict):
    uuid: str
    name: str
    summary: str
    labels: List[str]
    group_id: str
    created_at: str
    attributes: dict[str, Any]

class NodeSearchResponse(TypedDict):
    message: str
    nodes: List[NodeResult]

class FactSearchResponse(TypedDict):
    message: str
    facts: List[dict[str, Any]]

class AddMemoryRequest(BaseModel):
    name: str
    episode_body: str
    group_id: Optional[str] = None
    source: str = "text"
    source_description: str = ""
    uuid: Optional[str] = None

def get_new_graphiti_client_for_request() -> Graphiti:
    try:
        return create_graphiti_client()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_memory", response_model=SuccessAddMemoryResponse)
async def add_memory_endpoint(
    request: AddMemoryRequest,
) -> SuccessAddMemoryResponse:
    try:
        graphiti: Graphiti = create_graphiti_client()
        start_time = time()
        request_size = len(str(request))
        print(f"\nAdding memory - size: {request_size} - {str(request)[:500]}")
        # Here we need to replicate the logic from example_graphiti_mcp_server_simple.py's add_memory
        # This involves handling the queue and calling graphiti_client.add_episode
        # For simplicity in this direct exposure, we'll call add_episode directly.
        # In a real-world scenario, you might want to integrate a proper background task queue.
        from datetime import datetime, timezone
        from graphiti_core.nodes import EpisodeType

        source_type = EpisodeType.text
        if request.source.lower() == 'message':
            source_type = EpisodeType.message
        elif request.source.lower() == 'json':
            source_type = EpisodeType.json

        effective_group_id = request.group_id if request.group_id is not None else "default" # Assuming "default" if not provided, consistent with example_graphiti_mcp_server_simple.py's config.group_id fallback
        entity_types = cast(dict[str, BaseModel], ENTITY_TYPES) # Assuming custom entities are always enabled for simplicity here, or controlled by an env var.

        res: AddEpisodeResults = await graphiti.add_episode(
            name=request.name,
            episode_body=request.episode_body,
            source=source_type,
            source_description=request.source_description,
            group_id=effective_group_id,
            uuid=request.uuid,
            reference_time=datetime.now(timezone.utc),
            entity_types=entity_types,
        )
        duration = time() - start_time
        print(f"\n🔧 AddEpisodeResults: Episode '{request.name}' added successfully. Extracted {len(res.nodes)} nodes and {len(res.edges)} edges. Took {duration:.2f} seconds for request size: {request_size}\n")
        return {"message": f"Episode '{request.name}' added successfully", "nodes_added": len(res.nodes), "edges_added": len(res.edges)}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search_memory_nodes", response_model=NodeSearchResponse)
async def search_memory_nodes_endpoint(
    query: str,
    group_ids: Optional[List[str]] = None,
    max_nodes: int = 10,
    center_node_uuid: Optional[str] = None,
    entity: str = ""
) -> NodeSearchResponse:
    try:
        graphiti: Graphiti = create_graphiti_client()
        print("=== ENHANCED NODE SEARCH DEBUG ===")
        print(f"Query: '{query}'")
        print(f"Group IDs: {group_ids}")
        print(f"Max nodes: {max_nodes}")
        
        # Test direct database queries first - disabled due to linter issues
        print("Direct database queries disabled due to type issues, proceeding with main search...")
        
        # Now test different search configurations
        from graphiti_core.search.search_config import NodeSearchConfig, NodeSearchMethod, SearchConfig
        from graphiti_core.search.search_filters import SearchFilters

        effective_group_ids = group_ids if group_ids is not None else ["chrome-tracker-content"]

        # 🚀 BATTERIES INCLUDED HYBRID SEARCH - Avoiding cross-encoder issues
        # Based on analysis: cross-encoder fails with Ollama, but RRF reranking works
        from graphiti_core.search.search_config import (
            NodeReranker, EdgeSearchMethod, 
            CommunitySearchMethod, EdgeSearchConfig, CommunitySearchConfig, EpisodeSearchConfig,
            EdgeReranker, CommunityReranker, EpisodeReranker, EpisodeSearchMethod
        )
        
        # Create a robust hybrid configuration with Ollama cross-encoder
        search_config = SearchConfig(
            node_config=NodeSearchConfig(
                search_methods=[NodeSearchMethod.bm25, NodeSearchMethod.cosine_similarity],
                reranker=NodeReranker.cross_encoder,  # Now using Ollama cross-encoder!
                sim_min_score=0.2,  # Very permissive for vector similarity
            ),
            edge_config=EdgeSearchConfig(
                search_methods=[EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
                reranker=EdgeReranker.cross_encoder,  # Ollama cross-encoder reranking
                sim_min_score=0.2,  # Lower threshold for better recall
            ),
            episode_config=EpisodeSearchConfig(
                search_methods=[EpisodeSearchMethod.bm25],
                reranker=EpisodeReranker.cross_encoder,  # Ollama cross-encoder
            ),
            community_config=CommunitySearchConfig(
                search_methods=[CommunitySearchMethod.bm25, CommunitySearchMethod.cosine_similarity],
                reranker=CommunityReranker.cross_encoder,  # Ollama cross-encoder
                sim_min_score=0.2,  # Permissive similarity threshold
            ),
            limit=max_nodes,
            reranker_min_score=0.0  # Accept all reranked results
        )
        
        print("🚀 Using BATTERIES INCLUDED hybrid search (BM25 + Vector, Ollama Cross-Encoder)")
        print("   - Node methods: BM25 + cosine_similarity")
        print("   - Edge methods: BM25 + cosine_similarity") 
        print("   - Reranker: Ollama Cross-Encoder (fast startup!)")
        print("   - Similarity threshold: 0.2 (permissive)")
        print("   - Reranker threshold: 0.0 (accept all)")

        filters = SearchFilters()
        if entity != '':
            filters.node_labels = [entity]

        print(f"Search config: {search_config}")
        print(f"Group IDs: {effective_group_ids}")
        print(f"Filters: {filters}")
                
        search_results = await graphiti.search_(
            query=query,
            config=search_config,
            group_ids=effective_group_ids,
            center_node_uuid=center_node_uuid,
            search_filter=filters,
        )

        # print(f"Search results: {search_results}")
        print(f"Number of nodes found: {len(search_results.nodes) if search_results.nodes else 0}")

        if not search_results.nodes:
            return {"message": "No relevant nodes found", "nodes": []}

        formatted_nodes: List[NodeResult] = [
            {
                'uuid': node.uuid,
                'name': node.name,
                'summary': node.summary if hasattr(node, 'summary') else '',
                'labels': node.labels if hasattr(node, 'labels') else [],
                'group_id': node.group_id,
                'created_at': node.created_at.isoformat(),
                'attributes': filter_embedding_attributes(node.attributes) if hasattr(node, 'attributes') else {},
            }
            for node in search_results.nodes
        ]

        return {"message": "Nodes retrieved successfully", "nodes": formatted_nodes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search_memory_facts", response_model=FactSearchResponse)
async def search_memory_facts_endpoint(
    query: str,
    group_ids: Optional[List[str]] = None,
    max_facts: int = 10,
    center_node_uuid: Optional[str] = None
) -> FactSearchResponse:
    try:
        graphiti: Graphiti = create_graphiti_client()
        print("=== ENHANCED FACTS SEARCH DEBUG ===")
        print(f"Query: '{query}'")
        print(f"Group IDs: {group_ids}")
        print(f"Max facts: {max_facts}")
        
        # Test direct database queries first - disabled due to linter issues  
        print("Direct database queries disabled due to type issues, proceeding with main search...")
        
        # Now test search configuration with low thresholds
        from graphiti_core.search.search_config import EdgeSearchConfig, EdgeSearchMethod, SearchConfig
        from graphiti_core.search.search_filters import SearchFilters
        from graphiti_core.search.search_config import (
            NodeSearchConfig, NodeSearchMethod, NodeReranker,
            EdgeReranker, CommunitySearchMethod,
            CommunitySearchConfig, CommunityReranker, EpisodeSearchConfig,
            EpisodeReranker, EpisodeSearchMethod
        )

        effective_group_ids = group_ids if group_ids is not None else ["chrome-tracker-content"]

        # 🚀 BATTERIES INCLUDED HYBRID SEARCH with Ollama cross-encoder

        # Create a robust hybrid configuration with Ollama cross-encoder  
        search_config = SearchConfig(
            node_config=NodeSearchConfig(
                search_methods=[NodeSearchMethod.bm25, NodeSearchMethod.cosine_similarity],
                reranker=NodeReranker.cross_encoder,  # Ollama cross-encoder
                sim_min_score=0.2,  # Very permissive for vector similarity
            ),
            edge_config=EdgeSearchConfig(
                search_methods=[EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
                reranker=EdgeReranker.cross_encoder,  # Ollama cross-encoder
                sim_min_score=0.2,  # Very permissive for vector similarity
            ),
            episode_config=EpisodeSearchConfig(
                search_methods=[EpisodeSearchMethod.bm25],
                reranker=EpisodeReranker.cross_encoder,  # Ollama cross-encoder
            ),
            community_config=CommunitySearchConfig(
                search_methods=[CommunitySearchMethod.bm25, CommunitySearchMethod.cosine_similarity],
                reranker=CommunityReranker.cross_encoder,  # Ollama cross-encoder
                sim_min_score=0.2,  # Permissive similarity threshold
            ),
            limit=max_facts,
            reranker_min_score=0.0  # Accept all reranked results
        )
        
        print("🚀 Using BATTERIES INCLUDED hybrid FACTS search (BM25 + Vector, Ollama Cross-Encoder)")
        print("   - Edge methods: BM25 + cosine_similarity")
        print("   - Episode methods: BM25")
        print("   - Reranker: Ollama Cross-Encoder (fast startup!)")
        print("   - Similarity threshold: 0.2 (permissive)")
        print("   - Reranker threshold: 0.0 (accept all)")

        filters = SearchFilters()

        print(f"Search config: {search_config}")
        print(f"Group IDs: {effective_group_ids}")
        print(f"Filters: {filters}")

        search_results = await graphiti.search_(
            query=query,
            config=search_config,
            group_ids=effective_group_ids,
            center_node_uuid=center_node_uuid,
            search_filter=filters,
        )

        print(f"Number of edges found: {len(search_results.edges) if search_results.edges else 0}")

        if not search_results.edges:
            return {"message": "No relevant facts found", "facts": []}

        # Extract unique node UUIDs from edges to look up their names
        node_uuids = set()
        for edge in search_results.edges:
            if hasattr(edge, 'source_node_uuid') and edge.source_node_uuid:
                node_uuids.add(edge.source_node_uuid)
            if hasattr(edge, 'target_node_uuid') and edge.target_node_uuid:
                node_uuids.add(edge.target_node_uuid)

        # Look up node names using a search-based approach
        node_names = {}
        if node_uuids:
            try:
                # Extract node names from any nodes that might be in our search results
                print(f"🔍 Looking up names for {len(node_uuids)} unique nodes from search results")
                
                # First, check if any of the nodes from our search have the UUIDs we need
                if search_results.nodes:
                    for node in search_results.nodes:
                        if hasattr(node, 'uuid') and hasattr(node, 'name') and node.uuid in node_uuids:
                            node_names[node.uuid] = node.name
                
                # For remaining UUIDs, use direct node lookup by UUIDs
                remaining_uuids = node_uuids - set(node_names.keys())
                if remaining_uuids:
                    print(f"🔍 Direct lookup for {len(remaining_uuids)} remaining nodes by UUID")
                    
                    try:
                        # Use the EntityNode.get_by_uuids method for direct lookup
                        from graphiti_core.nodes import EntityNode
                        
                        uuid_list = list(remaining_uuids)
                        entity_nodes = await EntityNode.get_by_uuids(graphiti.driver, uuid_list)
                        
                        # Map the results
                        for node in entity_nodes:
                            if hasattr(node, 'uuid') and hasattr(node, 'name'):
                                node_names[node.uuid] = node.name
                                remaining_uuids.discard(node.uuid)
                        
                        print(f"🔍 Direct lookup found {len(entity_nodes)} nodes")
                        
                    except Exception as e:
                        print(f"⚠️ Direct lookup failed: {e}")
                        traceback.print_exc()
                    
                    # For any still remaining UUIDs, use UUID fallback
                    for uuid in remaining_uuids:
                        node_names[uuid] = f"Node-{uuid[:8]}"
                
                print(f"🔍 Successfully looked up names for {len(node_names)} out of {len(node_uuids)} nodes")
                
            except Exception as e:
                print(f"⚠️ Warning: Could not look up node names: {e}")
                # Continue without node names if lookup fails

        def format_fact_result(edge: Any) -> dict[str, Any]:
            source_uuid = edge.source_node_uuid if hasattr(edge, 'source_node_uuid') else ''
            target_uuid = edge.target_node_uuid if hasattr(edge, 'target_node_uuid') else ''
            
            return {
                'uuid': edge.uuid,
                'name': edge.name if hasattr(edge, 'name') else '',
                'fact': edge.fact if hasattr(edge, 'fact') else '',
                'source_node_uuid': source_uuid,
                'target_node_uuid': target_uuid,
                'source_node_name': node_names.get(source_uuid, 'Unknown'),
                'target_node_name': node_names.get(target_uuid, 'Unknown'),
                'group_id': edge.group_id,
                'created_at': edge.created_at.isoformat() if hasattr(edge, 'created_at') else '',
                'episodes': edge.episodes if hasattr(edge, 'episodes') else [],
                'attributes': filter_embedding_attributes(edge.attributes) if hasattr(edge, 'attributes') else {},
            }

        facts = [format_fact_result(edge) for edge in search_results.edges]
        return {"message": "Facts retrieved successfully", "facts": facts}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear_graph", response_model=SuccessResponse)
async def clear_graph_endpoint() -> SuccessResponse:
    try:
        graphiti: Graphiti = create_graphiti_client()
        from local_graphiti_utils.utils.maintenance.graph_data_operations import clear_data

        await clear_data(graphiti.driver)
        await build_indices_sequentially(graphiti.driver, delete_existing=True)
        return {"message": "Graph cleared successfully and indices rebuilt"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rebuild_indices", response_model=SuccessResponse)
async def rebuild_indices_endpoint() -> SuccessResponse:
    try:
        graphiti: Graphiti = create_graphiti_client()
        # The `delete_existing=True` parameter will drop existing indexes before creating new ones.
        await build_indices_sequentially(graphiti.driver, delete_existing=True)
        return {"message": "Graph indices rebuilt successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

```

## File: `diagnose_neo4j.py`
```
import asyncio
from neo4j import AsyncGraphDatabase
import os

# Neo4j configuration from environment
neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:8745")
neo4j_user = os.getenv("NEO4J_USER", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD", "demodemo")
DEFAULT_DATABASE = os.getenv("DEFAULT_DATABASE", "neo4j")

async def diagnose_neo4j() -> None:
    driver = None
    try:
        driver = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        await driver.verify_connectivity()
        print("Successfully connected to Neo4j.")

        async with driver.session(database=DEFAULT_DATABASE) as session:
            # Query 1: Check existing indexes
            print("\n--- Query 1: SHOW INDEXES ---")
            result = await session.run("SHOW INDEXES")
            records = await result.data()
            for record in records:
                print(record)

            # Query 2: Check if data and embeddings exist
            print("\n--- Query 2: Node data and embeddings ---")
            query_nodes = """
            MATCH (n:Entity {group_id: 'chrome-tracker-content'})
            RETURN n.name, n.summary, n.name_embedding IS NOT NULL AS has_embedding
            LIMIT 10;
            """
            result = await session.run(query_nodes)
            records = await result.data()
            for record in records:
                print(record)
            if not records:
                print("No entity nodes found with group_id 'chrome-tracker-content'.")

            # Query 3: Test the full-text index directly
            print("\n--- Query 3: Direct full-text query ---")
            query_fulltext = """
            CALL db.index.fulltext.queryNodes('node_name_and_summary', 'AI tools') YIELD node, score
            RETURN node.name, score;
            """
            result = await session.run(query_fulltext)
            records = await result.data()
            for record in records:
                print(record)
            if not records:
                print("Full-text query 'AI tools' returned no results.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if driver:
            await driver.close()
            print("Neo4j driver closed.")

def main() -> None:
    asyncio.run(diagnose_neo4j())

if __name__ == "__main__":
    main() ```

## File: `download_models.py`
```
#!/usr/bin/env python3
"""
Pre-download BGE models for Docker builds.
This script downloads and caches BGE models locally within the project structure.
"""

import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_bge_models():
    """Download BGE models to local models directory."""
    
    # Set up models directory
    project_root = Path(__file__).parent
    models_dir = project_root / "models" / "bge"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Set HuggingFace cache to our local directory
    os.environ['HF_HOME'] = str(models_dir)
    
    logger.info(f"📁 Models directory: {models_dir}")
    logger.info("🚀 Starting BGE model download...")
    
    try:
        # Import sentence-transformers and download the model
        from sentence_transformers import CrossEncoder
        
        model_name = 'BAAI/bge-reranker-v2-m3'
        logger.info(f"📥 Downloading {model_name}...")
        
        # This will download and cache the model
        model = CrossEncoder(model_name, cache_folder=str(models_dir))
        
        logger.info(f"✅ Successfully downloaded {model_name}")
        logger.info(f"📁 Model cached in: {models_dir}")
        
        # List downloaded files
        model_files = list(models_dir.rglob("*"))
        logger.info(f"📦 Downloaded {len(model_files)} files")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ sentence-transformers not available: {e}")
        logger.error("Install with: uv add sentence-transformers")
        return False
        
    except Exception as e:
        logger.error(f"❌ Error downloading models: {e}")
        return False

if __name__ == "__main__":
    success = download_bge_models()
    if success:
        print("🎉 BGE models downloaded successfully!")
    else:
        print("❌ Failed to download BGE models")
        exit(1) ```

## File: `example_logging_client.py`
```
#!/usr/bin/env python3
"""
Example usage of the LoggingOpenAIClient wrapper.

This example shows how to use the logging client wrapper for debugging and monitoring.
"""

import asyncio
import logging
from pydantic import BaseModel

from logging_llm_client import LoggingOpenAIClient
from graphiti_core.prompts.models import Message
from graphiti_core.llm_client.config import LLMConfig, ModelSize


# Configure logging to see the client logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class ExampleResponse(BaseModel):
    """Example response model for structured output."""
    summary: str
    key_points: list[str]


async def example_usage() -> None:
    """Example of using the LoggingOpenAIClient."""
    
    # Create configuration (you can also pass None for defaults)
    config = LLMConfig(
        api_key="your-api-key-here",  # or set OPENAI_API_KEY environment variable
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=1000
    )
    
    # Create the logging client
    client = LoggingOpenAIClient(config=config)
    
    # Example messages
    messages = [
        Message(role="system", content="You are a helpful assistant that summarizes text."),
        Message(role="user", content="Please summarize the benefits of using logging in software development.")
    ]
    
    try:
        # Example 1: Basic response without structured output
        print("=== Example 1: Basic Response ===")
        response = await client.generate_response(
            messages=messages,
            max_tokens=500,
            model_size=ModelSize.medium
        )
        print(f"Response: {response}")
        
        # Example 2: Structured response with Pydantic model
        print("\n=== Example 2: Structured Response ===")
        structured_response = await client.generate_response(
            messages=messages,
            response_model=ExampleResponse,
            max_tokens=500,
            model_size=ModelSize.medium
        )
        print(f"Structured Response: {structured_response}")
        
    except Exception as e:
        print(f"Error occurred: {e}")


def main() -> None:
    """Main function to run the example."""
    asyncio.run(example_usage())


if __name__ == "__main__":
    # Run the example
    main() ```

## File: `local_graphiti_utils/__init__.py`
```
# Local graphiti_core utilities
```

## File: `local_graphiti_utils/utils/__init__.py`
```
Content omitted due to reason: BINARY
```

## File: `local_graphiti_utils/utils/maintenance/__init__.py`
```
Content omitted due to reason: BINARY
```

## File: `local_graphiti_utils/utils/maintenance/graph_data_operations.py`
```
"""
Graph data operations utilities for maintenance tasks.
"""

import logging
from typing import Dict, Any
from graphiti_core.driver.driver import GraphDriver
from graphiti_core.helpers import DEFAULT_DATABASE

# Set up logger for this module
logger = logging.getLogger(__name__)



async def clear_data(driver: GraphDriver) -> Dict[str, Any]:
    """
    Clears all data from the graph database in a single atomic transaction.
    
    Args:
        driver: The GraphDriver instance to use for clearing data
        
    Returns:
        Dict containing operation status and details:
        - success: Boolean indicating if operation succeeded
        - message: Description of the result
        - nodes_deleted: Number of nodes deleted (if successful)
        - relationships_deleted: Number of relationships deleted (if successful)
        - error_type: Type of error if operation failed
    """
    try:
        # Use a single DETACH DELETE operation within a transaction for atomicity
        # This automatically handles relationship deletion before node deletion
        result = await driver.execute_query(
            "MATCH (n) DETACH DELETE n RETURN count(n) as nodes_deleted",
            database_=DEFAULT_DATABASE,
        )
        
        # Extract the count from the result
        nodes_deleted = 0
        if result and len(result) > 0:
            nodes_deleted = result[0].get('nodes_deleted', 0)
        
        print(f"Graph data cleared successfully: {nodes_deleted} nodes deleted")
        
        return {
            'success': True,
            'message': f'Successfully cleared graph data: {nodes_deleted} nodes and their relationships deleted',
            'nodes_deleted': nodes_deleted,
            'relationships_deleted': None,  # DETACH DELETE doesn't return relationship count separately
            'error_type': None
        }
        
    except Exception as e:
        # Handle specific database-related exceptions
        error_type = type(e).__name__
        error_message = str(e)
        
        # Log the error with appropriate level
        print(f"ERROR: Failed to clear graph data: {error_type} - {error_message}")
        
        # Determine if this is a database connectivity issue vs other errors
        if 'connection' in error_message.lower() or 'timeout' in error_message.lower():
            print("ERROR: Database connection issue detected during clear operation")
            return {
                'success': False,
                'message': f'Database connection error: {error_message}',
                'nodes_deleted': 0,
                'relationships_deleted': 0,
                'error_type': 'CONNECTION_ERROR'
            }
        elif 'permission' in error_message.lower() or 'authorization' in error_message.lower():
            print("ERROR: Database permission issue detected during clear operation")
            return {
                'success': False,
                'message': f'Database permission error: {error_message}',
                'nodes_deleted': 0,
                'relationships_deleted': 0,
                'error_type': 'PERMISSION_ERROR'
            }
        else:
            print("ERROR: General database error during clear operation")
            return {
                'success': False,
                'message': f'Database operation error: {error_message}',
                'nodes_deleted': 0,
                'relationships_deleted': 0,
                'error_type': 'DATABASE_ERROR'
            } ```

## File: `local_graphiti_utils/utils/maintenance/node_operations.py`
```
"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from contextlib import suppress
from time import time
from typing import Any, cast
import json
from uuid import uuid4

import pydantic
from pydantic import BaseModel, Field

from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.helpers import MAX_REFLEXION_ITERATIONS, semaphore_gather
from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode, create_entity_node_embeddings
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.dedupe_nodes import NodeResolutions
from graphiti_core.prompts.extract_nodes import (
    ExtractedEntities,
    MissedEntities,
)
from graphiti_core.search.search import search
from graphiti_core.search.search_config import SearchResults
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.datetime_utils import utc_now

# Import our text splitting utility
from local_graphiti_utils.utils.text_splitters import split_text_intelligently



# Token limit threshold - when episode content exceeds this, we'll chunk it
MAX_EPISODE_CHARS = 100000  # ~25k tokens for most models


async def extract_nodes_reflexion(
    llm_client: LLMClient,
    episode: EpisodicNode,
    previous_episodes: list[EpisodicNode],
    node_names: list[str],
) -> list[str]:
    # Prepare context for LLM
    context = {
        'episode_content': episode.content,
        'previous_episodes': [ep.content for ep in previous_episodes],
        'extracted_entities': node_names,
    }

    llm_response = await llm_client.generate_response(
        prompt_library.extract_nodes.reflexion(context), response_model=MissedEntities
    )
    missed_entities = llm_response.get('missed_entities', [])

    return cast(list[str], missed_entities)


async def extract_nodes_from_chunk(
    llm_client: LLMClient,
    episode_chunk: str,
    episode: EpisodicNode,
    previous_episodes: list[EpisodicNode],
    entity_types_context: list[dict[str, Any]],
    custom_prompt: str = '',
) -> list[dict[str, Any]]:
    """Extract nodes from a single chunk of episode content."""
    print(f"DEBUG: extract_nodes_from_chunk called with chunk size: {len(episode_chunk)}")
    context = {
        'episode_content': episode_chunk,
        'episode_timestamp': episode.valid_at.isoformat(),
        'previous_episodes': [ep.content for ep in previous_episodes],
        'custom_prompt': custom_prompt,
        'entity_types': entity_types_context,
        'source_description': episode.source_description,
    }

    if episode.source == EpisodeType.message:
        llm_response = await llm_client.generate_response(
            prompt_library.extract_nodes.extract_message(context),
            response_model=ExtractedEntities,
        )
    elif episode.source == EpisodeType.text:
        llm_response = await llm_client.generate_response(
            prompt_library.extract_nodes.extract_text(context), response_model=ExtractedEntities
        )
    elif episode.source == EpisodeType.json:
        llm_response = await llm_client.generate_response(
            prompt_library.extract_nodes.extract_json(context), response_model=ExtractedEntities
        )
    else:
        # Fallback to text extraction for unknown types
        llm_response = await llm_client.generate_response(
            prompt_library.extract_nodes.extract_text(context), response_model=ExtractedEntities
        )

    extracted_entities = llm_response.get('extracted_entities', [])
    print(f"DEBUG: LLM response for chunk - keys: {list(llm_response.keys())}")
    print(f"DEBUG: LLM response for chunk - full: {llm_response}")
    print(f"DEBUG: Extracted entities from chunk: {extracted_entities}")
    print(f"DEBUG: Type of extracted_entities: {type(extracted_entities)}")
    
    # Check if we got a 'content' response instead of structured JSON
    if 'content' in llm_response and not extracted_entities:
        print(f"DEBUG: WARNING - Got unstructured response from LLM. Content: {llm_response['content'][:500]}...")
        # Try to parse JSON from content if possible
        try:
            content_str = llm_response['content']
            if content_str.strip().startswith('{'):
                parsed_content = json.loads(content_str)
                extracted_entities = parsed_content.get('extracted_entities', [])
                print(f"DEBUG: Successfully parsed JSON from content, got {len(extracted_entities)} entities")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"DEBUG: Failed to parse JSON from content: {e}")
    
    return cast(list[dict[str, Any]], extracted_entities)


async def extract_nodes(
    clients: GraphitiClients,
    episode: EpisodicNode,
    previous_episodes: list[EpisodicNode],
    entity_types: dict[str, BaseModel] | None = None,
) -> list[EntityNode]:
    start = time()
    llm_client = clients.llm_client
    custom_prompt = ''
    entities_missed = True
    reflexion_iterations = 0
    all_extracted_entities = []

    entity_types_context = [
        {
            'entity_type_id': 0,
            'entity_type_name': 'Entity',
            'entity_type_description': 'Default entity classification. Use this entity type if the entity is not one of the other listed types.',
        }
    ]

    entity_types_context += (
        [
            {
                'entity_type_id': i + 1,
                'entity_type_name': type_name,
                'entity_type_description': type_model.__doc__,
            }
            for i, (type_name, type_model) in enumerate(entity_types.items())
        ]
        if entity_types is not None
        else []
    )

    # Check if episode content is too large and needs chunking
    print(f"DEBUG: Episode content length: {len(episode.content)}, MAX_EPISODE_CHARS: {MAX_EPISODE_CHARS}")
    if len(episode.content) > MAX_EPISODE_CHARS:
        print(f"Episode content is large ({len(episode.content)} chars), splitting into chunks...")
        content_chunks = split_text_intelligently(episode.content, max_chunk_size=MAX_EPISODE_CHARS)
        print(f"Split into {len(content_chunks)} chunks")
        
        # Process each chunk and collect entities
        chunk_results = await semaphore_gather(
            *[
                extract_nodes_from_chunk(
                    llm_client, chunk, episode, previous_episodes, entity_types_context, custom_prompt
                )
                for chunk in content_chunks
            ]
        )
        
        # Merge results from all chunks
        for chunk_entities in chunk_results:
            print(f"DEBUG: Chunk entities: {chunk_entities}")
            all_extracted_entities.extend(chunk_entities)
            
        print(f"DEBUG: All extracted entities before dedup: {all_extracted_entities}")
        
        # Remove duplicates based on entity name (case-insensitive)
        seen_names = set()
        unique_entities = []
        for entity in all_extracted_entities:
            # entity is a dict, not an object with .name attribute
            name_lower = entity.get('name', '').lower().strip()
            if name_lower and name_lower not in seen_names:
                seen_names.add(name_lower)
                unique_entities.append(entity)
        
        all_extracted_entities = unique_entities
        print(f"Extracted {len(all_extracted_entities)} unique entities from chunked content")
        
    else:
        # Original logic for smaller content
        print(f"DEBUG: Using non-chunked path for content length: {len(episode.content)}")
        while entities_missed and reflexion_iterations <= MAX_REFLEXION_ITERATIONS:
            chunk_entities = await extract_nodes_from_chunk(
                llm_client, episode.content, episode, previous_episodes, entity_types_context, custom_prompt
            )
            print(f"DEBUG: Non-chunked entities: {chunk_entities}")
            all_extracted_entities = chunk_entities
            
            reflexion_iterations += 1
            if reflexion_iterations < MAX_REFLEXION_ITERATIONS:
                missing_entities = await extract_nodes_reflexion(
                    llm_client,
                    episode,
                    previous_episodes,
                    [entity['name'] for entity in all_extracted_entities],
                )

                entities_missed = len(missing_entities) != 0

                custom_prompt = 'Make sure that the following entities are extracted: '
                for entity in missing_entities:
                    custom_prompt += f'\n{entity},'

    filtered_extracted_entities = [entity for entity in all_extracted_entities if entity['name'].strip()]
    end = time()
    print(f'DEBUG: Extracted new nodes: {filtered_extracted_entities} in {(end - start) * 1000} ms')
    
    # Convert the extracted data into EntityNode objects
    extracted_nodes = []
    for extracted_entity in filtered_extracted_entities:
        # Validate entity_type_id and provide fallback
        entity_type_id = extracted_entity.get('entity_type_id', 0)
        if 0 <= entity_type_id < len(entity_types_context):
            entity_type_name = entity_types_context[entity_type_id].get(
                'entity_type_name', 'Entity'  # fallback to 'Entity' if key missing
            )
        else:
            # Log the issue and fallback to default
            print(
                f"WARNING: Invalid entity_type_id {entity_type_id} for entity '{extracted_entity['name']}'. "
                f"Using default 'Entity' type. Available types: {len(entity_types_context)}"
            )
            entity_type_name = 'Entity'
        
        labels: list[str] = list({'Entity', str(entity_type_name)})
        new_node = EntityNode(
            name=extracted_entity['name'],
            group_id=episode.group_id,
            labels=labels,
            summary='',
            created_at=utc_now(),
        )
        extracted_nodes.append(new_node)
        print(f'DEBUG: Created new node: {new_node.name} (UUID: {new_node.uuid})')

    print(f'DEBUG: Extracted nodes: {[(n.name, n.uuid) for n in extracted_nodes]}')
    print(f'DEBUG: Final extracted_nodes count: {len(extracted_nodes)}')
    print(f'DEBUG: All extracted entities before EntityNode creation: {all_extracted_entities}')
    print(f'DEBUG: Filtered extracted entities: {filtered_extracted_entities}')
    return extracted_nodes


async def dedupe_extracted_nodes(
    llm_client: LLMClient,
    extracted_nodes: list[EntityNode],
    existing_nodes: list[EntityNode],
) -> tuple[list[EntityNode], dict[str, str]]:
    start = time()

    # build existing node map
    node_map: dict[str, EntityNode] = {}
    for node in existing_nodes:
        node_map[node.uuid] = node

    # Prepare context for LLM
    existing_nodes_context = [
        {'uuid': node.uuid, 'name': node.name, 'summary': node.summary} for node in existing_nodes
    ]

    extracted_nodes_context = [
        {'uuid': node.uuid, 'name': node.name, 'summary': node.summary} for node in extracted_nodes
    ]

    context = {
        'existing_nodes': existing_nodes_context,
        'extracted_nodes': extracted_nodes_context,
    }

    llm_response = await llm_client.generate_response(prompt_library.dedupe_nodes.node(context))

    duplicate_data = llm_response.get('duplicates', [])

    end = time()
    print(f'DEBUG: Deduplicated nodes: {duplicate_data} in {(end - start) * 1000} ms')

    uuid_map: dict[str, str] = {}
    for duplicate in duplicate_data:
        uuid_value = duplicate['duplicate_of']
        uuid_map[duplicate['uuid']] = uuid_value

    nodes: list[EntityNode] = []
    for node in extracted_nodes:
        if node.uuid in uuid_map:
            existing_uuid = uuid_map[node.uuid]
            existing_node = node_map[existing_uuid]
            nodes.append(existing_node)
        else:
            nodes.append(node)

    return nodes, uuid_map


async def resolve_extracted_nodes(
    clients: GraphitiClients,
    extracted_nodes: list[EntityNode],
    episode: EpisodicNode | None = None,
    previous_episodes: list[EpisodicNode] | None = None,
    entity_types: dict[str, BaseModel] | None = None,
) -> tuple[list[EntityNode], dict[str, str]]:
    llm_client = clients.llm_client

    search_results: list[SearchResults] = await semaphore_gather(
        *[
            search(
                clients=clients,
                query=node.name,
                group_ids=[node.group_id],
                search_filter=SearchFilters(),
                config=NODE_HYBRID_SEARCH_RRF,
            )
            for node in extracted_nodes
        ]
    )

    existing_nodes_dict: dict[str, EntityNode] = {
        node.uuid: node for result in search_results for node in result.nodes
    }

    existing_nodes: list[EntityNode] = list(existing_nodes_dict.values())

    existing_nodes_context = (
        [
            {
                **{
                    'idx': i,
                    'name': candidate.name,
                    'entity_types': candidate.labels,
                },
                **candidate.attributes,
            }
            for i, candidate in enumerate(existing_nodes)
        ],
    )

    entity_types_dict: dict[str, BaseModel] = entity_types if entity_types is not None else {}

    # Prepare context for LLM
    extracted_nodes_context = [
        {
            'id': i,
            'name': node.name,
            'entity_type': node.labels,
            'entity_type_description': entity_types_dict.get(
                next((item for item in node.labels if item != 'Entity'), '')
            ).__doc__
            or 'Default Entity Type',
        }
        for i, node in enumerate(extracted_nodes)
    ]

    context = {
        'extracted_nodes': extracted_nodes_context,
        'existing_nodes': existing_nodes_context,
        'episode_content': episode.content if episode is not None else '',
        'previous_episodes': [ep.content for ep in previous_episodes]
        if previous_episodes is not None
        else [],
    }

    llm_response = await llm_client.generate_response(
        prompt_library.dedupe_nodes.nodes(context),
        response_model=NodeResolutions,
    )

    node_resolutions: list[Any] = llm_response.get('entity_resolutions', [])

    resolved_nodes: list[EntityNode] = []
    uuid_map: dict[str, str] = {}
    for resolution in node_resolutions:
        resolution_id = resolution.get('id', -1)
        duplicate_idx = resolution.get('duplicate_idx', -1)

        extracted_node = extracted_nodes[resolution_id]

        resolved_node = (
            existing_nodes[duplicate_idx]
            if 0 <= duplicate_idx < len(existing_nodes)
            else extracted_node
        )

        resolved_node.name = resolution.get('name')

        resolved_nodes.append(resolved_node)
        uuid_map[extracted_node.uuid] = resolved_node.uuid

    print(f'DEBUG: Resolved nodes: {[(n.name, n.uuid) for n in resolved_nodes]}')

    return resolved_nodes, uuid_map


async def extract_attributes_from_nodes(
    clients: GraphitiClients,
    nodes: list[EntityNode],
    episode: EpisodicNode | None = None,
    previous_episodes: list[EpisodicNode] | None = None,
    entity_types: dict[str, BaseModel] | None = None,
) -> list[EntityNode]:
    llm_client = clients.llm_client
    embedder = clients.embedder
    updated_nodes: list[EntityNode] = await semaphore_gather(
        *[
            extract_attributes_from_node(
                llm_client,
                node,
                episode,
                previous_episodes,
                entity_types.get(next((item for item in node.labels if item != 'Entity'), ''))
                if entity_types is not None
                else None,
            )
            for node in nodes
        ]
    )

    await create_entity_node_embeddings(embedder, updated_nodes)

    return updated_nodes


async def extract_attributes_from_node(
    llm_client: LLMClient,
    node: EntityNode,
    episode: EpisodicNode | None = None,
    previous_episodes: list[EpisodicNode] | None = None,
    entity_type: BaseModel | None = None,
) -> EntityNode:
    node_context: dict[str, Any] = {
        'name': node.name,
        'summary': node.summary,
        'entity_types': node.labels,
        'attributes': node.attributes,
    }

    attributes_definitions: dict[str, Any] = {
        'summary': (
            str,
            Field(
                description='Summary containing the important information about the entity. Under 250 words',
            ),
        )
    }

    if entity_type is not None:
        for field_name, field_info in entity_type.model_fields.items():
            attributes_definitions[field_name] = (
                field_info.annotation,
                Field(description=field_info.description),
            )

    unique_model_name = f'EntityAttributes_{uuid4().hex}'
    entity_attributes_model = pydantic.create_model(unique_model_name, **attributes_definitions)

    summary_context: dict[str, Any] = {
        'node': node_context,
        'episode_content': episode.content if episode is not None else '',
        'previous_episodes': [ep.content for ep in previous_episodes]
        if previous_episodes is not None
        else [],
    }

    llm_response = await llm_client.generate_response(
        prompt_library.extract_nodes.extract_attributes(summary_context),
        response_model=entity_attributes_model,
        model_size=ModelSize.small,
    )

    node.summary = llm_response.get('summary', node.summary)
    node_attributes = {key: value for key, value in llm_response.items()}

    with suppress(KeyError):
        del node_attributes['summary']

    node.attributes.update(node_attributes)

    return node


async def dedupe_node_list(
    llm_client: LLMClient,
    nodes: list[EntityNode],
) -> tuple[list[EntityNode], dict[str, str]]:
    start = time()

    # build node map
    node_map = {}
    for node in nodes:
        node_map[node.uuid] = node

    # Prepare context for LLM
    nodes_context = [{'uuid': node.uuid, 'name': node.name, **node.attributes} for node in nodes]

    context = {
        'nodes': nodes_context,
    }

    llm_response = await llm_client.generate_response(
        prompt_library.dedupe_nodes.node_list(context)
    )

    nodes_data = llm_response.get('nodes', [])

    end = time()
    print(f'DEBUG: Deduplicated nodes: {nodes_data} in {(end - start) * 1000} ms')

    # Get full node data
    unique_nodes = []
    uuid_map: dict[str, str] = {}
    for node_data in nodes_data:
        node_instance: EntityNode | None = node_map.get(node_data['uuids'][0])
        if node_instance is None:
            print(f'WARNING: Node {node_data["uuids"][0]} not found in node map')
            continue
        node_instance.summary = node_data['summary']
        unique_nodes.append(node_instance)

        for uuid in node_data['uuids'][1:]:
            uuid_value = node_map[node_data['uuids'][0]].uuid
            uuid_map[uuid] = uuid_value

    return unique_nodes, uuid_map```

## File: `local_graphiti_utils/utils/text_splitters.py`
```
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, Union


@dataclass
class Document:
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class TextSplitter:
    """Base class for text splitters."""

    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
        keep_separator: Union[bool, Literal["start", "end"]] = False,
        add_start_index: bool = False,
        strip_whitespace: bool = True,
    ) -> None:
        """Create a new TextSplitter."""
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
                f"({chunk_size}), should be smaller."
            )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        self._keep_separator = keep_separator
        self._add_start_index = add_start_index
        self._strip_whitespace = strip_whitespace

    def split_text(self, text: str) -> List[str]:
        raise NotImplementedError

    def create_documents(
        self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[Document]:
        """Create documents from a list of texts."""
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            start_index = 0
            for chunk in self.split_text(text):
                metadata = _metadatas[i].copy()
                if self._add_start_index:
                    metadata["start_index"] = text.find(chunk, start_index)
                    start_index += len(chunk)
                new_doc = Document(page_content=chunk, metadata=metadata)
                documents.append(new_doc)
        return documents

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        # We now want to combine these smaller pieces into medium size
        # chunks to send to the LLM.
        separator_len = self._length_function(separator)

        docs = []
        current_doc: List[str] = []
        total = 0
        for d in splits:
            _len = self._length_function(d)
            if (
                total + _len + (separator_len if len(current_doc) > 0 else 0)
                > self._chunk_size
            ):
                if total > self._chunk_size:
                    print(
                        f"Created a chunk of size {total}, "
                        f"which is longer than the specified {self._chunk_size}"
                    )
                if len(current_doc) > 0:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    # Keep on popping if:
                    # - we have a larger chunk than in the chunk overlap
                    # - or if we still have any chunks and the length is long
                    while total > self._chunk_overlap or (
                        total + _len + (separator_len if len(current_doc) > 0 else 0)
                        > self._chunk_size
                        and total > 0
                    ):
                        total -= self._length_function(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += _len + (separator_len if len(current_doc) > 1 else 0)
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs

    def _join_docs(self, docs: List[str], separator: str) -> Optional[str]:
        text = separator.join(docs)
        if self._strip_whitespace:
            text = text.strip()
        if text == "":
            return None
        return text


class Language(str, Enum):
    """Enum for languages."""

    CPP = "cpp"
    GO = "go"
    JAVA = "java"
    KOTLIN = "kotlin"
    JS = "js"
    TS = "ts"
    PHP = "php"
    PROTO = "proto"
    PYTHON = "python"
    RST = "rst"
    RUBY = "ruby"
    RUST = "rust"
    SCALA = "scala"
    SWIFT = "swift"
    MARKDOWN = "markdown"
    LATEX = "latex"
    HTML = "html"
    SOL = "sol"
    CSHARP = "csharp"
    COBOL = "cobol"
    C = "c"
    LUA = "lua"
    HASKELL = "haskell"
    POWERSHELL = "powershell"
    ELIXIR = "elixir"


def _split_text_with_regex(
    text: str, separator: str, keep_separator: Union[bool, Literal["start", "end"]]
) -> List[str]:
    # Now that we have the separator, split the text
    if separator:
        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result.
            _splits = re.split(f"({separator})", text)
            if keep_separator == "end":
                splits = [_splits[i] + _splits[i + 1] for i in range(0, len(_splits) - 1, 2)]
                if len(_splits) % 2 == 1:
                    splits += _splits[-1:]
            else: # keep_separator == 'start' or True
                splits = [_splits[0]] + [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]

        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]


class RecursiveCharacterTextSplitter(TextSplitter):
    """Splitting text by recursively look at characters.

    Recursively tries to split by different characters to find one
    that works.
    """

    def __init__(
        self,
        separators: Optional[List[str]] = None,
        keep_separator: Union[bool, Literal["start", "end"]] = True,
        is_separator_regex: bool = False,
        **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or ["\n\n", "\n", " ", ""]
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        # Get appropriate separator to use
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1 :]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex(text, _separator, self._keep_separator)

        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        _separator_for_merge = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator_for_merge)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator_for_merge)
            final_chunks.extend(merged_text)
        return final_chunks

    def split_text(self, text: str) -> List[str]:
        """Split the input text into smaller chunks based on predefined separators.

        Args:
            text (str): The input text to be split.

        Returns:
            List[str]: A list of text chunks obtained after splitting.
        """
        return self._split_text(text, self._separators)

    @staticmethod
    def get_separators_for_language(language: Language) -> List[str]:
        if language == Language.PYTHON:
            return [
                # First, try to split along class definitions
                "\nclass ",
                "\ndef ",
                "\n\tdef ",
                # Now split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.MARKDOWN:
            return [
                # First, try to split along Markdown headings (starting with level 2)
                "\n#{1,6} ",
                # Note the alternative syntax for headings (below) is not handled here
                # Heading level 2
                # ---------------
                # End of code block
                "```\n",
                # Horizontal lines
                "\n\\*\\*\\*+\n",
                "\n---+\n",
                "\n___+\n",
                # Note that this splitter doesn't handle horizontal lines defined
                # by *three or more* of ***, ---, or ___, but this is not handled
                "\n\n",
                "\n",
                " ",
                "",
            ]
        # Add other languages as needed from the user's provided code
        else:
            return ["\n\n", "\n", " ", ""]


def split_text_intelligently(text: str, max_chunk_size: int = 4000) -> List[str]:
    """Split text intelligently preserving markdown structure and preventing token limit errors.
    
    Args:
        text: The input text to split
        max_chunk_size: Maximum size per chunk (default 4000 characters)
        
    Returns:
        List of text chunks that respect markdown structure
    """
    if len(text) <= max_chunk_size:
        return [text]
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=200,  # 5% overlap for context preservation
        separators=RecursiveCharacterTextSplitter.get_separators_for_language(Language.MARKDOWN)
    )
    return splitter.split_text(text) ```

## File: `logging_llm_client.py`
```
"""
Logging LLM Client Wrapper

A logging wrapper for OpenAIClient that logs all interactions for debugging and monitoring purposes.
"""

import time
from typing import Any
from pydantic import BaseModel

# Import the base OpenAIClient and required types
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.prompts.models import Message
from graphiti_core.llm_client.config import ModelSize


class LoggingOpenAIClient(OpenAIClient):
    """
    A logging wrapper for OpenAIClient that logs all generate_response calls.
    
    This client logs:
    - Input messages (sanitized for sensitive data)
    - Response model type
    - Max tokens
    - Model size
    - Response data (sanitized)
    - Execution time
    - Any errors that occur
    """
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
    
    def _sanitize_content(self, content: str, max_length: int = 200000) -> str:
        """
        Sanitize content for logging by truncating and removing sensitive patterns.
        
        Args:
            content: The content to sanitize
            max_length: Maximum length of content to log
            
        Returns:
            Sanitized content string
        """
        if not content:
            return ""
        
        # Truncate if too long
        if len(content) > max_length:
            content = content[:max_length] + "..."
        
        # Remove common sensitive patterns (you can extend this as needed)
        import re
        # Remove API keys, tokens, passwords
        content = re.sub(r'(api[_-]?key|token|password|secret)["\']?\s*[:=]\s*["\']?[\w\-]+', 
                        r'\1=***REDACTED***', content, flags=re.IGNORECASE)
        
        return content
    
    def _log_messages(self, messages: list[Message]) -> str:
        """
        Create a loggable representation of messages.
        
        Args:
            messages: List of Message objects
            
        Returns:
            String representation for logging
        """
        log_messages = []
        for i, msg in enumerate(messages):
            sanitized_content = self._sanitize_content(msg.content)
            log_messages.append(f"[{i}] {msg.role}: {sanitized_content}")
        
        return " | ".join(log_messages)
    
    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, Any]:
        """
        Generate a response from the LLM with comprehensive logging.
        
        Args:
            messages: List of message objects to send to the LLM.
            response_model: Optional Pydantic model to use for structured output.
            max_tokens: Maximum number of tokens to generate.
            model_size: Size of the model to use.
            
        Returns:
            Dictionary containing the structured response from the LLM.
        """
        start_time = time.time()
        
        # Log the incoming request
        model_name = self.config.model or "default"
        max_tokens_if_not_provided: int = 65500 if "gemini-2.5" in model_name.lower() else 8192
        print(
            f"LLM Request - Model: {self.model or 'default'}, "
            f"ModelSize: {model_size.value}, "
            f"MaxTokens: {max_tokens} (will set to {max_tokens_if_not_provided} if not provided), "
            f"ResponseModel: {response_model.__name__ if response_model else 'None'}, "
            f"Messages: {self._log_messages(messages)}"
        )
        
        try:
            # Call the parent's generate_response method
            response = await super().generate_response(
                messages=messages,
                response_model=response_model,
                max_tokens=max_tokens_if_not_provided if max_tokens is None else max_tokens,
                model_size=model_size
            )
            
            execution_time = time.time() - start_time
            
            # Log the successful response
            response_summary = self._sanitize_content(str(response))
            print(
                f"LLM Response - Success in {execution_time:.2f}s, "
                f"Response: {response_summary}"
            )
            
            return response
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Log the error
            print(
                f"LLM Response - Error in {execution_time:.2f}s, "
                f"Error: {type(e).__name__}: {str(e)}"
            )
            
            # Re-raise the original exception
            raise ```

## File: `ollama_embedder.py`
```
from graphiti_core.embedder.openai import OpenAIEmbedder
from typing import List


class NonBatchingOllamaEmbedder(OpenAIEmbedder):
    """
    Custom Ollama embedder that processes embeddings one by one instead of in batches.
    This avoids the "cannot decode batches with this context" error in Ollama.
    """
    
    async def create_batch(self, input_data_list: List[str]) -> List[List[float]]:
        """
        Override the batch method to process items sequentially instead of as a batch.
        This prevents Ollama's batch processing issues.
        """
        results: List[List[float]] = []
        for input_data in input_data_list:
            # Process each item individually using the single create method
            result = await self.create(input_data)
            results.append(result)
        return results ```

## File: `ollama_reranker_client.py`
```
#!/usr/bin/env python3
"""
Ollama-based Cross-Encoder Reranker Client
Uses Ollama API for reranking instead of local HuggingFace models
"""

import asyncio
import logging
from typing import Any
import json

try:
    import aiohttp
except ImportError:
    aiohttp = None

from graphiti_core.cross_encoder.client import CrossEncoderClient

logger = logging.getLogger(__name__)

class OllamaRerankerClient(CrossEncoderClient):
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "bge-m3"):
        """
        Initialize Ollama reranker client.
        
        Args:
            base_url: Ollama API base URL
            model_name: BGE embedding model for similarity-based reranking
        """
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        
    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        """
        Rank passages using BGE embeddings and cosine similarity.
        Uses BGE-M3 model for better multilingual performance.
        """
        if not passages:
            return []
            
        try:
            # Get embedding for the query
            query_embedding = await self._get_embedding(query)
            if query_embedding is None:
                return [(passage, 0.5) for passage in passages]
            
            # Get embeddings for all passages
            passage_embeddings = await asyncio.gather(
                *[self._get_embedding(passage) for passage in passages]
            )
            
            # Calculate cosine similarities
            scores = []
            for i, passage_embedding in enumerate(passage_embeddings):
                if passage_embedding is not None:
                    similarity = self._cosine_similarity(query_embedding, passage_embedding)
                    scores.append((passages[i], float(similarity)))
                else:
                    scores.append((passages[i], 0.0))
            
            # Sort by score descending
            ranked_passages = sorted(scores, key=lambda x: x[1], reverse=True)
            
            logger.debug(f"Ranked {len(passages)} passages using BGE embeddings")
            return ranked_passages
            
        except Exception as e:
            logger.error(f"Error ranking passages with BGE: {e}")
            # Fallback: return passages in original order with neutral scores
            return [(passage, 0.5) for passage in passages]
    
    async def _get_embedding(self, text: str) -> list[float] | None:
        """Get embedding for text using BGE model via Ollama API."""
        url = f"{self.base_url}/api/embeddings"
        
        payload = {
            "model": self.model_name,
            "prompt": text
        }
        
        if aiohttp is None:
            logger.error("aiohttp not available, cannot make API calls")
            return None
            
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("embedding")
                    else:
                        logger.error(f"Ollama API error: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Failed to get BGE embedding from Ollama: {e}")
            return None
    
    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
            
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Calculate magnitudes
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
            
        return dot_product / (magnitude1 * magnitude2)
    
 ```

## File: `runpod_openai_client.py`
```
"""
Runpod OpenAI Client Wrapper

A wrapper for OpenAIClient that adapts it for Runpod's OpenAI-compatible endpoint
and logs all interactions for debugging and monitoring purposes.
"""

import time
import re
import json
from typing import Any
from pydantic import BaseModel
import httpx

from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.llm_client.errors import RefusalError, RateLimitError
from graphiti_core.prompts.models import Message
from graphiti_core.llm_client.config import ModelSize, LLMConfig


class RunpodWrapperOpenAIClient(OpenAIClient):
    """
    A wrapper for OpenAIClient that logs all generate_response calls
    and adapts to Runpod's OpenAI-compatible endpoint.
    """
    
    def __init__(self, 
                 base_url: str, # e.g. "https://api.runpod.ai/v2/your_endpoint_id"
                 api_key: str | None = None,
                 model: str = "qwen3",
                 temperature: float = 0.0,
                 max_tokens: int = 8192,
                 *args: Any, **kwargs: Any) -> None:

        config = LLMConfig(
            api_key=api_key,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens
        )
        # We don't initialize the parent with a client, as we'll be overriding the request logic.
        super().__init__(config, *args, **kwargs)
        self.runpod_base_url = base_url
        self.httpx_client = httpx.AsyncClient(timeout=120)

    def _sanitize_content(self, content: str, max_length: int = 200000) -> str:
        """Sanitizes content for logging."""
        if not content:
            return ""
        if len(content) > max_length:
            content = content[:max_length] + "..."
        content = re.sub(r'(api[_-]?key|token|password|secret)["\']?\s*[:=]\s*["\']?[\w\-]+', 
                        r'\\1=***REDACTED***', content, flags=re.IGNORECASE)
        return content

    def _log_messages(self, messages: list[Message]) -> str:
        """Creates a loggable representation of messages."""
        log_messages = []
        for i, msg in enumerate(messages):
            sanitized_content = self._sanitize_content(msg.content)
            log_messages.append(f"[{i}] {msg.role}: {sanitized_content}")
        return " | ".join(log_messages)

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, Any]:
        """Generates a response by wrapping the request for the Runpod /runsync endpoint."""
        
        openai_messages: list[dict[str, str]] = [
            {'role': m.role, 'content': self._clean_input(m.content)} for m in messages
        ]

        model_to_use = self.small_model if model_size == ModelSize.small else self.model
        
        openai_payload: dict[str, Any] = {
            "model": model_to_use,
            "messages": openai_messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if response_model:
            openai_payload["response_format"] = {"type": "json_object"}


        runpod_payload = {
            "input": {
                **openai_payload,
                "url_path": "/api/chat"
            }
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }

        url = f"{self.runpod_base_url}/runsync"
        
        try:
            response = await self.httpx_client.post(url, json=runpod_payload, headers=headers)
            response.raise_for_status()
            response_json = response.json()

            if "output" not in response_json or not response_json["output"]:
                raise ValueError(f"Invalid response from Runpod: {response_json}")

            openai_response_content = response_json["output"]["choices"][0]["message"]["content"]
            
            if response_model:
                 parsed_content = json.loads(openai_response_content)
                 return response_model(**parsed_content).model_dump()

            # Ensure we return a proper dict[str, Any] instead of Any
            parsed_response: dict[str, Any] = json.loads(openai_response_content)
            return parsed_response

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError from e
            raise
        except Exception:
            # Handle potential refusal from the model if wrapped in the output
            if "output" in locals() and "refusal" in str(locals().get("output")):
                raise RefusalError(str(locals().get("output")))
            raise

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, Any]:
        """
        Generate a response from the LLM with comprehensive logging.
        """
        start_time = time.time()
        
        max_tokens_to_use = max_tokens or self.max_tokens
        print(
            f"LLM Request - Model: {self.model or 'default'}, "
            f"ModelSize: {model_size.value}, "
            f"MaxTokens: {max_tokens_to_use}, "
            f"ResponseModel: {response_model.__name__ if response_model else 'None'}, "
            f"Messages: {self._log_messages(messages)}"
        )
        
        try:
            response = await self._generate_response(
                messages=messages,
                response_model=response_model,
                max_tokens=max_tokens_to_use,
                model_size=model_size
            )
            
            execution_time = time.time() - start_time
            response_summary = self._sanitize_content(str(response))
            print(
                f"LLM Response - Success in {execution_time:.2f}s, "
                f"Response: {response_summary}"
            )
            
            return response
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(
                f"LLM Response - Error in {execution_time:.2f}s, "
                f"Error: {type(e).__name__}: {str(e)}"
            )
            raise ```

## File: `runpod_openai_embedder.py`
```
"""
Runpod OpenAI Embedder Wrapper

A wrapper for OpenAIEmbedder that adapts it for Runpod's OpenAI-compatible endpoint
and processes embeddings sequentially.
"""

from typing import List, Iterable, Any
from openai import AsyncOpenAI
import httpx

from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

class RunpodWrapperOpenAIEmbedder(OpenAIEmbedder):
    """
    Custom Runpod embedder that sends requests to a Runpod OpenAI-compatible endpoint
    and processes embeddings one by one instead of in batches.
    """

    def __init__(self, 
                 base_url: str, # e.g. "https://api.runpod.ai/v2/your_endpoint_id"
                 api_key: str | None = None,
                 embedding_model: str = "bge-m3",
                 *args: Any, **kwargs: Any) -> None:
        
        config = OpenAIEmbedderConfig(
            api_key=api_key,
            embedding_model=embedding_model,
            base_url=base_url # This will be the Runpod endpoint base
        )
        # We don't call super().__init__() with a client because we'll handle requests differently.
        super().__init__(config, client=AsyncOpenAI(api_key=api_key, base_url=None)) # Pass dummy client
        
        self.runpod_base_url = base_url
        self.httpx_client = httpx.AsyncClient(timeout=120)

    async def create(self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]) -> List[float]:
        """
        Create embeddings for the given input data using the Runpod endpoint.
        """
        openai_payload = {
            "input": input_data,
            "model": self.config.embedding_model
        }

        runpod_payload = {
            "input": {
                **openai_payload,
                "url_path": "/api/embeddings"
            }
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }

        url = f"{self.runpod_base_url}/runsync"
        
        response = await self.httpx_client.post(url, json=runpod_payload, headers=headers)
        response.raise_for_status()
        
        response_json = response.json()

        if "output" not in response_json or not response_json["output"]:
            raise ValueError(f"Invalid response from Runpod: {response_json}")

        # The output from runsync is the OpenAI-compatible response
        openai_response = response_json["output"]

        # Ensure we return the expected type
        embedding_data: List[float] = openai_response["data"][0]["embedding"]
        return embedding_data

    async def create_batch(self, input_data_list: List[str]) -> List[List[float]]:
        """
        Override the batch method to process items sequentially instead of as a batch.
        This is often necessary for custom or non-standard API endpoints.
        """
        results: List[List[float]] = []
        for input_data in input_data_list:
            # Process each item individually using the single create method
            result = await self.create(input_data)
            results.append(result)
        return results ```

## File: `session_log_file_writer.py`
```
import json
from typing import Any


class SessionLogFileWriter:
    def __init__(self, session_id: str, file_pattern: str = "session_log_{session_id}_{section_name}.json"):
        self.session_id: str = session_id
        self.file_pattern: str = file_pattern

    async def write_session_log(self, section: str, data: dict[str, Any]) -> None:
        file_path = self.file_pattern.format(session_id=self.session_id, section_name=section)
        with open(file_path, "a") as f:
            json.dump(data, f, indent=2)```

## File: `standalone_tests/__init__.py`
```
# Local graphiti_core utilities
```

## File: `standalone_tests/standalone_batteries_test.py`
```
#!/usr/bin/env python3
"""
Test script for "batteries included" Graphiti hybrid search
Validates that hybrid BM25 + vector search works with Ollama setup
"""

import requests
from datetime import datetime
from typing import Tuple, Dict, Any, List

# Test the improved endpoints
BASE_URL = "http://localhost:8767"

def test_endpoint(endpoint: str, params: Dict[str, Any], description: str) -> Tuple[bool, int]:
    """Test a single endpoint and return results"""
    print(f"\n🧪 {description}")
    print("-" * 50)
    
    try:
        start_time = datetime.now()
        response = requests.get(f"{BASE_URL}{endpoint}", params=params, timeout=30)
        end_time = datetime.now()
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract results count based on endpoint type
            if 'nodes' in data:
                count = len(data.get('nodes', []))
                results_type = "nodes"
                results = data['nodes']
            elif 'facts' in data:
                count = len(data.get('facts', []))
                results_type = "facts"
                results = data['facts']
            else:
                count = 0
                results_type = "items"
                results = []
            
            duration = (end_time - start_time).total_seconds()
            
            print(f"✅ SUCCESS: Found {count} {results_type}")
            print(f"⏱️  Response time: {duration:.3f}s")
            print(f"📄 Message: {data.get('message', 'N/A')}")
            
            # Show sample results
            if count > 0:
                print("\n📋 Sample results:")
                for i, result in enumerate(results[:3]):  # Show first 3
                    if results_type == "nodes":
                        name = result.get('name', 'N/A')[:50]
                        summary = result.get('summary', 'N/A')[:100]
                        print(f"   {i+1}. {name}")
                        print(f"      {summary}...")
                    elif results_type == "facts":
                        fact = result.get('fact', 'N/A')[:80]
                        name = result.get('name', 'N/A')[:30]
                        print(f"   {i+1}. [{name}] {fact}")
            
            return True, count
            
        else:
            print(f"❌ HTTP Error {response.status_code}: {response.text}")
            return False, 0
            
    except requests.exceptions.Timeout:
        print("⏰ TIMEOUT: Request took longer than 30 seconds")
        return False, 0
    except Exception as e:
        print(f"💥 ERROR: {str(e)}")
        return False, 0

def main() -> None:
    """Run comprehensive test of batteries included search"""
    
    print("🚗 TESTING 'BATTERIES INCLUDED' GRAPHITI SEARCH")
    print("=" * 60)
    print("Testing hybrid BM25 + vector similarity with RRF reranking")
    print("Designed to work with Ollama (avoiding cross-encoder issues)")
    print()
    
    # Test cases from the original analysis that failed
    test_cases: List[Dict[str, Any]] = [
        {
            "endpoint": "/search_memory_nodes",
            "params": {
                "query": "Smart",
                "group_ids": ["chrome-tracker-content"],
                "max_nodes": 5
            },
            "description": "Node Search: 'Smart' (previously failed)"
        },
        {
            "endpoint": "/search_memory_nodes", 
            "params": {
                "query": "Christian",
                "group_ids": ["chrome-tracker-content"],
                "max_nodes": 5
            },
            "description": "Node Search: 'Christian' (previously failed)"
        },
        {
            "endpoint": "/search_memory_nodes",
            "params": {
                "query": "AI technology",
                "group_ids": ["chrome-tracker-content"], 
                "max_nodes": 5
            },
            "description": "Node Search: 'AI technology' (semantic test)"
        },
        {
            "endpoint": "/search_memory_facts",
            "params": {
                "query": "Smart AI",
                "group_ids": ["chrome-tracker-content"],
                "max_facts": 5
            },
            "description": "Facts Search: 'Smart AI' (previously failed)"
        },
        {
            "endpoint": "/search_memory_facts",
            "params": {
                "query": "blog software",
                "group_ids": ["chrome-tracker-content"],
                "max_facts": 5
            },
            "description": "Facts Search: 'blog software' (semantic test)"
        }
    ]
    
    # Run all tests
    total_tests = len(test_cases)
    successful_tests = 0
    total_results = 0
    
    for test_case in test_cases:
        success, count = test_endpoint(
            test_case["endpoint"],
            test_case["params"], 
            test_case["description"]
        )
        if success:
            successful_tests += 1
            total_results += count
    
    # Summary
    print("\n" + "=" * 60)
    print("🏁 FINAL RESULTS")
    print("=" * 60)
    print(f"✅ Successful tests: {successful_tests}/{total_tests}")
    print(f"📊 Total results found: {total_results}")
    
    if successful_tests == total_tests and total_results > 0:
        print("🎉 SUCCESS: 'Batteries included' hybrid search is working!")
        print("🚀 You now have BM25 + vector similarity + RRF reranking")
        print("🔧 Power level: ~60-70% (up from 20%)")
    elif successful_tests > 0:
        print("⚠️  PARTIAL SUCCESS: Some searches working")
        print("🔍 Check individual test results above")
    else:
        print("❌ FAILURE: No searches working")
        print("🔧 May need to fall back to BM25-only or investigate further")
    
    print("\n💡 Next steps if successful:")
    print("• Add graph traversal (BFS) for connected concepts")
    print("• Build communities for community-based search")
    print("• Implement temporal filtering for time-aware queries")
    print("• Consider MMR reranking for result diversity")

if __name__ == "__main__":
    main() ```

## File: `standalone_tests/test_all_readonly_endpoints.py`
```
#!/usr/bin/env python3
"""
Comprehensive test suite for all read-only REST endpoints
Uses structural validation on live data - tests field existence and non-emptiness
rather than specific content values for robustness against data changes.
"""

import requests
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, TypedDict
import json

# Test configuration
BASE_URL = "http://localhost:8767"
TEST_GROUP_IDS = ["chrome-tracker-content"]  # Live data group
TIMEOUT_SECONDS = 30

class TestCase(TypedDict):
    params: Dict[str, Any]
    description: str

class EndpointTester:
    """Helper class for testing REST endpoints with structural validation"""
    
    def __init__(self, base_url: str = BASE_URL) -> None:
        self.base_url = base_url
        self.results: List[Dict[str, Any]] = []
    
    def test_endpoint(self, method: str, endpoint: str, params: Optional[Dict[str, Any]] = None, 
                     json_data: Optional[Dict[str, Any]] = None, description: str = "") -> Tuple[bool, Dict[str, Any]]:
        """Test a single endpoint and return success status and response data"""
        print(f"\n🧪 {description}")
        print("-" * 60)
        
        try:
            start_time = datetime.now()
            
            if method.upper() == "GET":
                response = requests.get(f"{self.base_url}{endpoint}", params=params, timeout=TIMEOUT_SECONDS)
            elif method.upper() == "POST":
                response = requests.post(f"{self.base_url}{endpoint}", params=params, json=json_data, timeout=TIMEOUT_SECONDS)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print(f"📡 Request: {method} {endpoint}")
            print(f"⏱️  Response time: {duration:.3f}s")
            print(f"🌐 Status code: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    print("✅ SUCCESS: Valid JSON response received")
                    return True, data
                except json.JSONDecodeError:
                    print("❌ ERROR: Invalid JSON in response")
                    return False, {}
            else:
                print(f"❌ HTTP Error {response.status_code}: {response.text[:200]}...")
                return False, {}
                
        except requests.exceptions.Timeout:
            print("⏰ TIMEOUT: Request exceeded timeout limit")
            return False, {}
        except Exception as e:
            print(f"💥 ERROR: {str(e)}")
            return False, {}
    
    def validate_node_structure(self, nodes: List[Dict[str, Any]]) -> bool:
        """Validate structure of node objects"""
        if not isinstance(nodes, list):
            print("❌ Nodes should be a list")
            return False
        
        for i, node in enumerate(nodes):
            print(f"   Validating node {i+1}/{len(nodes)}...")
            
            # Required fields
            required_fields = ['uuid', 'name', 'group_id', 'created_at', 'labels']
            for field in required_fields:
                if field not in node:
                    print(f"❌ Missing required field: {field}")
                    return False
                if not node[field]:  # Check non-empty
                    print(f"❌ Empty required field: {field}")
                    return False
            
            # Type validations
            if not isinstance(node['uuid'], str) or len(node['uuid']) < 10:
                print(f"❌ Invalid UUID format: {node['uuid']}")
                return False
            
            if not isinstance(node['name'], str):
                print(f"❌ Name should be string: {type(node['name'])}")
                return False
            
            if not isinstance(node['labels'], list):
                print(f"❌ Labels should be list: {type(node['labels'])}")
                return False
            
            if not isinstance(node['group_id'], str):
                print(f"❌ Group ID should be string: {type(node['group_id'])}")
                return False
            
            # Optional fields (can be empty but should exist)
            optional_fields = ['summary', 'attributes']
            for field in optional_fields:
                if field in node and node[field] is not None:
                    if field == 'attributes' and not isinstance(node[field], dict):
                        print(f"❌ Attributes should be dict: {type(node[field])}")
                        return False
        
        print(f"✅ All {len(nodes)} nodes have valid structure")
        return True
    
    def validate_facts_structure(self, facts: List[Dict[str, Any]], expect_enhanced: bool = False) -> bool:
        """Validate structure of facts/edges objects"""
        if not isinstance(facts, list):
            print("❌ Facts should be a list")
            return False
        
        for i, fact in enumerate(facts):
            print(f"   Validating fact {i+1}/{len(facts)}...")
            
            # Required fields
            required_fields = ['uuid', 'source_node_uuid', 'target_node_uuid', 'group_id', 'created_at']
            for field in required_fields:
                if field not in fact:
                    print(f"❌ Missing required field: {field}")
                    return False
                if not fact[field]:  # Check non-empty
                    print(f"❌ Empty required field: {field}")
                    return False
            
            # Enhanced fields validation (for when we add the join enhancement)
            if expect_enhanced:
                enhanced_fields = ['source_node_name', 'target_node_name']
                for field in enhanced_fields:
                    if field not in fact:
                        print(f"❌ Missing enhanced field: {field}")
                        return False
                    if not fact[field]:  # Check non-empty
                        print(f"❌ Empty enhanced field: {field}")
                        return False
                    if not isinstance(fact[field], str):
                        print(f"❌ Enhanced field should be string: {field}")
                        return False
                    # Sanity check: shouldn't be a UUID
                    if fact[field].count('-') >= 4:
                        print(f"❌ Enhanced field looks like UUID, should be name: {field}")
                        return False
                print(f"✅ Enhanced fields validated for fact {i+1}")
            
            # Type validations
            uuid_fields = ['uuid', 'source_node_uuid', 'target_node_uuid']
            for field in uuid_fields:
                if not isinstance(fact[field], str) or len(fact[field]) < 10:
                    print(f"❌ Invalid UUID format for {field}: {fact[field]}")
                    return False
            
            if not isinstance(fact['group_id'], str):
                print(f"❌ Group ID should be string: {type(fact['group_id'])}")
                return False
            
            # Optional fields
            optional_fields = ['name', 'fact', 'episodes', 'attributes']
            for field in optional_fields:
                if field in fact and fact[field] is not None:
                    if field == 'episodes' and not isinstance(fact[field], list):
                        print(f"❌ Episodes should be list: {type(fact[field])}")
                        return False
                    if field == 'attributes' and not isinstance(fact[field], dict):
                        print(f"❌ Attributes should be dict: {type(fact[field])}")
                        return False
        
        print(f"✅ All {len(facts)} facts have valid structure")
        return True
    
    def validate_response_structure(self, data: Dict[str, Any], expected_type: str) -> bool:
        """Validate top-level response structure"""
        if 'message' not in data:
            print("❌ Missing 'message' field in response")
            return False
        
        if not isinstance(data['message'], str):
            print(f"❌ Message should be string: {type(data['message'])}")
            return False
        
        if expected_type == 'nodes':
            if 'nodes' not in data:
                print("❌ Missing 'nodes' field in response")
                return False
            return self.validate_node_structure(data['nodes'])
        
        elif expected_type == 'facts':
            if 'facts' not in data:
                print("❌ Missing 'facts' field in response")
                return False
            return self.validate_facts_structure(data['facts'])
        
        elif expected_type == 'raw_query':
            # Raw query returns a list directly
            if not isinstance(data, list):
                print(f"❌ Raw query should return list: {type(data)}")
                return False
            print(f"✅ Raw query returned {len(data)} records")
            return True
        
        return True


def test_search_memory_nodes() -> Tuple[int, int]:
    """Test /search_memory_nodes endpoint"""
    tester = EndpointTester()
    
    test_cases: List[TestCase] = [
        {
            "params": {"query": "Smart", "group_ids": TEST_GROUP_IDS, "max_nodes": 3},
            "description": "Search nodes: 'Smart' (keyword search)"
        },
        {
            "params": {"query": "AI technology", "group_ids": TEST_GROUP_IDS, "max_nodes": 5},
            "description": "Search nodes: 'AI technology' (semantic search)"
        },
        {
            "params": {"query": "Christian", "group_ids": TEST_GROUP_IDS, "max_nodes": 2},
            "description": "Search nodes: 'Christian' (name search)"
        },
        {
            "params": {"query": "nonexistent query that should return empty", "group_ids": TEST_GROUP_IDS, "max_nodes": 10},
            "description": "Search nodes: nonexistent query (empty result test)"
        }
    ]
    
    successful_tests = 0
    total_tests = len(test_cases)
    
    print("🔍 TESTING /search_memory_nodes ENDPOINT")
    print("=" * 60)
    
    for test_case in test_cases:
        success, data = tester.test_endpoint("GET", "/search_memory_nodes", 
                                           params=test_case["params"], 
                                           description=test_case["description"])
        
        if success:
            # Validate response structure
            if tester.validate_response_structure(data, 'nodes'):
                nodes = data.get('nodes', [])
                print(f"📊 Found {len(nodes)} nodes")
                if nodes:
                    # Show sample for debugging
                    sample = nodes[0]
                    print(f"📋 Sample node: {sample.get('name', 'N/A')[:40]}...")
                successful_tests += 1
            else:
                print("❌ Response structure validation failed")
        
        print()
    
    print(f"🏁 Node search tests: {successful_tests}/{total_tests} successful")
    return successful_tests, total_tests


def test_search_memory_facts() -> Tuple[int, int]:
    """Test /search_memory_facts endpoint"""
    tester = EndpointTester()
    
    test_cases: List[TestCase] = [
        {
            "params": {"query": "Smart AI", "group_ids": TEST_GROUP_IDS, "max_facts": 3},
            "description": "Search facts: 'Smart AI' (relationship search)"
        },
        {
            "params": {"query": "blog software", "group_ids": TEST_GROUP_IDS, "max_facts": 5},
            "description": "Search facts: 'blog software' (semantic relationship search)"
        },
        {
            "params": {"query": "Christian", "group_ids": TEST_GROUP_IDS, "max_facts": 2},
            "description": "Search facts: 'Christian' (entity-based relationships)"
        },
        {
            "params": {"query": "nonexistent relationship query", "group_ids": TEST_GROUP_IDS, "max_facts": 10},
            "description": "Search facts: nonexistent query (empty result test)"
        }
    ]
    
    successful_tests = 0
    total_tests = len(test_cases)
    
    print("🔗 TESTING /search_memory_facts ENDPOINT (WITH NODE NAMES)")
    print("=" * 60)
    
    for test_case in test_cases:
        success, data = tester.test_endpoint("GET", "/search_memory_facts", 
                                           params=test_case["params"], 
                                           description=test_case["description"])
        
        if success:
            # Validate response structure with enhanced validation expecting node names
            if tester.validate_response_structure(data, 'facts'):
                facts = data.get('facts', [])
                print(f"📊 Found {len(facts)} facts")
                if facts and tester.validate_facts_structure(facts, expect_enhanced=True):
                    # Show enhanced sample for debugging
                    sample = facts[0]
                    fact_text = sample.get('fact', sample.get('name', 'N/A'))
                    source_name = sample.get('source_node_name', 'N/A')
                    target_name = sample.get('target_node_name', 'N/A')
                    print(f"📋 Enhanced sample: {source_name} -> {target_name}")
                    print(f"    Fact: {fact_text[:50]}...")
                    successful_tests += 1
                else:
                    print("❌ Enhanced structure validation failed (missing node names)")
            else:
                print("❌ Response structure validation failed")
        
        print()
    
    print(f"🏁 Facts search tests: {successful_tests}/{total_tests} successful")
    return successful_tests, total_tests


def test_search_memory_facts_enhanced() -> Tuple[int, int]:
    """Test /search_memory_facts endpoint with enhanced validation (for after join enhancement)"""
    tester = EndpointTester()
    
    test_cases: List[TestCase] = [
        {
            "params": {"query": "Smart AI", "group_ids": TEST_GROUP_IDS, "max_facts": 3},
            "description": "Search facts with node names: 'Smart AI'"
        },
        {
            "params": {"query": "Christian", "group_ids": TEST_GROUP_IDS, "max_facts": 2},
            "description": "Search facts with node names: 'Christian'"
        }
    ]
    
    successful_tests = 0
    total_tests = len(test_cases)
    
    print("🔗✨ TESTING /search_memory_facts ENDPOINT (ENHANCED WITH NODE NAMES)")
    print("=" * 60)
    
    for test_case in test_cases:
        success, data = tester.test_endpoint("GET", "/search_memory_facts", 
                                           params=test_case["params"], 
                                           description=test_case["description"])
        
        if success:
            # Validate response structure with enhanced fields
            if tester.validate_response_structure(data, 'facts'):
                facts = data.get('facts', [])
                print(f"📊 Found {len(facts)} facts")
                if facts and tester.validate_facts_structure(facts, expect_enhanced=True):
                    # Show enhanced sample for debugging
                    sample = facts[0]
                    fact_text = sample.get('fact', sample.get('name', 'N/A'))
                    source_name = sample.get('source_node_name', 'N/A')
                    target_name = sample.get('target_node_name', 'N/A')
                    print(f"📋 Enhanced sample: {source_name} -> {target_name}")
                    print(f"    Fact: {fact_text[:50]}...")
                    successful_tests += 1
                else:
                    print("❌ Enhanced structure validation failed")
            else:
                print("❌ Response structure validation failed")
        
        print()
    
    print(f"🏁 Enhanced facts search tests: {successful_tests}/{total_tests} successful")
    return successful_tests, total_tests



def main() -> None:
    """Run all read-only endpoint tests"""
    print("🚀 COMPREHENSIVE READ-ONLY ENDPOINT TESTING")
    print("=" * 70)
    print("Testing all read-only REST endpoints with structural validation")
    print("Using live data but validating structure rather than content")
    print(f"Target: {BASE_URL}")
    print(f"Group IDs: {TEST_GROUP_IDS}")
    print()
    
    total_successful = 0
    total_tests = 0
    
    # Test all endpoints
    test_functions = [
        test_search_memory_nodes,
        test_search_memory_facts,
        test_search_memory_facts_enhanced
    ]
    
    for test_func in test_functions:
        try:
            successful, tests = test_func()
            total_successful += successful
            total_tests += tests
        except Exception as e:
            print(f"💥 Error in {test_func.__name__}: {str(e)}")
        print()
    
    # Final summary
    print("=" * 70)
    print("🏁 FINAL COMPREHENSIVE TEST RESULTS")
    print("=" * 70)
    print(f"✅ Successful tests: {total_successful}/{total_tests}")
    print(f"📊 Success rate: {(total_successful/total_tests*100):.1f}%" if total_tests > 0 else "📊 No tests run")
    
    if total_successful == total_tests and total_tests > 0:
        print("🎉 SUCCESS: All read-only endpoints working perfectly!")
        print("✅ All response structures validated")
        print("✅ All field types validated")
        print("✅ All required fields present and non-empty")
    elif total_successful > 0:
        print("⚠️  PARTIAL SUCCESS: Some endpoints working")
        print("🔍 Check individual test results above for details")
    else:
        print("❌ FAILURE: No endpoints working")
        print("🔧 Check server status and configuration")
    
    print()
    print("🔮 NEXT STEPS:")
    print("• To test enhanced facts endpoint, run test_search_memory_facts_enhanced() after implementing join")
    print("• All tests use structural validation - robust against data changes")
    print("• Tests validate field existence, types, and non-emptiness")
    print("• Safe to run on live data (read-only operations)")


if __name__ == "__main__":
    main() ```

## File: `test_concurrency_demo.py`
```
#!/usr/bin/env python3
"""
Demo script to test Vertex AI concurrency limiting.

Usage:
    export VERTEX_AI_MAX_CONCURRENT=2
    uv run python test_concurrency_demo.py
"""

import asyncio
import os
from typing import Any
import pytest
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.prompts.models import Message
from vertex_ai_client import VertexAIClient

# Set a low concurrency limit for demo
os.environ["VERTEX_AI_MAX_CONCURRENT"] = "2"

@pytest.fixture
def client() -> VertexAIClient:
    """Create a VertexAIClient for testing."""
    config = LLMConfig(
        model="gemini-2.5-flash",
        temperature=0.1,
        max_tokens=100
    )
    return VertexAIClient(config)

async def single_request_helper(client: VertexAIClient, request_id: int) -> dict[str, Any] | None:
    """Test a single request with timing."""
    print(f"🚀 Starting request {request_id}")
    
    messages = [
        Message(role="user", content=f"Hello! This is test request #{request_id}. Please respond with a short greeting.")
    ]
    
    try:
        start_time = asyncio.get_event_loop().time()
        result = await client._generate_response(messages)
        duration = asyncio.get_event_loop().time() - start_time
        
        print(f"✅ Request {request_id} completed in {duration:.2f}s")
        print(f"   Response: {result.get('content', '')[:100]}...")
        return result
        
    except Exception as e:
        print(f"❌ Request {request_id} failed: {e}")
        return None

@pytest.mark.asyncio
async def test_single_request(client: VertexAIClient) -> None:
    """Test a single request."""
    result = await single_request_helper(client, 1)
    assert result is not None
    assert "content" in result

@pytest.mark.asyncio 
async def test_concurrent_requests(client: VertexAIClient) -> None:
    """Test multiple concurrent requests to see semaphore in action."""
    print("🧪 Testing Vertex AI Concurrency Limiting")
    print("=" * 50)
    
    print(f"\n📊 Max concurrent requests: {client._max_concurrent}")
    print("Starting 5 requests simultaneously...")
    print("Expected: Only 2 should run at once, others will wait\n")
    
    # Launch 5 requests simultaneously
    tasks = [
        single_request_helper(client, i) 
        for i in range(1, 6)
    ]
    
    start_time = asyncio.get_event_loop().time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    total_time = asyncio.get_event_loop().time() - start_time
    
    print(f"\n🏁 All requests completed in {total_time:.2f}s")
    successful = sum(1 for r in results if r is not None and not isinstance(r, Exception))
    failed = sum(1 for r in results if r is None or isinstance(r, Exception))
    print(f"✅ Successful requests: {successful}")
    print(f"❌ Failed requests: {failed}")
    
    # At least some requests should succeed
    assert successful > 0

async def demo_concurrent_requests() -> None:
    """Demo function for running outside of pytest."""
    # Create client
    config = LLMConfig(
        model="gemini-2.5-flash",
        temperature=0.1,
        max_tokens=100
    )
    
    client = VertexAIClient(config)
    
    print("🧪 Testing Vertex AI Concurrency Limiting")
    print("=" * 50)
    
    print(f"\n📊 Max concurrent requests: {client._max_concurrent}")
    print("Starting 5 requests simultaneously...")
    print("Expected: Only 2 should run at once, others will wait\n")
    
    # Launch 5 requests simultaneously
    tasks = [
        single_request_helper(client, i) 
        for i in range(1, 6)
    ]
    
    start_time = asyncio.get_event_loop().time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    total_time = asyncio.get_event_loop().time() - start_time
    
    print(f"\n🏁 All requests completed in {total_time:.2f}s")
    print(f"✅ Successful requests: {sum(1 for r in results if r is not None and not isinstance(r, Exception))}")
    print(f"❌ Failed requests: {sum(1 for r in results if r is None or isinstance(r, Exception))}")

if __name__ == "__main__":
    asyncio.run(demo_concurrent_requests()) ```

## File: `test_max_tokens_failure.py`
```
#!/usr/bin/env python3
"""
Test script to verify MAX_TOKENS hard failure behavior.
This script intentionally triggers MAX_TOKENS to test the hard failure.
"""

import asyncio
import pytest
from vertex_ai_client import VertexAIClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.prompts.models import Message


@pytest.mark.asyncio
async def test_max_tokens_failure() -> bool:
    """Test that MAX_TOKENS causes a hard failure (RuntimeError)."""
    
    print("🧪 Testing MAX_TOKENS hard failure behavior...")
    
    # Create client with very low token limit
    config = LLMConfig(temperature=0.0)
    client = VertexAIClient(config)
    
    # Create a very long prompt that will definitely exceed a small token limit
    long_prompt = "Write a comprehensive 10,000 word essay about the history of artificial intelligence, covering every major milestone, researcher, breakthrough, and technological advancement from the 1940s to present day. Include detailed explanations of neural networks, machine learning algorithms, deep learning, transformer architectures, and recent developments in large language models. " * 10
    
    try:
        print("🔥 Attempting to generate with intentionally low max_tokens (should fail hard)...")
        
        # Use very low max_tokens to trigger MAX_TOKENS
        response = await client._generate_response(
            messages=[Message(role="user", content=long_prompt)],
            max_tokens=50,  # Intentionally very low
        )
        
        print("❌ ERROR: Should have failed hard but didn't!")
        print(f"Response: {response}")
        return False
        
    except RuntimeError as e:
        if "MAX_TOKENS" in str(e) or "token limit" in str(e):
            print("✅ SUCCESS: Hard failure correctly triggered!")
            print(f"✅ Error message: {e}")
            return True
        else:
            print(f"❌ ERROR: Got RuntimeError but wrong type: {e}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: Got unexpected exception type: {type(e).__name__}: {e}")
        return False


@pytest.mark.asyncio
async def test_normal_operation() -> bool:
    """Test that normal operation still works."""
    
    print("\n🧪 Testing normal operation (should work fine)...")
    
    config = LLMConfig(temperature=0.0)
    client = VertexAIClient(config)
    
    try:
        response = await client._generate_response(
            messages=[Message(role="user", content="Say hello in exactly 3 words.")],
            max_tokens=100,  # Reasonable limit
        )
        
        print("✅ SUCCESS: Normal operation works fine")
        print(f"✅ Response: {response}")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Normal operation failed: {type(e).__name__}: {e}")
        return False


async def main() -> bool:
    """Run all tests."""
    
    print("=" * 60)
    print("🚨 MAX_TOKENS HARD FAILURE TEST SUITE")
    print("=" * 60)
    
    # Test 1: MAX_TOKENS should fail hard
    test1_passed = await test_max_tokens_failure()
    
    # Test 2: Normal operation should still work
    test2_passed = await test_normal_operation()
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS:")
    print(f"✅ MAX_TOKENS Hard Failure: {'PASS' if test1_passed else 'FAIL'}")
    print(f"✅ Normal Operation: {'PASS' if test2_passed else 'FAIL'}")
    
    if test1_passed and test2_passed:
        print("🎉 ALL TESTS PASSED!")
        return True
    else:
        print("❌ SOME TESTS FAILED!")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1) ```

## File: `test_runpod_real.py`
```
#!/usr/bin/env python3
"""
Real-world test script for Runpod OpenAI wrappers.
Tests both the LLM client and embedder against actual Runpod endpoints.
"""

import os
import asyncio
import json
import pytest
from pydantic import BaseModel, Field
from runpod_openai_client import RunpodWrapperOpenAIClient
from runpod_openai_embedder import RunpodWrapperOpenAIEmbedder
from graphiti_core.prompts.models import Message

# Configuration
RUNPOD_BASE_URL = "https://api.runpod.ai/v2/3vor3c2onmy4do"
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")

if not RUNPOD_API_KEY:
    raise ValueError("RUNPOD_API_KEY environment variable is required")

# Define a simple response model for structured output
class SimpleResponse(BaseModel):
    summary: str = Field(description="A brief summary of the response")
    sentiment: str = Field(description="The sentiment: positive, negative, or neutral")

@pytest.mark.asyncio
async def test_llm_client() -> None:
    """Test the Runpod OpenAI LLM client wrapper."""
    print("🤖 Testing Runpod OpenAI LLM Client...")
    
    client = RunpodWrapperOpenAIClient(
        base_url=RUNPOD_BASE_URL,
        api_key=RUNPOD_API_KEY,
        model="qwen3",
        temperature=0.1,
        max_tokens=150
    )
    
    # Test 1: Simple text completion
    print("\n📝 Test 1: Simple text completion")
    messages = [
        Message(role="user", content="Explain what artificial intelligence is in one paragraph.")
    ]
    
    try:
        response = await client.generate_response(messages)
        print(f"✅ Response: {json.dumps(response, indent=2)}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Structured output with Pydantic model
    print("\n📊 Test 2: Structured output")
    messages = [
        Message(role="user", content="Tell me about machine learning and classify the sentiment of your response.")
    ]
    
    try:
        response = await client.generate_response(messages, response_model=SimpleResponse)
        print(f"✅ Structured Response: {json.dumps(response, indent=2)}")
    except Exception as e:
        print(f"❌ Error: {e}")

@pytest.mark.asyncio
async def test_embedder() -> None:
    """Test the Runpod OpenAI Embedder wrapper."""
    print("\n🔢 Testing Runpod OpenAI Embedder...")
    
    embedder = RunpodWrapperOpenAIEmbedder(
        base_url=RUNPOD_BASE_URL,
        api_key=RUNPOD_API_KEY,
        embedding_model="bge-m3"
    )
    
    # Test 1: Single embedding
    print("\n📊 Test 1: Single embedding")
    text = "This is a test sentence for embedding."
    
    try:
        embedding = await embedder.create(text)
        print("✅ Embedding created successfully!")
        print(f"   Dimension: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
        print(f"   Last 5 values: {embedding[-5:]}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Batch embeddings (sequential processing)
    print("\n📊 Test 2: Batch embeddings (sequential)")
    texts = [
        "Natural language processing is amazing.",
        "Machine learning transforms data into insights.",
        "Artificial intelligence is the future."
    ]
    
    try:
        embeddings = await embedder.create_batch(texts)
        print("✅ Batch embeddings created successfully!")
        print(f"   Number of embeddings: {len(embeddings)}")
        for i, emb in enumerate(embeddings):
            print(f"   Text {i+1}: dimension {len(emb)}, sample values: {emb[:3]}")
    except Exception as e:
        print(f"❌ Error: {e}")

async def main() -> None:
    """Run all tests."""
    print("🚀 Starting Runpod Wrapper Tests")
    print(f"   Base URL: {RUNPOD_BASE_URL}")
    print(f"   API Key: {'***' + RUNPOD_API_KEY[-8:] if RUNPOD_API_KEY else 'NOT SET'}")
    
    try:
        await test_llm_client()
        await test_embedder()
        print("\n✅ All tests completed!")
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

def run_main() -> None:
    """Wrapper function to run the main async function."""
    asyncio.run(main())

if __name__ == "__main__":
    run_main() ```

## File: `test_runpod_wrappers.py`
```
# To run these tests, you will first need to install the testing dependencies:
# pip install pytest pytest-asyncio respx httpx

import pytest
import respx
import httpx
import json
from pydantic import BaseModel, Field

from runpod_openai_client import RunpodWrapperOpenAIClient
from runpod_openai_embedder import RunpodWrapperOpenAIEmbedder
from graphiti_core.prompts.models import Message

RUNPOD_BASE_URL = "https://api.runpod.ai/v2/test_endpoint"
RUNPOD_API_KEY = "test_api_key"

class MockResponse(BaseModel):
    text: str = Field(...)
    number: int = Field(...)

@pytest.mark.asyncio
async def test_runpod_openai_client_request_wrapping() -> None:
    """
    Tests that the RunpodWrapperOpenAIClient correctly wraps an OpenAI-style request
    into the Runpod /runsync format.
    """
    client = RunpodWrapperOpenAIClient(base_url=RUNPOD_BASE_URL, api_key=RUNPOD_API_KEY)
    
    messages = [Message(role="user", content="Test prompt")]
    
    expected_openai_payload = {
        "model": "qwen3",
        "messages": [{"role": "user", "content": "Test prompt"}],
        "temperature": 0.0,
        "max_tokens": 8192,
        "response_format": {"type": "json_object"},
    }

    expected_runpod_payload = {
        "input": {
            **expected_openai_payload,
            "url_path": "/api/chat"
        }
    }

    mock_runpod_response = {
        "id": "test-id",
        "status": "COMPLETED",
        "output": {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": '{"text": "mocked response", "number": 123}'
                    }
                }
            ]
        }
    }

    async with respx.mock(base_url=RUNPOD_BASE_URL) as mock:
        mock.post("/runsync").mock(return_value=httpx.Response(200, json=mock_runpod_response))
        
        response = await client.generate_response(messages, response_model=MockResponse)
        
        assert mock.calls.call_count == 1
        request = mock.calls.last.request
        assert request.url == f"{RUNPOD_BASE_URL}/runsync"
        assert request.headers["authorization"] == f"Bearer {RUNPOD_API_KEY}"
        assert json.loads(request.content) == expected_runpod_payload
        
        assert isinstance(response, dict)
        assert response["text"] == "mocked response"
        assert response["number"] == 123

@pytest.mark.asyncio
async def test_runpod_openai_embedder_request_wrapping() -> None:
    """
    Tests that the RunpodWrapperOpenAIEmbedder correctly wraps an OpenAI-style request
    into the Runpod /runsync format for embeddings.
    """
    embedder = RunpodWrapperOpenAIEmbedder(base_url=RUNPOD_BASE_URL, api_key=RUNPOD_API_KEY)
    
    input_text = "This is a test sentence."
    
    expected_openai_payload = {
        "input": input_text,
        "model": "bge-m3"
    }

    expected_runpod_payload = {
        "input": {
            **expected_openai_payload,
            "url_path": "/api/embeddings"
        }
    }

    mock_runpod_response = {
        "id": "test-embedding-id",
        "status": "COMPLETED",
        "output": {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [0.1, 0.2, 0.3, 0.4],
                    "index": 0
                }
            ],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 8, "total_tokens": 8}
        }
    }

    async with respx.mock(base_url=RUNPOD_BASE_URL) as mock:
        mock.post("/runsync").mock(return_value=httpx.Response(200, json=mock_runpod_response))
        
        embedding = await embedder.create(input_text)
        
        assert mock.calls.call_count == 1
        request = mock.calls.last.request
        assert request.url == f"{RUNPOD_BASE_URL}/runsync"
        assert request.headers["authorization"] == f"Bearer {RUNPOD_API_KEY}"
        assert json.loads(request.content) == expected_runpod_payload
        
        assert isinstance(embedding, list)
        assert embedding == [0.1, 0.2, 0.3, 0.4]

@pytest.mark.asyncio
async def test_runpod_openai_embedder_non_batching() -> None:
    """
    Tests that the RunpodWrapperOpenAIEmbedder processes batch requests sequentially.
    """
    embedder = RunpodWrapperOpenAIEmbedder(base_url=RUNPOD_BASE_URL, api_key=RUNPOD_API_KEY)
    
    input_texts = ["First sentence.", "Second sentence."]
    
    mock_runpod_response_1 = {"id": "1", "status": "COMPLETED", "output": {"data": [{"embedding": [1.1, 1.2]}]}}
    mock_runpod_response_2 = {"id": "2", "status": "COMPLETED", "output": {"data": [{"embedding": [2.1, 2.2]}]}}

    async with respx.mock(base_url=RUNPOD_BASE_URL) as mock:
        # Set up two different responses for the sequential calls
        mock.post("/runsync").mock(side_effect=[
            httpx.Response(200, json=mock_runpod_response_1),
            httpx.Response(200, json=mock_runpod_response_2),
        ])
        
        embeddings = await embedder.create_batch(input_texts)
        
        assert mock.calls.call_count == 2
        assert embeddings == [[1.1, 1.2], [2.1, 2.2]]
        
        # Verify the first call
        first_request = mock.calls[0].request
        assert json.loads(first_request.content)["input"]["input"] == "First sentence."

        # Verify the second call
        second_request = mock.calls[1].request
        assert json.loads(second_request.content)["input"]["input"] == "Second sentence." ```

## File: `test_search_power.py`
```
#!/usr/bin/env python3
"""
Test script to demonstrate the power difference between basic and hybrid search
"""

import asyncio
import sys
import pytest
from datetime import datetime

# Add the current directory to path for imports
sys.path.append('.')

from app import get_new_graphiti_client_for_request
from graphiti_core.search.search_config import (
    NodeSearchConfig, NodeSearchMethod, SearchConfig, NodeReranker,
    EdgeSearchConfig, EdgeSearchMethod, EdgeReranker,
    CommunitySearchConfig, CommunitySearchMethod, CommunityReranker,
    EpisodeSearchConfig, EpisodeSearchMethod, EpisodeReranker
)
from graphiti_core.search.search_filters import SearchFilters

@pytest.mark.asyncio
async def test_search_configurations() -> None:
    """Test different search configurations to show power differences"""
    
    # Get the graphiti client
    graphiti = get_new_graphiti_client_for_request()
    
    test_query = "AI technology"
    group_ids = ["chrome-tracker-content"]
    
    print("🚗 TESTING SEARCH POWER CONFIGURATIONS")
    print("=" * 60)
    print(f"Query: '{test_query}'")
    print(f"Group IDs: {group_ids}")
    print()
    
    # 1. BASIC BM25 ONLY (Current/Old Way)
    print("1️⃣  BASIC BM25 ONLY (First Gear)")
    print("-" * 40)
    basic_config = SearchConfig(
        node_config=NodeSearchConfig(
            search_methods=[NodeSearchMethod.bm25],  # Only text search
            reranker=NodeReranker.rrf,
            sim_min_score=0.6,  # High threshold
        ),
        limit=10,
        reranker_min_score=0.0
    )
    
    try:
        start_time = datetime.now()
        basic_results = await graphiti.search_(
            query=test_query,
            config=basic_config,
            group_ids=group_ids,
            search_filter=SearchFilters(),
        )
        end_time = datetime.now()
        
        print(f"✅ Nodes found: {len(basic_results.nodes)}")
        print(f"✅ Edges found: {len(basic_results.edges)}")
        print(f"✅ Episodes found: {len(basic_results.episodes)}")
        print(f"✅ Communities found: {len(basic_results.communities)}")
        print(f"⏱️  Time taken: {(end_time - start_time).total_seconds():.3f}s")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    print()
    
    # 2. HYBRID BM25 + VECTOR (New Way)
    print("2️⃣  HYBRID BM25 + VECTOR (Second Gear)")
    print("-" * 40)
    hybrid_config = SearchConfig(
        node_config=NodeSearchConfig(
            search_methods=[NodeSearchMethod.bm25, NodeSearchMethod.cosine_similarity],
            reranker=NodeReranker.rrf,
            sim_min_score=0.3,  # Lower threshold
        ),
        edge_config=EdgeSearchConfig(
            search_methods=[EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
            reranker=EdgeReranker.rrf,
            sim_min_score=0.3,
        ),
        episode_config=EpisodeSearchConfig(
            search_methods=[EpisodeSearchMethod.bm25],
            reranker=EpisodeReranker.rrf,
        ),
        community_config=CommunitySearchConfig(
            search_methods=[CommunitySearchMethod.bm25, CommunitySearchMethod.cosine_similarity],
            reranker=CommunityReranker.rrf,
            sim_min_score=0.3,
        ),
        limit=10,
        reranker_min_score=0.0
    )
    
    try:
        start_time = datetime.now()
        hybrid_results = await graphiti.search_(
            query=test_query,
            config=hybrid_config,
            group_ids=group_ids,
            search_filter=SearchFilters(),
        )
        end_time = datetime.now()
        
        print(f"✅ Nodes found: {len(hybrid_results.nodes)}")
        print(f"✅ Edges found: {len(hybrid_results.edges)}")
        print(f"✅ Episodes found: {len(hybrid_results.episodes)}")
        print(f"✅ Communities found: {len(hybrid_results.communities)}")
        print(f"⏱️  Time taken: {(end_time - start_time).total_seconds():.3f}s")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    print()
    
    # 3. ADVANCED WITH BFS GRAPH TRAVERSAL (Third Gear)
    print("3️⃣  ADVANCED WITH BFS GRAPH TRAVERSAL (Third Gear)")
    print("-" * 40)
    advanced_config = SearchConfig(
        node_config=NodeSearchConfig(
            search_methods=[
                NodeSearchMethod.bm25, 
                NodeSearchMethod.cosine_similarity,
                NodeSearchMethod.bfs  # Graph traversal!
            ],
            reranker=NodeReranker.rrf,
            sim_min_score=0.3,
            bfs_max_depth=2,
        ),
        edge_config=EdgeSearchConfig(
            search_methods=[
                EdgeSearchMethod.bm25, 
                EdgeSearchMethod.cosine_similarity,
                EdgeSearchMethod.bfs  # Graph traversal!
            ],
            reranker=EdgeReranker.rrf,
            sim_min_score=0.3,
            bfs_max_depth=2,
        ),
        episode_config=EpisodeSearchConfig(
            search_methods=[EpisodeSearchMethod.bm25],
            reranker=EpisodeReranker.rrf,
        ),
        community_config=CommunitySearchConfig(
            search_methods=[CommunitySearchMethod.bm25, CommunitySearchMethod.cosine_similarity],
            reranker=CommunityReranker.rrf,
            sim_min_score=0.3,
        ),
        limit=10,
        reranker_min_score=0.0
    )
    
    try:
        start_time = datetime.now()
        advanced_results = await graphiti.search_(
            query=test_query,
            config=advanced_config,
            group_ids=group_ids,
            search_filter=SearchFilters(),
        )
        end_time = datetime.now()
        
        print(f"✅ Nodes found: {len(advanced_results.nodes)}")
        print(f"✅ Edges found: {len(advanced_results.edges)}")
        print(f"✅ Episodes found: {len(advanced_results.episodes)}")
        print(f"✅ Communities found: {len(advanced_results.communities)}")
        print(f"⏱️  Time taken: {(end_time - start_time).total_seconds():.3f}s")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    print()
    print("🏁 SUMMARY")
    print("=" * 60)
    print("The configurations above show the progression from:")
    print("• 1st Gear (Basic): Only text search, high thresholds")
    print("• 2nd Gear (Hybrid): Text + vector search, better thresholds")  
    print("• 3rd Gear (Advanced): + graph traversal for connected concepts")
    print()
    print("💡 Still missing from our 'Ferrari':")
    print("• Cross-encoder reranking (requires compatible model)")
    print("• MMR reranking for diversity")
    print("• Community detection and community-based search")
    print("• Temporal filtering and time-aware queries")
    print("• Multi-hop reasoning across distant concepts")


if __name__ == "__main__":
    asyncio.run(test_search_configurations()) ```

## File: `test_vertex_improvements.py`
```
Content omitted due to reason: BINARY
```

## File: `vertex_ai_client.py`
```
import asyncio
import json
import os
from time import time
from typing import Any, Optional

from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig, GenerateContentResponse, FinishReason
from google.api_core.exceptions import ResourceExhausted, GoogleAPICallError
from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.config import LLMConfig, ModelSize
from graphiti_core.prompts.models import Message
from pydantic import BaseModel, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from session_log_file_writer import SessionLogFileWriter

GCP_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
GCP_LOCATION = os.getenv("GCP_LOCATION", "global") # suggested by google to use 'global' for maximum number of requests

class StopNotEndException(Exception):
    """Exception raised when the generation is stopped but not ended."""
    pass

class VertexAIClient(LLMClient):
    """
    A client for interacting with Google's Vertex AI models using the google-genai SDK.
    
    Features:
    - Concurrency limiting to prevent rate limit errors
    - Queue statistics to monitor waiting requests
    - Automatic retry with exponential backoff
    - Structured output support
    
    Environment Variables:
    - GOOGLE_CLOUD_PROJECT: GCP project ID (required)
    - GCP_LOCATION: GCP region (default: global) # suggested by google to use 'global' for maximum number of requests
    - VERTEX_AI_MAX_CONCURRENT: Max concurrent requests (default: 3)
    - GOOGLE_GENAI_USE_VERTEXAI: Must be "True"
    """

    # Class-level concurrency control and queue statistics
    _semaphore: asyncio.Semaphore | None = None
    _max_concurrent = int(os.getenv("VERTEX_AI_MAX_CONCURRENT", "3"))
    _waiting_count = 0  # Track requests waiting for semaphore
    _queue_lock = asyncio.Lock()  # Protect the waiting counter

    def __init__(
        self,
        config: LLMConfig,
        project_id: str | None = GCP_PROJECT_ID,
        location: str | None = GCP_LOCATION,
        session_log_writer: Optional[SessionLogFileWriter] = None
    ):
        super().__init__(config)
        
        # Validate required environment
        if not project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT environment variable not set.")
        if not location:
            raise ValueError("GCP_LOCATION environment variable not set.")

        # Initialize class-level semaphore (shared across all instances)
        if VertexAIClient._semaphore is None:
            VertexAIClient._semaphore = asyncio.Semaphore(self._max_concurrent)

        # Configure Vertex AI environment
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
        os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
        os.environ["GOOGLE_CLOUD_LOCATION"] = location
        
        # Initialize client
        self.client = genai.Client()
        self.model_name = self.config.model or "gemini-2.5-flash"

        print("🚀 VertexAI Client initialized:")
        print(f"   Model: {self.model_name}")
        print(f"   Max concurrent requests: {self._max_concurrent}")
        print("   Retry strategy: 5 attempts with exponential backoff")

    def _is_gemini_2_5_model(self) -> bool:
        """Check if model supports thinking budget (Gemini 2.5+ models)."""
        return "gemini-2.5" in (self.model_name or "").lower()

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Detect rate limiting errors that should trigger retry logic."""
        if isinstance(error, ResourceExhausted):
            return True
            
        if isinstance(error, GoogleAPICallError):
            error_str = str(error).lower()
            rate_limit_indicators = [
                'rate limit', 'quota exceeded', 'too many requests', 
                'resource exhausted', '429', 'quotaexceeded'
            ]
            return any(indicator in error_str for indicator in rate_limit_indicators)
            
        error_message = str(error).lower()
        return any(pattern in error_message for pattern in [
            'rate limit', 'quota', '429', 'too many requests'
        ])

    def _extract_response_text(self, response: GenerateContentResponse) -> str:
        """Extract text content from Vertex AI response."""
        try:
            # Primary: direct text access (modern SDK)
            if hasattr(response, 'text') and response.text:
                return str(response.text)
            
            # Fallback: extract from candidates structure
            if (hasattr(response, 'candidates') and response.candidates and
                len(response.candidates) > 0):
                candidate = response.candidates[0]
                if (hasattr(candidate, 'content') and candidate.content and
                    hasattr(candidate.content, 'parts') and candidate.content.parts):
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            return str(part.text)
            
            return ""
            
        except Exception as e:
            print(f"⚠️ Could not extract text from response: {e}")
            return ""

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = 65500,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, Any]:
        """
        Generate response with concurrency limiting to prevent rate limits.
        
        The semaphore ensures only N requests run simultaneously across all instances.
        Queue statistics show both active and waiting requests.
        """
        
        # 🔒 CONCURRENCY CONTROL: Wait for available slot
        # At this point _semaphore is guaranteed to be initialized
        semaphore = self._semaphore
        if semaphore is None:
            raise RuntimeError("Semaphore not initialized")
        
        # Check if we need to wait (before entering queue)
        will_wait = semaphore._value == 0
        if will_wait:
            # Atomically increment waiting counter
            async with VertexAIClient._queue_lock:
                VertexAIClient._waiting_count += 1
                active_requests = self._max_concurrent - semaphore._value
                waiting_requests = VertexAIClient._waiting_count
                print(f"⏳ Waiting for slot... ({active_requests}/{self._max_concurrent} active, {waiting_requests} waiting)")
        
        try:
            async with semaphore:
                # If we were waiting, decrement the counter
                if will_wait:
                    async with VertexAIClient._queue_lock:
                        VertexAIClient._waiting_count -= 1
                
                active_requests = self._max_concurrent - semaphore._value
                waiting_requests = VertexAIClient._waiting_count
                print(f"🔒 Request slot acquired ({active_requests}/{self._max_concurrent} active, {waiting_requests} waiting)")
                
                try:
                    return await self._do_generate(messages, response_model, max_tokens)
                finally:
                    waiting_requests = VertexAIClient._waiting_count
                    print(f"🔓 Request slot released ({waiting_requests} still waiting)")
        except asyncio.CancelledError:
            # If we were waiting and the task gets cancelled, decrement counter
            if will_wait:
                async with VertexAIClient._queue_lock:
                    VertexAIClient._waiting_count = max(0, VertexAIClient._waiting_count - 1)
            raise
        except Exception:
            # If we were waiting and an exception occurs, decrement counter
            if will_wait:
                async with VertexAIClient._queue_lock:
                    VertexAIClient._waiting_count = max(0, VertexAIClient._waiting_count - 1)
            raise

    async def _do_generate(
        self,
        messages: list[Message], 
        response_model: type[BaseModel] | None,
        max_tokens: int
    ) -> dict[str, Any]:
        """Handle the actual generation logic (separated for cleaner code)."""
        
        # Prepare request format
        contents = []
        system_instruction = None
        
        for msg in messages:
            if msg.role == "user":
                contents.append({"role": "user", "parts": [{"text": msg.content}]})
            elif msg.role == "assistant":
                contents.append({"role": "model", "parts": [{"text": msg.content}]})
            elif msg.role == "system":
                system_instruction = msg.content

        # Build configuration
        config_params: dict[str, Any] = {
            "temperature": self.config.temperature,
            "max_output_tokens": max_tokens
        }

        if system_instruction:
            config_params["system_instruction"] = system_instruction

        if self._is_gemini_2_5_model():
            config_params["thinking_config"] = ThinkingConfig(
                thinking_budget=0, 
                include_thoughts=False
            )

        if response_model:
            config_params["response_mime_type"] = "application/json"
            config_params["response_schema"] = response_model.model_json_schema()
            print(f"📝 Using structured JSON output: {response_model.__name__}")
            
        config = GenerateContentConfig(**config_params)

        # Generate with retry logic
        @retry(
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=2, min=1, max=30),
            retry=retry_if_exception_type((ResourceExhausted, GoogleAPICallError, StopNotEndException))
        )
        def generate_with_retry() -> str:
            """Sync generation with retry logic."""
            try:
                print(f"🤖 Generating with {self.model_name}...")
                start_time = time()
                
                response: GenerateContentResponse = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config
                )
                token_dict = {}
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    token_dict = {
                        "prompt": response.usage_metadata.prompt_token_count,
                        "cached_content": response.usage_metadata.cached_content_token_count,
                        "candidates": response.usage_metadata.candidates_token_count,
                        "total": response.usage_metadata.total_token_count,
                        "thoughts": response.usage_metadata.thoughts_token_count
                    }

                duration = time() - start_time

                if (hasattr(response, 'prompt_feedback') and 
                    response.prompt_feedback and 
                    hasattr(response.prompt_feedback, 'block_reason') and
                    response.prompt_feedback.block_reason):
                    raise RuntimeError(f"Blocked by safety filters: {response.prompt_feedback.block_reason} after {duration:.2f} seconds with token usage: {token_dict} for request: {str(contents)[:500]}")
                
                if (hasattr(response, 'candidates') and response.candidates and 
                    len(response.candidates) > 0):
                    candidate = response.candidates[0]
                    if (hasattr(candidate, 'finish_reason') and 
                        candidate.finish_reason != FinishReason.STOP):
                        log_data = {
                            "finish_reason": candidate.finish_reason,
                            "duration": duration,
                            "token_dict": token_dict,
                            "request": str(contents),
                            "response": str(response)
                        }
                        if self.session_log_writer:
                            self.session_log_writer.write_session_log("StopNotEndException", log_data)
                        print(f"🔢 Unexpected finish reason: {candidate.finish_reason} after {duration:.2f} seconds with token usage: {token_dict}\n  === request: {str(contents)}\n  === response: {str(response)}")
                        raise StopNotEndException()

                # Extract response
                response_text = self._extract_response_text(response)
                if not response_text:
                    raise RuntimeError("No response text extracted")

                print(f"🔢 Successfully generated {len(response_text)} characters in {duration:.2f} seconds using {self.model_name}. Tokens: {token_dict}\n"
                      f"  - Preview: {response_text[:250]}...")
                
                return response_text
                
            except (ResourceExhausted, GoogleAPICallError) as e:
                if self._is_rate_limit_error(e):
                    print(f"🔄 Rate limit detected, will retry: {e}")
                    raise
                else:
                    print(f"❌ Non-retryable API error: {e}")
                    return ""
                    
            except Exception as e:
                print(f"❌ Generation error: {e}")
                if self._is_rate_limit_error(e):
                    raise ResourceExhausted(f"Potential rate limit: {e}")
                return ""

        # Execute in thread pool (Google client is sync)
        loop = asyncio.get_running_loop()
        response_text = await loop.run_in_executor(None, generate_with_retry)

        # Handle structured output
        if response_model and response_text:
            try:
                parsed = json.loads(response_text)
                response_model.model_validate(parsed)
                print("✅ Structured response validated")
                # Ensure we return dict[str, Any] as expected
                if isinstance(parsed, dict):
                    return parsed
                else:
                    return {"content": str(parsed)}
            except (json.JSONDecodeError, ValidationError) as e:
                print(f"⚠️ JSON validation failed: {e}")
                return {"content": response_text}
        
        return {"content": response_text}
```
