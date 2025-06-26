#!/usr/bin/env python3
"""
Graphiti MCP Server - Exposes Graphiti functionality through the Model Context Protocol (MCP)
Gemini + Ollama version - simplified for just Gemini LLM and Ollama embeddings
"""

import argparse
import asyncio
import logging
import os
from datetime import datetime, timezone
import sys
from typing import Any, cast

from typing_extensions import TypedDict

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from graphiti_core import Graphiti
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder.openai import OpenAIEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import (
    NODE_HYBRID_SEARCH_NODE_DISTANCE,
    NODE_HYBRID_SEARCH_RRF,
)
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

# Import our custom clients - no try/except since we need these to work
from vertex_ai_client import VertexAIClient
from ollama_embedder import NonBatchingOllamaEmbedder
from ollama_reranker_client import OllamaRerankerClient

load_dotenv()

DEFAULT_LLM_MODEL = 'gemini-2.5-flash-lite-preview-06-17'
SMALL_LLM_MODEL = 'gemini-2.5-flash-lite-preview-06-17'
DEFAULT_EMBEDDER_MODEL = 'nomic-embed-text'


class Requirement(BaseModel):
    """A Requirement represents a specific need, feature, or functionality that a product or service must fulfill."""

    project_name: str = Field(
        ...,
        description='The name of the project to which the requirement belongs.',
    )
    description: str = Field(
        ...,
        description='Description of the requirement. Only use information mentioned in the context to write this description.',
    )


class Preference(BaseModel):
    """A Preference represents a user's expressed like, dislike, or preference for something."""

    category: str = Field(
        ...,
        description="The category of the preference. (e.g., 'Brands', 'Food', 'Music')",
    )
    description: str = Field(
        ...,
        description='Brief description of the preference. Only use information mentioned in the context to write this description.',
    )


class Procedure(BaseModel):
    """A Procedure informing the agent what actions to take or how to perform in certain scenarios."""

    description: str = Field(
        ...,
        description='Brief description of the procedure. Only use information mentioned in the context to write this description.',
    )


ENTITY_TYPES: dict[str, BaseModel] = {
    'Requirement': Requirement,  # type: ignore
    'Preference': Preference,  # type: ignore
    'Procedure': Procedure,  # type: ignore
}


def ensure_no_embeddings(data: dict[str, Any]) -> dict[str, Any]:
    """Utility function to ensure embeddings are never exposed in MCP responses.
    
    This provides an additional safety layer to prevent fact_embedding, name_embedding,
    summary_embedding, or any other embedding fields from being accidentally returned in responses.
    
    Args:
        data: Dictionary that might contain embedding data
        
    Returns:
        Dictionary with all embedding fields removed recursively
    """
    # List of all known embedding fields that should never be exposed
    embedding_fields = [
        'fact_embedding',      # Entity edge fact embeddings
        'name_embedding',      # Entity node name embeddings  
        'summary_embedding',   # Entity node summary embeddings
        'embedding',           # Generic embedding field
    ]
    
    # Remove any embedding fields that might be present at top level
    for field in embedding_fields:
        data.pop(field, None)
    
    # Also remove any field that ends with '_embedding' as a catch-all
    keys_to_remove = [key for key in data.keys() if key.endswith('_embedding')]
    for key in keys_to_remove:
        data.pop(key, None)
    
    # Recursively clean nested dictionaries
    for key, value in data.items():
        if isinstance(value, dict):
            data[key] = ensure_no_embeddings(value)
        elif isinstance(value, list):
            # Clean any dictionaries in lists
            cleaned_list = []
            for item in value:
                if isinstance(item, dict):
                    cleaned_list.append(ensure_no_embeddings(item))
                else:
                    cleaned_list.append(item)
            data[key] = cleaned_list
    
    return data


# Type definitions for API responses
class ErrorResponse(TypedDict):
    error: str


class SuccessResponse(TypedDict):
    message: str


class NodeResult(TypedDict):
    uuid: str
    name: str
    summary: str
    labels: list[str]
    group_id: str
    created_at: str
    attributes: dict[str, Any]


class NodeSearchResponse(TypedDict):
    message: str
    nodes: list[NodeResult]


class FactSearchResponse(TypedDict):
    message: str
    facts: list[dict[str, Any]]


class EpisodeSearchResponse(TypedDict):
    message: str
    episodes: list[dict[str, Any]]


class StatusResponse(TypedDict):
    status: str
    message: str


class GraphitiLLMConfig(BaseModel):
    """Configuration for the Gemini LLM client."""

    model: str = DEFAULT_LLM_MODEL
    small_model: str = SMALL_LLM_MODEL
    temperature: float = 0.3
    project_id: str | None = None
    location: str = "global"

    @classmethod
    def from_env(cls) -> 'GraphitiLLMConfig':
        """Create LLM configuration from environment variables."""
        model_env = os.environ.get('MODEL_NAME', '')
        model = model_env if model_env.strip() else DEFAULT_LLM_MODEL

        small_model_env = os.environ.get('SMALL_MODEL_NAME', '')
        small_model = small_model_env if small_model_env.strip() else SMALL_LLM_MODEL

        return cls(
            model=model,
            small_model=small_model,
            temperature=float(os.environ.get('LLM_TEMPERATURE', '0.3')),
            project_id=os.environ.get('GOOGLE_CLOUD_PROJECT'),
            location=os.environ.get('GOOGLE_CLOUD_LOCATION', 'global'),
        )

    @classmethod
    def from_cli_and_env(cls, args: argparse.Namespace) -> 'GraphitiLLMConfig':
        """Create LLM configuration from CLI arguments, falling back to environment variables."""
        config = cls.from_env()

        if hasattr(args, 'model') and args.model:
            if args.model.strip():
                config.model = args.model

        if hasattr(args, 'small_model') and args.small_model:
            if args.small_model.strip():
                config.small_model = args.small_model

        if hasattr(args, 'temperature') and args.temperature is not None:
            config.temperature = args.temperature

        return config

    def create_client(self):
        """Create a Gemini LLM client based on this configuration."""
        if not self.project_id:
            raise ValueError('GOOGLE_CLOUD_PROJECT must be set when using Vertex AI Gemini')
        
        return VertexAIClient(
            config=LLMConfig(
                model=self.model,
                temperature=self.temperature,
            ),
            project_id=self.project_id,
            location=self.location,
        )


class GraphitiEmbedderConfig(BaseModel):
    """Configuration for the Ollama embedder client."""

    model: str = DEFAULT_EMBEDDER_MODEL
    base_url: str = "http://localhost:11434"

    @classmethod
    def from_env(cls) -> 'GraphitiEmbedderConfig':
        """Create embedder configuration from environment variables."""
        model_env = os.environ.get('EMBEDDER_MODEL_NAME', '')
        model = model_env if model_env.strip() else DEFAULT_EMBEDDER_MODEL

        base_url_env = os.environ.get('OLLAMA_BASE_URL', '')
        base_url = base_url_env if base_url_env.strip() else "http://localhost:11434"

        return cls(
            model=model,
            base_url=base_url,
        )

    def create_client(self):
        """Create an Ollama embedder client based on this configuration."""
        return NonBatchingOllamaEmbedder(
            config=OpenAIEmbedderConfig(
                api_key="ollama",
                base_url=self.base_url,
                embedding_model=self.model
            )
        )


class Neo4jConfig(BaseModel):
    """Configuration for Neo4j database connection."""

    uri: str = 'bolt://localhost:7687'
    user: str = 'neo4j'
    password: str = 'password'

    @classmethod
    def from_env(cls) -> 'Neo4jConfig':
        """Create Neo4j configuration from environment variables."""
        return cls(
            uri=os.environ.get('NEO4J_URI', 'bolt://localhost:7687'),
            user=os.environ.get('NEO4J_USER', 'neo4j'),
            password=os.environ.get('NEO4J_PASSWORD', 'password'),
        )


class GraphitiConfig(BaseModel):
    """Configuration for Graphiti client."""

    llm: GraphitiLLMConfig = Field(default_factory=GraphitiLLMConfig)
    embedder: GraphitiEmbedderConfig = Field(default_factory=GraphitiEmbedderConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    group_id: str | None = None
    use_custom_entities: bool = False
    destroy_graph: bool = False

    @classmethod
    def from_env(cls) -> 'GraphitiConfig':
        """Create Graphiti configuration from environment variables."""
        return cls(
            llm=GraphitiLLMConfig.from_env(),
            embedder=GraphitiEmbedderConfig.from_env(),
            neo4j=Neo4jConfig.from_env(),
        )

    @classmethod
    def from_cli_and_env(cls, args: argparse.Namespace) -> 'GraphitiConfig':
        """Create Graphiti configuration from CLI arguments, falling back to environment variables."""
        config = cls.from_env()
        
        config.llm = GraphitiLLMConfig.from_cli_and_env(args)
        
        if hasattr(args, 'group_id') and args.group_id:
            config.group_id = args.group_id
            
        if hasattr(args, 'use_custom_entities'):
            print(f"DEBUG: use_custom_entities = {args.use_custom_entities}")
            config.use_custom_entities = args.use_custom_entities
        else:
            print("DEBUG: use_custom_entities argument not found")
            
        if hasattr(args, 'destroy_graph'):
            config.destroy_graph = args.destroy_graph
            
        return config


class MCPConfig(BaseModel):
    """Configuration for MCP server."""

    transport: str = 'sse'

    @classmethod
    def from_cli(cls, args: argparse.Namespace) -> 'MCPConfig':
        """Create MCP configuration from CLI arguments."""
        transport = 'sse'
        if hasattr(args, 'transport') and args.transport:
            transport = args.transport
        return cls(transport=transport)


# Global state variables
config: GraphitiConfig
graphiti_client: Graphiti | None = None
mcp = FastMCP("Graphiti MCP Server")
logger = logging.getLogger(__name__)

# Episode queues for processing
episode_queues: dict[str, asyncio.Queue] = {}
queue_workers: dict[str, bool] = {}

# Set default host for Docker accessibility
mcp.settings.host = "0.0.0.0"


def create_graphiti_client():
    """Create a Graphiti client following the kg_code.md pattern."""
    
    # Use Vertex AI Client (from kg_code.md)
    llm_client = VertexAIClient(
        config=LLMConfig(
            model=config.llm.model,
            temperature=config.llm.temperature,
        )
    )

    # Using Ollama for embeddings (from kg_code.md)
    embedder = NonBatchingOllamaEmbedder(
        config=OpenAIEmbedderConfig(
            api_key="ollama",
            base_url=config.embedder.base_url,
            embedding_model=config.embedder.model
        )
    )

    # Use Ollama cross-encoder (from kg_code.md)
    from ollama_reranker_client import OllamaRerankerClient
    cross_encoder = OllamaRerankerClient(
        base_url=config.embedder.base_url,
        model_name="bge-m3"
    )

    # Initialize Graphiti with Vertex AI client and Ollama cross-encoder (from kg_code.md)
    logger.info(f"Initializing Graphiti with Neo4j URI: {config.neo4j.uri}, Vertex AI Client, and Ollama Cross-Encoder")
    return Graphiti(
        config.neo4j.uri,
        config.neo4j.user,
        config.neo4j.password,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder
    )


async def initialize_graphiti():
    """Initialize the Graphiti client with the configured settings."""
    global graphiti_client, config

    try:
        # Google Cloud credentials are handled by environment variables
        # Docker: GOOGLE_APPLICATION_CREDENTIALS=/app/adc.json (set in docker-compose.yml)
        # Local: GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/application_default_credentials.json (set in shell)
        if 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
            logger.info(f'Using Google Cloud credentials: {os.environ["GOOGLE_APPLICATION_CREDENTIALS"]}')
        else:
            logger.warning('GOOGLE_APPLICATION_CREDENTIALS not set')

        if not config.neo4j.uri or not config.neo4j.user or not config.neo4j.password:
            raise ValueError('NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set')

        graphiti_client = create_graphiti_client()

        if config.destroy_graph:
            logger.info('Destroying graph...')
            await clear_data(graphiti_client.driver)

        await graphiti_client.build_indices_and_constraints()
        logger.info('Graphiti client initialized successfully')

        logger.info(f'Using Gemini model: {config.llm.model}')
        logger.info(f'Using temperature: {config.llm.temperature}')
        logger.info(f'Using group_id: {config.group_id}')
        logger.info(
            f'Custom entity extraction: {"enabled" if config.use_custom_entities else "disabled"}'
        )

    except Exception as e:
        logger.error(f'Failed to initialize Graphiti: {str(e)}')
        raise


def format_fact_result(edge: EntityEdge) -> dict[str, Any]:
    """Format an entity edge into a readable result.
    
    IMPORTANT: This function explicitly excludes ALL embedding fields to prevent large embedding 
    vectors from being returned in responses, which would consume unnecessary context space 
    and provide no value to the caller.
    
    Args:
        edge: The EntityEdge to format

    Returns:
        A dictionary representation of the edge with serialized dates and excluded embeddings
    """
    result = edge.model_dump(
        mode='json',
        exclude={
            'fact_embedding',      # Entity edge fact embeddings
            'name_embedding',      # Entity node name embeddings (if present)
            'summary_embedding',   # Entity node summary embeddings (if present)
        },
    )
    
    # Additional safety: use utility function to ensure no embeddings leak through
    # This is especially important for the 'attributes' field which can contain fact_embedding
    result = ensure_no_embeddings(result)
    
    return result


async def process_episode_queue(group_id: str):
    """Process episodes for a specific group_id sequentially."""
    global queue_workers

    logger.info(f'Starting episode queue worker for group_id: {group_id}')
    queue_workers[group_id] = True

    try:
        while True:
            process_func = await episode_queues[group_id].get()

            try:
                await process_func()
            except Exception as e:
                logger.error(f'Error processing queued episode for group_id {group_id}: {str(e)}')
            finally:
                episode_queues[group_id].task_done()
    except asyncio.CancelledError:
        logger.info(f'Episode queue worker for group_id {group_id} was cancelled')
    except Exception as e:
        logger.error(f'Unexpected error in queue worker for group_id {group_id}: {str(e)}')
    finally:
        queue_workers[group_id] = False
        logger.info(f'Stopped episode queue worker for group_id: {group_id}')


@mcp.tool()
async def add_memory(
    name: str,
    episode_body: str,
    group_id: str | None = None,
    source: str = 'text',
    source_description: str = '',
    uuid: str | None = None,
) -> SuccessResponse | ErrorResponse:
    """Add an episode to memory. This is the primary way to add information to the graph."""
    global graphiti_client, episode_queues, queue_workers

    if graphiti_client is None:
        return {'error': 'Graphiti client not initialized'}

    try:
        source_type = EpisodeType.text
        if source.lower() == 'message':
            source_type = EpisodeType.message
        elif source.lower() == 'json':
            source_type = EpisodeType.json

        effective_group_id = group_id if group_id is not None else config.group_id
        group_id_str = str(effective_group_id) if effective_group_id is not None else ''

        assert graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        async def process_episode():
            try:
                logger.info(f"Processing queued episode '{name}' for group_id: {group_id_str}")
                entity_types = ENTITY_TYPES if config.use_custom_entities else {}

                await client.add_episode(
                    name=name,
                    episode_body=episode_body,
                    source=source_type,
                    source_description=source_description,
                    group_id=group_id_str,
                    uuid=uuid,
                    reference_time=datetime.now(timezone.utc),
                    entity_types=entity_types,
                )
                logger.info(f"Episode '{name}' processed successfully")
            except Exception as e:
                error_msg = str(e)
                logger.error(
                    f"Error processing episode '{name}' for group_id {group_id_str}: {error_msg}"
                )

        if group_id_str not in episode_queues:
            episode_queues[group_id_str] = asyncio.Queue()

        await episode_queues[group_id_str].put(process_episode)

        if not queue_workers.get(group_id_str, False):
            asyncio.create_task(process_episode_queue(group_id_str))

        return {
            'message': f"Episode '{name}' queued for processing (position: {episode_queues[group_id_str].qsize()})"
        }
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error queuing episode task: {error_msg}')
        return {'error': f'Error queuing episode task: {error_msg}'}


@mcp.tool()
async def search_memory_nodes(
    query: str,
    group_ids: list[str] | None = None,
    max_nodes: int = 10,
    center_node_uuid: str | None = None,
    entity: str = '',
) -> NodeSearchResponse | ErrorResponse:
    """Search the graph memory for relevant node summaries."""
    global graphiti_client

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        effective_group_ids = (
            group_ids if group_ids is not None else [config.group_id] if config.group_id else []
        )

        if center_node_uuid is not None:
            search_config = NODE_HYBRID_SEARCH_NODE_DISTANCE.model_copy(deep=True)
        else:
            search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        search_config.limit = max_nodes

        filters = SearchFilters()
        if entity != '':
            filters.node_labels = [entity]

        assert graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        search_results = await client._search(
            query=query,
            config=search_config,
            group_ids=effective_group_ids,
            center_node_uuid=center_node_uuid,
            search_filter=filters,
        )

        if not search_results.nodes:
            return NodeSearchResponse(message='No relevant nodes found', nodes=[])

        formatted_nodes: list[NodeResult] = [
            cast(NodeResult, ensure_no_embeddings({
                'uuid': node.uuid,
                'name': node.name,
                'summary': node.summary if hasattr(node, 'summary') else '',
                'labels': node.labels if hasattr(node, 'labels') else [],
                'group_id': node.group_id,
                'created_at': node.created_at.isoformat(),
                'attributes': node.attributes if hasattr(node, 'attributes') else {},
            }))
            for node in search_results.nodes
        ]

        return NodeSearchResponse(message='Nodes retrieved successfully', nodes=formatted_nodes)
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error searching nodes: {error_msg}')
        return ErrorResponse(error=f'Error searching nodes: {error_msg}')


@mcp.tool()
async def search_memory_facts(
    query: str,
    group_ids: list[str] | None = None,
    max_facts: int = 10,
    center_node_uuid: str | None = None,
) -> FactSearchResponse | ErrorResponse:
    """Search the graph memory for relevant facts."""
    global graphiti_client

    if graphiti_client is None:
        return {'error': 'Graphiti client not initialized'}

    try:
        effective_group_ids = (
            group_ids if group_ids is not None else [config.group_id] if config.group_id else []
        )

        assert graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        relevant_edges = await client.search(
            group_ids=effective_group_ids,
            query=query,
            num_results=max_facts,
            center_node_uuid=center_node_uuid,
        )

        if not relevant_edges:
            return {'message': 'No relevant facts found', 'facts': []}

        facts = [format_fact_result(edge) for edge in relevant_edges]
        return {'message': 'Facts retrieved successfully', 'facts': facts}
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error searching facts: {error_msg}')
        return {'error': f'Error searching facts: {error_msg}'}


@mcp.tool()
async def clear_graph() -> SuccessResponse | ErrorResponse:
    """Clear all data from the graph memory and rebuild indices."""
    global graphiti_client

    if graphiti_client is None:
        return {'error': 'Graphiti client not initialized'}

    try:
        assert graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        await clear_data(client.driver)
        await client.build_indices_and_constraints()
        return {'message': 'Graph cleared successfully and indices rebuilt'}
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error clearing graph: {error_msg}')
        return {'error': f'Error clearing graph: {error_msg}'}


@mcp.tool()
async def export_memories(
    group_ids: list[str] | None = None,
    include_entities: bool = True,
    include_relationships: bool = True,
    save_to_file: bool = True
) -> dict[str, Any] | ErrorResponse:
    """Export all memories in human-readable JSON format for backup/migration."""
    global graphiti_client

    if graphiti_client is None:
        return {'error': 'Graphiti client not initialized'}

    try:
        import json
        import os
        from pathlib import Path
        
        assert graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        logger.info("Starting memory export...")

        # Determine which group_ids to export
        effective_group_ids = group_ids if group_ids is not None else []
        if not effective_group_ids:
            # Get all group_ids if none specified
            records, _, _ = await client.driver.execute_query(
                """
                MATCH (n) WHERE n.group_id IS NOT NULL
                RETURN DISTINCT n.group_id AS group_id
                """,
                database_="neo4j"
            )
            effective_group_ids = [record['group_id'] for record in records]

        export_data = {
            "export_metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "format_version": "1.0",
                "group_ids": effective_group_ids,
                "total_groups": len(effective_group_ids)
            },
            "episodes": [],
            "entities": [] if include_entities else None,
            "relationships": [] if include_relationships else None
        }

        # Export Episodes (the main memories)
        logger.info("Exporting episodes...")
        group_filter = "WHERE e.group_id IN $group_ids" if effective_group_ids else ""
        
        episode_query = f"""
            MATCH (e:Episodic)
            {group_filter}
            RETURN e.uuid AS uuid, e.name AS name, e.content AS content,
                   e.source AS source, e.source_description AS source_description,
                   e.group_id AS group_id, e.created_at AS created_at,
                   e.valid_at AS valid_at, e.entity_edges AS entity_edges
            ORDER BY e.created_at ASC
        """
        
        episode_records, _, _ = await client.driver.execute_query(
            episode_query,
            group_ids=effective_group_ids,
            database_="neo4j"
        )

        for record in episode_records:
            episode_data = {
                "uuid": record['uuid'],
                "name": record['name'],
                "content": record['content'],
                "source": record['source'],
                "source_description": record['source_description'],
                "group_id": record['group_id'],
                "created_at": record['created_at'].isoformat() if record['created_at'] else None,
                "valid_at": record['valid_at'].isoformat() if record['valid_at'] else None,
                "entity_edges": record.get('entity_edges', [])
            }
            export_data["episodes"].append(episode_data)

        # Export Entities (if requested)
        if include_entities:
            logger.info("Exporting entities...")
            entity_filter = "WHERE n.group_id IN $group_ids" if effective_group_ids else ""
            
            entity_query = f"""
                MATCH (n:Entity)
                {entity_filter}
                RETURN n.uuid AS uuid, n.name AS name, n.summary AS summary,
                       n.group_id AS group_id, n.created_at AS created_at,
                       labels(n) AS labels, properties(n) AS properties
                ORDER BY n.created_at ASC
            """
            
            entity_records, _, _ = await client.driver.execute_query(
                entity_query,
                group_ids=effective_group_ids,
                database_="neo4j"
            )

            for record in entity_records:
                # Filter out embedding and system properties
                clean_properties = {
                    k: v for k, v in record['properties'].items() 
                    if not k.endswith('_embedding') and k not in ['uuid', 'name', 'summary', 'group_id', 'created_at']
                }
                
                entity_data = {
                    "uuid": record['uuid'],
                    "name": record['name'],
                    "summary": record['summary'],
                    "group_id": record['group_id'],
                    "created_at": record['created_at'].isoformat() if record['created_at'] else None,
                    "labels": [label for label in record['labels'] if label != 'Entity'],
                    "attributes": clean_properties
                }
                export_data["entities"].append(entity_data)

        # Export Relationships (if requested)
        if include_relationships:
            logger.info("Exporting relationships...")
            relationship_filter = "WHERE r.group_id IN $group_ids" if effective_group_ids else ""
            
            relationship_query = f"""
                MATCH (source:Entity)-[r:RELATES_TO]->(target:Entity)
                {relationship_filter}
                RETURN r.uuid AS uuid, r.name AS name, r.fact AS fact,
                       r.group_id AS group_id, r.created_at AS created_at,
                       r.episodes AS episodes, r.expired_at AS expired_at,
                       r.valid_at AS valid_at, r.invalid_at AS invalid_at,
                       source.uuid AS source_uuid, source.name AS source_name,
                       target.uuid AS target_uuid, target.name AS target_name,
                       properties(r) AS properties
                ORDER BY r.created_at ASC
            """
            
            relationship_records, _, _ = await client.driver.execute_query(
                relationship_query,
                group_ids=effective_group_ids,
                database_="neo4j"
            )

            for record in relationship_records:
                # Filter out embedding properties
                clean_properties = {
                    k: v for k, v in record['properties'].items() 
                    if not k.endswith('_embedding') and k not in [
                        'uuid', 'name', 'fact', 'group_id', 'created_at', 
                        'episodes', 'expired_at', 'valid_at', 'invalid_at'
                    ]
                }
                
                relationship_data = {
                    "uuid": record['uuid'],
                    "name": record['name'],
                    "fact": record['fact'],
                    "group_id": record['group_id'],
                    "created_at": record['created_at'].isoformat() if record['created_at'] else None,
                    "expired_at": record['expired_at'].isoformat() if record['expired_at'] else None,
                    "valid_at": record['valid_at'].isoformat() if record['valid_at'] else None,
                    "invalid_at": record['invalid_at'].isoformat() if record['invalid_at'] else None,
                    "episodes": record.get('episodes', []),
                    "source": {
                        "uuid": record['source_uuid'],
                        "name": record['source_name']
                    },
                    "target": {
                        "uuid": record['target_uuid'],
                        "name": record['target_name']
                    },
                    "attributes": clean_properties
                }
                export_data["relationships"].append(relationship_data)

        # Add summary statistics
        export_data["export_metadata"]["statistics"] = {
            "total_episodes": len(export_data["episodes"]),
            "total_entities": len(export_data["entities"]) if include_entities else 0,
            "total_relationships": len(export_data["relationships"]) if include_relationships else 0
        }

        # Save to file if requested
        if save_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"graphiti_export_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            export_data["export_metadata"]["saved_to_file"] = filename
            logger.info(f"Export saved to file: {filename}")

        logger.info(f"Export completed: {export_data['export_metadata']['statistics']}")
        
        return export_data

    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error during memory export: {error_msg}')
        return {'error': f'Error during memory export: {error_msg}'}


@mcp.tool()
async def import_episodes_from_export(
    export_data: dict[str, Any],
    target_group_id: str | None = None,
    skip_existing: bool = True
) -> SuccessResponse | ErrorResponse:
    """Import episodes from exported data back into the system."""
    global graphiti_client

    if graphiti_client is None:
        return {'error': 'Graphiti client not initialized'}

    try:
        if 'episodes' not in export_data:
            return {'error': 'No episodes found in export data'}

        episodes = export_data['episodes']
        imported_count = 0
        skipped_count = 0
        
        for episode in episodes:
            try:
                # Use target_group_id if provided, otherwise use original
                group_id = target_group_id if target_group_id else episode.get('group_id')
                
                # Check if episode already exists (if skip_existing is True)
                if skip_existing:
                    existing_check, _, _ = await graphiti_client.driver.execute_query(
                        "MATCH (e:Episodic {uuid: $uuid}) RETURN COUNT(e) AS count",
                        uuid=episode['uuid'],
                        database_="neo4j"
                    )
                    if existing_check[0]['count'] > 0:
                        skipped_count += 1
                        continue

                # Import via add_memory
                result = await add_memory(
                    name=episode['name'],
                    episode_body=episode['content'],
                    group_id=group_id,
                    source=episode.get('source', 'text'),
                    source_description=episode.get('source_description', ''),
                    uuid=episode['uuid']
                )
                
                if 'error' not in result:
                    imported_count += 1
                else:
                    logger.warning(f"Failed to import episode {episode['uuid']}: {result['error']}")
                    
            except Exception as e:
                logger.error(f"Error importing episode {episode.get('uuid', 'unknown')}: {str(e)}")

        return {
            'message': f'Import completed: {imported_count} episodes imported, {skipped_count} skipped'
        }

    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error during import: {error_msg}')
        return {'error': f'Error during import: {error_msg}'}


async def initialize_server() -> MCPConfig:
    """Parse CLI arguments and initialize the Graphiti server configuration."""
    global config

    parser = argparse.ArgumentParser(
        description='Run the Graphiti MCP server with Gemini and Ollama'
    )
    parser.add_argument(
        '--group-id',
        help='Namespace for the graph. If not provided, defaults to "default".',
    )
    parser.add_argument(
        '--transport',
        choices=['sse', 'stdio'],
        default='sse',
        help='Transport to use for communication with the client. (default: sse)',
    )
    parser.add_argument(
        '--model', help=f'Model name to use with the LLM client. (default: {DEFAULT_LLM_MODEL})'
    )
    parser.add_argument(
        '--small-model',
        help=f'Small model name to use with the LLM client. (default: {SMALL_LLM_MODEL})',
    )
    parser.add_argument(
        '--temperature',
        type=float,
        help='Temperature setting for the LLM (0.0-2.0). (default: 0.3)',
    )
    parser.add_argument('--destroy-graph', action='store_true', help='Destroy all Graphiti graphs')
    parser.add_argument(
        '--use-custom-entities',
        action='store_true',
        help='Enable entity extraction using the predefined ENTITY_TYPES',
    )
    parser.add_argument(
        '--host',
        default=os.environ.get('MCP_SERVER_HOST', '0.0.0.0'),
        help='Host to bind the MCP server to (default: 0.0.0.0 for Docker)',
    )

    args = parser.parse_args()

    config = GraphitiConfig.from_cli_and_env(args)

    await initialize_graphiti()

    if args.host:
        logger.info(f'Setting MCP server host to: {args.host}')
        mcp.settings.host = args.host

    return MCPConfig.from_cli(args)


async def run_mcp_server():
    """Run the MCP server."""
    mcp_config = await initialize_server()

    logger.info(f'Starting MCP server with transport: {mcp_config.transport}')
    if mcp_config.transport == 'stdio':
        await mcp.run_stdio_async()
    elif mcp_config.transport == 'sse':
        logger.info(
            f'Running MCP server with SSE transport on {mcp.settings.host}:{mcp.settings.port}'
        )
        await mcp.run_sse_async()


def main():
    """Main entry point."""
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_mcp_server())


if __name__ == '__main__':
    main() 