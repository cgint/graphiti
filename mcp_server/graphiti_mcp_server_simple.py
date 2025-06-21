#!/usr/bin/env python3
"""
Graphiti MCP Server - Exposes Graphiti functionality through the Model Context Protocol (MCP)
Simplified version for regular OpenAI API only (no Azure support)
"""

import argparse
import asyncio
import logging
import os
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, TypedDict, cast

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from graphiti_core import Graphiti
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.search.search_config_recipes import (
    NODE_HYBRID_SEARCH_NODE_DISTANCE,
    NODE_HYBRID_SEARCH_RRF,
)
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

load_dotenv()

DEFAULT_LLM_MODEL = 'gpt-4.1-mini'
SMALL_LLM_MODEL = 'gpt-4.1-nano'
DEFAULT_EMBEDDER_MODEL = 'text-embedding-3-small'


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
        Dictionary with all embedding fields removed
    """
    # List of all known embedding fields that should never be exposed
    embedding_fields = [
        'fact_embedding',      # Entity edge fact embeddings
        'name_embedding',      # Entity node name embeddings  
        'summary_embedding',   # Entity node summary embeddings
        'embedding',           # Generic embedding field
    ]
    
    # Remove any embedding fields that might be present
    for field in embedding_fields:
        data.pop(field, None)
    
    # Also remove any field that ends with '_embedding' as a catch-all
    keys_to_remove = [key for key in data.keys() if key.endswith('_embedding')]
    for key in keys_to_remove:
        data.pop(key, None)
    
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
    """Configuration for the LLM client."""

    api_key: str | None = None
    model: str = DEFAULT_LLM_MODEL
    small_model: str = SMALL_LLM_MODEL
    temperature: float = 0.0

    @classmethod
    def from_env(cls) -> 'GraphitiLLMConfig':
        """Create LLM configuration from environment variables."""
        model_env = os.environ.get('MODEL_NAME', '')
        model = model_env if model_env.strip() else DEFAULT_LLM_MODEL

        small_model_env = os.environ.get('SMALL_MODEL_NAME', '')
        small_model = small_model_env if small_model_env.strip() else SMALL_LLM_MODEL

        return cls(
            api_key=os.environ.get('OPENAI_API_KEY'),
            model=model,
            small_model=small_model,
            temperature=float(os.environ.get('LLM_TEMPERATURE', '0.0')),
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

    def create_client(self) -> LLMClient:
        """Create an LLM client based on this configuration."""
        if not self.api_key:
            raise ValueError('OPENAI_API_KEY must be set when using OpenAI API')

        llm_client_config = LLMConfig(
            api_key=self.api_key, model=self.model, small_model=self.small_model
        )
        llm_client_config.temperature = self.temperature

        return OpenAIClient(config=llm_client_config)


class GraphitiEmbedderConfig(BaseModel):
    """Configuration for the embedder client."""

    model: str = DEFAULT_EMBEDDER_MODEL
    api_key: str | None = None

    @classmethod
    def from_env(cls) -> 'GraphitiEmbedderConfig':
        """Create embedder configuration from environment variables."""
        model_env = os.environ.get('EMBEDDER_MODEL_NAME', '')
        model = model_env if model_env.strip() else DEFAULT_EMBEDDER_MODEL

        return cls(
            model=model,
            api_key=os.environ.get('OPENAI_API_KEY'),
        )

    def create_client(self) -> EmbedderClient | None:
        """Create an embedder client based on this configuration."""
        if not self.api_key:
            return None

        embedder_config = OpenAIEmbedderConfig(api_key=self.api_key, embedding_model=self.model)
        return OpenAIEmbedder(config=embedder_config)


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
        """Create a configuration instance from environment variables."""
        return cls(
            llm=GraphitiLLMConfig.from_env(),
            embedder=GraphitiEmbedderConfig.from_env(),
            neo4j=Neo4jConfig.from_env(),
        )

    @classmethod
    def from_cli_and_env(cls, args: argparse.Namespace) -> 'GraphitiConfig':
        """Create configuration from CLI arguments, falling back to environment variables."""
        config = cls.from_env()

        if args.group_id:
            config.group_id = args.group_id
        else:
            config.group_id = 'default'

        config.use_custom_entities = args.use_custom_entities
        config.destroy_graph = args.destroy_graph
        config.llm = GraphitiLLMConfig.from_cli_and_env(args)

        return config


class MCPConfig(BaseModel):
    """Configuration for MCP server."""

    transport: str = 'sse'

    @classmethod
    def from_cli(cls, args: argparse.Namespace) -> 'MCPConfig':
        """Create MCP configuration from CLI arguments."""
        return cls(transport=args.transport)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Create global config instance
config = GraphitiConfig()

# MCP server instructions
GRAPHITI_MCP_INSTRUCTIONS = """
Graphiti is a memory service for AI agents built on a knowledge graph. Graphiti performs well
with dynamic data such as user interactions, changing enterprise data, and external information.

Graphiti transforms information into a richly connected knowledge network, allowing you to 
capture relationships between concepts, entities, and information. The system organizes data as episodes 
(content snippets), nodes (entities), and facts (relationships between entities), creating a dynamic, 
queryable memory store that evolves with new information.

Key capabilities:
1. Add episodes (text, messages, or JSON) to the knowledge graph with the add_memory tool
2. Search for nodes (entities) in the graph using natural language queries with search_memory_nodes
3. Find relevant facts (relationships between entities) with search_memory_facts
4. Retrieve specific entity edges or episodes by UUID
5. Manage the knowledge graph with tools like delete_episode, delete_entity_edge, and clear_graph
"""

# MCP server instance
mcp = FastMCP(
    'Graphiti Agent Memory',
    instructions=GRAPHITI_MCP_INSTRUCTIONS,
)

# Initialize Graphiti client
graphiti_client: Graphiti | None = None


async def initialize_graphiti():
    """Initialize the Graphiti client with the configured settings."""
    global graphiti_client, config

    try:
        llm_client = config.llm.create_client()
        if not llm_client and config.use_custom_entities:
            raise ValueError('OPENAI_API_KEY must be set when custom entities are enabled')

        if not config.neo4j.uri or not config.neo4j.user or not config.neo4j.password:
            raise ValueError('NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set')

        embedder_client = config.embedder.create_client()

        graphiti_client = Graphiti(
            uri=config.neo4j.uri,
            user=config.neo4j.user,
            password=config.neo4j.password,
            llm_client=llm_client,
            embedder=embedder_client,
        )

        if config.destroy_graph:
            logger.info('Destroying graph...')
            await clear_data(graphiti_client.driver)

        await graphiti_client.build_indices_and_constraints()
        logger.info('Graphiti client initialized successfully')

        if llm_client:
            logger.info(f'Using OpenAI model: {config.llm.model}')
            logger.info(f'Using temperature: {config.llm.temperature}')
        else:
            logger.info('No LLM client configured - entity extraction will be limited')

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
    result = ensure_no_embeddings(result)
    
    return result


# Dictionary to store queues for each group_id
episode_queues: dict[str, asyncio.Queue] = {}
queue_workers: dict[str, bool] = {}


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


async def initialize_server() -> MCPConfig:
    """Parse CLI arguments and initialize the Graphiti server configuration."""
    global config

    parser = argparse.ArgumentParser(
        description='Run the Graphiti MCP server with OpenAI API (no Azure support)'
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
        help='Temperature setting for the LLM (0.0-2.0). (default: 0.0)',
    )
    parser.add_argument('--destroy-graph', action='store_true', help='Destroy all Graphiti graphs')
    parser.add_argument(
        '--use-custom-entities',
        action='store_true',
        help='Enable entity extraction using the predefined ENTITY_TYPES',
    )
    parser.add_argument(
        '--host',
        default=os.environ.get('MCP_SERVER_HOST'),
        help='Host to bind the MCP server to (default: MCP_SERVER_HOST environment variable)',
    )

    args = parser.parse_args()

    config = GraphitiConfig.from_cli_and_env(args)

    if args.group_id:
        logger.info(f'Using provided group_id: {config.group_id}')
    else:
        logger.info(f'Using default group_id: {config.group_id}')

    if config.use_custom_entities:
        logger.info('Entity extraction enabled using predefined ENTITY_TYPES')
    else:
        logger.info('Entity extraction disabled (no custom entities will be used)')

    await initialize_graphiti()

    if args.host:
        logger.info(f'Setting MCP server host to: {args.host}')
        mcp.settings.host = args.host

    return MCPConfig.from_cli(args)


async def run_mcp_server():
    """Run the MCP server in the current event loop."""
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
    """Main function to run the Graphiti MCP server."""
    try:
        asyncio.run(run_mcp_server())
    except Exception as e:
        logger.error(f'Error initializing Graphiti MCP server: {str(e)}')
        raise


if __name__ == '__main__':
    main() 