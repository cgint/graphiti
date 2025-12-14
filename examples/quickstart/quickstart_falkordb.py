"""
Copyright 2025, Zep Software, Inc.

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

#################################################
# EPISODE NAMING BEST PRACTICES
#################################################
# The 'name' parameter in add_episode() is a human-readable identifier
# for provenance tracking and debugging. It should answer:
# "Where did this fact come from?"
#
# General Principles:
#   - Unique & Identifiable: Distinguish between episodes in the same batch
#   - Concise but Descriptive: Readable in logs (aim for <60 chars)
#   - Tied to Source: Makes provenance tracking meaningful
#
# Recommendations by Use Case:
#   - Documents (PDF, articles): Use document title
#       e.g., 'California AG Report 2011'
#   - Documents with sections: Title + section/page
#       e.g., 'Budget Report - Chapter 3'
#   - Chat messages: Actor + timestamp or context
#       e.g., 'User message - 2024-01-15 14:30'
#   - JSON/structured data: Key identifying field(s)
#       e.g., 'Gavin Newsom - Governor Profile'
#   - Meeting transcripts: Meeting name + date
#       e.g., 'Q4 Planning - 2024-12-01'
#   - Podcast/video: Episode title + segment
#       e.g., 'Freakonomics Ep.123 - Harris Interview'
#
# For chunked documents, use: 'Document Title - Chunk N' or
# 'Document Title - Page X-Y'
#################################################

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from logging import INFO

from dotenv import load_dotenv

from graphiti_core import Graphiti
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF
from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

#################################################
# CONFIGURATION
#################################################
# Set up logging and environment variables for
# connecting to FalkorDB database
#################################################

# Configure logging
logging.basicConfig(
    level=INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

load_dotenv()

# FalkorDB connection parameters
# Make sure FalkorDB (on-premises) is running — see https://docs.falkordb.com/
# By default, FalkorDB does not require a username or password,
# but you can set them via environment variables for added security.
#
# If you're using FalkorDB Cloud, set the environment variables accordingly.
# For on-premises use, you can leave them as None or set them to your preferred values.
#
# The default host and port are 'localhost' and '6379', respectively.
# You can override these values in your environment variables or directly in the code.

falkor_username = os.environ.get('FALKORDB_USERNAME', None) or None
falkor_password = os.environ.get('FALKORDB_PASSWORD', None) or None
falkor_host = os.environ.get('FALKORDB_HOST', 'localhost') or 'localhost'
falkor_port = os.environ.get('FALKORDB_PORT', '6379') or '6379'

# Gemini API key configuration
gemini_api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')


async def export_graph_data(graphiti: Graphiti):
    """Exports all nodes and relationships to JSONL files."""
    print("\nStarting graph data export to JSONL files...")

    # Collect all nodes
    all_nodes = []

    # Export Entity Nodes using raw query
    entity_records, _, _ = await graphiti.driver.execute_query(
        """
        MATCH (n:Entity)
        RETURN n.uuid AS uuid, n.name AS name, n.group_id AS group_id, 
               n.summary AS summary, n.created_at AS created_at, labels(n) AS labels
        """,
        routing_='r',
    )
    
    for record in entity_records:
        all_nodes.append({
            'type': 'Entity',
            'uuid': record['uuid'],
            'name': record['name'],
            'group_id': record['group_id'],
            'summary': record['summary'],
            'created_at': str(record['created_at']) if record['created_at'] else None,
            'labels': record['labels'],
        })
    print(f"Found {len(entity_records)} entity nodes")

    # Export Episodic Nodes (source episodes) using raw query
    episodic_records, _, _ = await graphiti.driver.execute_query(
        """
        MATCH (e:Episodic)
        RETURN e.uuid AS uuid, e.name AS name, e.group_id AS group_id,
               e.content AS content, e.source AS source, 
               e.source_description AS source_description,
               e.created_at AS created_at, e.valid_at AS valid_at,
               e.entity_edges AS entity_edges
        """,
        routing_='r',
    )
    
    for record in episodic_records:
        all_nodes.append({
            'type': 'Episodic',
            'uuid': record['uuid'],
            'name': record['name'],
            'group_id': record['group_id'],
            'content': record['content'],
            'source': record['source'],
            'source_description': record['source_description'],
            'created_at': str(record['created_at']) if record['created_at'] else None,
            'valid_at': str(record['valid_at']) if record['valid_at'] else None,
            'entity_edges': record['entity_edges'],
        })
    print(f"Found {len(episodic_records)} episodic nodes")

    # Export Community Nodes (if any exist)
    community_records, _, _ = await graphiti.driver.execute_query(
        """
        MATCH (c:Community)
        RETURN c.uuid AS uuid, c.name AS name, c.group_id AS group_id,
               c.summary AS summary, c.created_at AS created_at
        """,
        routing_='r',
    )
    
    for record in community_records:
        all_nodes.append({
            'type': 'Community',
            'uuid': record['uuid'],
            'name': record['name'],
            'group_id': record['group_id'],
            'summary': record['summary'],
            'created_at': str(record['created_at']) if record['created_at'] else None,
        })
    print(f"Found {len(community_records)} community nodes")

    # Write all nodes to single file
    with open("exported_nodes.jsonl", "w", encoding="utf-8") as f:
        for node in all_nodes:
            f.write(json.dumps(node, ensure_ascii=False) + "\n")
    print(f"Exported {len(all_nodes)} total nodes to exported_nodes.jsonl")

    # Collect all edges
    all_edges = []

    # Export Entity Edges (RELATES_TO relationships) using raw query
    relates_to_records, _, _ = await graphiti.driver.execute_query(
        """
        MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)
        RETURN e.uuid AS uuid, e.name AS name, e.fact AS fact, e.group_id AS group_id,
               e.episodes AS episodes, e.created_at AS created_at, 
               e.valid_at AS valid_at, e.invalid_at AS invalid_at,
               n.uuid AS source_node_uuid, m.uuid AS target_node_uuid
        """,
        routing_='r',
    )
    
    for record in relates_to_records:
        all_edges.append({
            'type': 'RELATES_TO',
            'uuid': record['uuid'],
            'name': record['name'],
            'fact': record['fact'],
            'group_id': record['group_id'],
            'episodes': record['episodes'],
            'created_at': str(record['created_at']) if record['created_at'] else None,
            'valid_at': str(record['valid_at']) if record['valid_at'] else None,
            'invalid_at': str(record['invalid_at']) if record['invalid_at'] else None,
            'source_node_uuid': record['source_node_uuid'],
            'target_node_uuid': record['target_node_uuid'],
        })
    print(f"Found {len(relates_to_records)} RELATES_TO edges")

    # Export MENTIONS edges (Episodic -> Entity) using raw query
    mentions_records, _, _ = await graphiti.driver.execute_query(
        """
        MATCH (ep:Episodic)-[m:MENTIONS]->(en:Entity)
        RETURN m.uuid AS uuid, m.group_id AS group_id, m.created_at AS created_at,
               ep.uuid AS source_node_uuid, en.uuid AS target_node_uuid
        """,
        routing_='r',
    )
    
    for record in mentions_records:
        all_edges.append({
            'type': 'MENTIONS',
            'uuid': record['uuid'],
            'group_id': record['group_id'],
            'created_at': str(record['created_at']) if record['created_at'] else None,
            'source_node_uuid': record['source_node_uuid'],
            'target_node_uuid': record['target_node_uuid'],
        })
    print(f"Found {len(mentions_records)} MENTIONS edges")

    # Export HAS_MEMBER edges (Community -> Entity) if any exist
    has_member_records, _, _ = await graphiti.driver.execute_query(
        """
        MATCH (c:Community)-[h:HAS_MEMBER]->(e:Entity)
        RETURN h.uuid AS uuid, h.group_id AS group_id, h.created_at AS created_at,
               c.uuid AS source_node_uuid, e.uuid AS target_node_uuid
        """,
        routing_='r',
    )
    
    for record in has_member_records:
        all_edges.append({
            'type': 'HAS_MEMBER',
            'uuid': record['uuid'],
            'group_id': record['group_id'],
            'created_at': str(record['created_at']) if record['created_at'] else None,
            'source_node_uuid': record['source_node_uuid'],
            'target_node_uuid': record['target_node_uuid'],
        })
    print(f"Found {len(has_member_records)} HAS_MEMBER edges")

    # Write all edges to single file
    with open("exported_edges.jsonl", "w", encoding="utf-8") as f:
        for edge in all_edges:
            f.write(json.dumps(edge, ensure_ascii=False) + "\n")
    print(f"Exported {len(all_edges)} total edges to exported_edges.jsonl")

    print("\nGraph data export completed.")
    print(f"Summary: {len(all_nodes)} nodes, {len(all_edges)} edges")

async def main(query_only: bool = False, clear: bool = False, export: bool = False):
    #################################################
    # INITIALIZATION
    #################################################
    # Connect to FalkorDB and set up Graphiti indices
    # This is required before using other Graphiti
    # functionality
    #################################################

    # Initialize Graphiti with FalkorDB connection and Gemini clients
    falkor_driver = FalkorDriver(
        host=falkor_host, port=falkor_port, username=falkor_username, password=falkor_password
    )
    
    # Initialize Graphiti with Gemini clients
    graphiti = Graphiti(
        graph_driver=falkor_driver,
        llm_client=GeminiClient(
            config=LLMConfig(
                api_key=gemini_api_key,
                model='gemini-2.5-flash'
            )
        ),
        embedder=GeminiEmbedder(
            config=GeminiEmbedderConfig(
                api_key=gemini_api_key,
                embedding_model='embedding-001'
            )
        ),
        cross_encoder=GeminiRerankerClient(
            config=LLMConfig(
                api_key=gemini_api_key,
                model='gemini-2.5-flash-lite-preview-06-17'
            )
        )
    )

    try:
        #################################################
        # CLEAR GRAPH (if requested)
        #################################################
        # Clear all data from the graph for a clean start
        #################################################
        

        #################################################
        # EXPORT GRAPH DATA (if requested)
        #################################################
        # Export all nodes and relationships to JSONL files
        #################################################
        if export:
            await export_graph_data(graphiti)
            return


        if clear:
            print('\nClearing graph data for clean install...')
            await clear_data(graphiti.driver)
            await graphiti.build_indices_and_constraints()
            print('Graph cleared and indices rebuilt successfully\n')


        #################################################
        # ADDING EPISODES
        #################################################
        # Episodes are the primary units of information
        # in Graphiti. They can be text or structured JSON
        # and are automatically processed to extract entities
        # and relationships.
        #################################################

        if not query_only:
            # Example: Add Episodes with Temporal Information
            # Episodes list containing both text and JSON episodes with meaningful timestamps
            # The reference_time represents when the information in the episode was valid/occurred
            episodes = [
                {
                    'content': 'Kamala Harris is the Attorney General of California. She was previously '
                    'the district attorney for San Francisco.',
                    'type': EpisodeType.text,
                    'description': 'podcast transcript',
                    'reference_time': datetime(2011, 1, 3, tzinfo=timezone.utc),
                },
                {
                    'content': 'As AG, Harris was in office from January 3, 2011 – January 3, 2017',
                    'type': EpisodeType.text,
                    'description': 'podcast transcript',
                    'reference_time': datetime(2019, 2, 5, tzinfo=timezone.utc)
                },
                {
                    'content': {
                        'name': 'Gavin Newsom',
                        'position': 'Governor',
                        'state': 'California',
                        'previous_role': 'Lieutenant Governor',
                        'previous_location': 'San Francisco',
                    },
                    'type': EpisodeType.json,
                    'description': 'podcast metadata',
                    'reference_time': datetime(2019, 1, 7, tzinfo=timezone.utc),
                },
                {
                    'content': {
                        'name': 'Gavin Newsom',
                        'position': 'Governor',
                        'term_start': 'January 7, 2019',
                        'term_end': 'Present',
                    },
                    'type': EpisodeType.json,
                    'description': 'podcast metadata',
                    'reference_time': datetime(2022, 3, 30, tzinfo=timezone.utc),
                },
            ]

            # Add episodes to the graph with temporal information
            episode_uuids = []
            for i, episode in enumerate(episodes):
                result = await graphiti.add_episode(
                    name=f'Freakonomics Radio {i}',
                    episode_body=episode['content']
                    if isinstance(episode['content'], str)
                    else json.dumps(episode['content']),
                    source=episode['type'],
                    source_description=episode['description'],
                    reference_time=episode['reference_time'],
                )
                episode_uuids.append(result.episode.uuid)
                print(
                    f'Added episode: Freakonomics Radio {i} ({episode["type"].value}) '
                    f'at {episode["reference_time"].strftime("%Y-%m-%d")}'
                )
        else:
            print('\nQuery-only mode: Skipping episode addition to preserve existing data')

        #################################################
        # BASIC SEARCH
        #################################################
        # The simplest way to retrieve relationships (edges)
        # from Graphiti is using the search method, which
        # performs a hybrid search combining semantic
        # similarity and BM25 text retrieval.
        #################################################

        # Perform a hybrid search combining semantic similarity and BM25 retrieval
        print("\nSearching for: 'Who was the California Attorney General?'")
        results = await graphiti.search('Who was the California Attorney General?')

        # Print search results with traceability and temporal information
        print('\nSearch Results:')
        for result in results:
            print(f'UUID: {result.uuid}')
            print(f'Fact: {result.fact}')
            
            # Temporal information
            if hasattr(result, 'valid_at') and result.valid_at:
                print(f'Valid from: {result.valid_at}')
            if hasattr(result, 'invalid_at') and result.invalid_at:
                print(f'Valid until: {result.invalid_at}')
            
            # Traceability: Show source episodes
            if hasattr(result, 'episodes') and result.episodes:
                print(f'Source Episodes: {len(result.episodes)} episode(s)')
                for episode_uuid in result.episodes:
                    try:
                        episode = await EpisodicNode.get_by_uuid(graphiti.driver, episode_uuid)
                        print(f'  - Episode: {episode.name}')
                        print(f'    Content preview: {episode.content[:80]}...')
                        print(f'    Valid at: {episode.valid_at}')
                        print(f'    Source: {episode.source_description}')
                    except Exception as e:
                        print(f'  - Episode UUID: {episode_uuid} (could not retrieve: {e})')
            else:
                print('Source Episodes: None')
            
            print('---')

        #################################################
        # CENTER NODE SEARCH
        #################################################
        # For more contextually relevant results, you can
        # use a center node to rerank search results based
        # on their graph distance to a specific node
        #################################################

        # Use the top search result's UUID as the center node for reranking
        if results and len(results) > 0:
            # Get the source node UUID from the top result
            center_node_uuid = results[0].source_node_uuid

            print('\nReranking search results based on graph distance:')
            print(f'Using center node UUID: {center_node_uuid}')

            reranked_results = await graphiti.search(
                'Who was the California Attorney General?', center_node_uuid=center_node_uuid
            )

            # Print reranked search results with traceability
            print('\nReranked Search Results:')
            for result in reranked_results:
                print(f'UUID: {result.uuid}')
                print(f'Fact: {result.fact}')
                
                # Temporal information
                if hasattr(result, 'valid_at') and result.valid_at:
                    print(f'Valid from: {result.valid_at}')
                if hasattr(result, 'invalid_at') and result.invalid_at:
                    print(f'Valid until: {result.invalid_at}')
                
                # Traceability: Show source episodes
                if hasattr(result, 'episodes') and result.episodes:
                    print(f'Source Episodes: {len(result.episodes)} episode(s)')
                    for episode_uuid in result.episodes[:2]:  # Show first 2 to avoid clutter
                        try:
                            episode = await EpisodicNode.get_by_uuid(graphiti.driver, episode_uuid)
                            print(f'  - Episode: {episode.name} (valid at: {episode.valid_at})')
                        except Exception:
                            print(f'  - Episode UUID: {episode_uuid}')
                    if len(result.episodes) > 2:
                        print(f'  ... and {len(result.episodes) - 2} more episode(s)')
                
                print('---')
        else:
            print('No results found in the initial search to use as center node.')

        #################################################
        # NODE SEARCH USING SEARCH RECIPES
        #################################################
        # Graphiti provides predefined search recipes
        # optimized for different search scenarios.
        # Here we use NODE_HYBRID_SEARCH_RRF for retrieving
        # nodes directly instead of edges.
        #################################################

        # Example: Perform a node search using _search method with standard recipes
        print(
            '\nPerforming node search using _search method with standard recipe NODE_HYBRID_SEARCH_RRF:'
        )

        # Use a predefined search configuration recipe and modify its limit
        node_search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        node_search_config.limit = 5  # Limit to 5 results

        # Execute the node search
        node_search_results = await graphiti._search(
            query='California Governor',
            config=node_search_config,
        )

        # Print node search results
        print('\nNode Search Results:')
        for node in node_search_results.nodes:
            print(f'Node UUID: {node.uuid}')
            print(f'Node Name: {node.name}')
            node_summary = node.summary[:100] + '...' if len(node.summary) > 100 else node.summary
            print(f'Content Summary: {node_summary}')
            print(f'Node Labels: {", ".join(node.labels)}')
            print(f'Created At: {node.created_at}')
            if hasattr(node, 'attributes') and node.attributes:
                print('Attributes:')
                for key, value in node.attributes.items():
                    print(f'  {key}: {value}')
            print('---')

    finally:
        #################################################
        # CLEANUP
        #################################################
        # Always close the connection to FalkorDB when
        # finished to properly release resources
        #################################################

        # Close the connection
        await graphiti.close()
        print('\nConnection closed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Graphiti FalkorDB Quickstart Example',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--query-only',
        action='store_true',
        help='Run queries only without adding new episodes (preserves existing data)',
    )
    parser.add_argument(
        '--clear',
        action='store_true',
        help='Clear all graph data before running (clean install)',
    )
    parser.add_argument(
        '--export',
        action='store_true',
        help='Export all graph nodes and relationships to JSONL files',
    )
    args = parser.parse_args()
    asyncio.run(main(query_only=args.query_only, clear=args.clear, export=args.export))
