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


#################################################
# SCENARIO DEFINITIONS
#################################################
# Three example scenarios demonstrating temporal
# knowledge graph capabilities with different
# use cases and natural time-based evolution.
#################################################

SCENARIOS = {
    'politics': {
        'name': 'California Politics',
        'description': 'Political careers and government positions in California',
        'search_query': 'Who was the California Attorney General?',
        'episodes': [
            {
                'name': 'Harris - CA Attorney General Background',
                'content': 'Kamala Harris is the Attorney General of California. She was previously '
                           'the district attorney for San Francisco.',
                'type': EpisodeType.text,
                'description': 'podcast transcript',
                'reference_time': datetime(2011, 1, 3, tzinfo=timezone.utc),
            },
            {
                'name': 'Harris - AG Term Dates',
                'content': 'As AG, Harris was in office from January 3, 2011 – January 3, 2017',
                'type': EpisodeType.text,
                'description': 'podcast transcript',
                'reference_time': datetime(2019, 2, 5, tzinfo=timezone.utc),
            },
            {
                'name': 'Newsom - Governor Profile',
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
                'name': 'Newsom - Governor Term Info',
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
        ],
    },
    'employee': {
        'name': 'Employee Career Journey',
        'description': 'Career progression of Sarah Chen at TechCorp - promotions, team changes, projects',
        'search_query': 'What is Sarah Chen\'s current role?',
        'episodes': [
            {
                'name': 'Sarah Chen - Hiring',
                'content': 'Sarah Chen joined TechCorp as a Software Engineer on the Platform team. '
                           'Her manager is Michael Torres. She has a background in distributed systems.',
                'type': EpisodeType.text,
                'description': 'HR onboarding record',
                'reference_time': datetime(2020, 3, 15, tzinfo=timezone.utc),
            },
            {
                'name': 'Sarah Chen - Project Success',
                'content': 'Sarah Chen led the migration of the payment system to microservices. '
                           'The project reduced latency by 40% and was completed ahead of schedule. '
                           'Team members included Jake Miller and Priya Sharma.',
                'type': EpisodeType.text,
                'description': 'project review',
                'reference_time': datetime(2021, 6, 1, tzinfo=timezone.utc),
            },
            {
                'name': 'Sarah Chen - Promotion to Senior',
                'content': {
                    'employee': 'Sarah Chen',
                    'previous_title': 'Software Engineer',
                    'new_title': 'Senior Software Engineer',
                    'effective_date': 'July 1, 2021',
                    'manager': 'Michael Torres',
                    'team': 'Platform',
                },
                'type': EpisodeType.json,
                'description': 'HR promotion record',
                'reference_time': datetime(2021, 7, 1, tzinfo=timezone.utc),
            },
            {
                'name': 'Sarah Chen - Team Transfer',
                'content': 'Sarah Chen is transferring from Platform team to the AI/ML team. '
                           'Her new manager is Lisa Park. The transfer is effective April 1, 2022. '
                           'She will focus on recommendation systems.',
                'type': EpisodeType.text,
                'description': 'internal transfer notice',
                'reference_time': datetime(2022, 4, 1, tzinfo=timezone.utc),
            },
            {
                'name': 'Sarah Chen - Tech Lead Appointment',
                'content': {
                    'employee': 'Sarah Chen',
                    'previous_title': 'Senior Software Engineer',
                    'new_title': 'Tech Lead',
                    'team': 'AI/ML',
                    'manager': 'Lisa Park',
                    'direct_reports': ['Alex Kim', 'Jordan Lee', 'Casey Brown'],
                    'effective_date': 'January 15, 2023',
                },
                'type': EpisodeType.json,
                'description': 'org announcement',
                'reference_time': datetime(2023, 1, 15, tzinfo=timezone.utc),
            },
            {
                'name': 'AI/ML Team - Product Launch',
                'content': 'The AI/ML team under Sarah Chen\'s leadership shipped the new '
                           'recommendation engine to production. The system serves 10M requests/day. '
                           'Key contributors: Alex Kim, Jordan Lee, and Casey Brown.',
                'type': EpisodeType.text,
                'description': 'product launch announcement',
                'reference_time': datetime(2023, 8, 1, tzinfo=timezone.utc),
            },
        ],
    },
    'customer': {
        'name': 'B2B Customer Relationship',
        'description': 'Acme Corp customer journey - contracts, expansions, contact changes',
        'search_query': 'Who is the primary contact at Acme Corp?',
        'episodes': [
            {
                'name': 'Acme Corp - Initial Contract',
                'content': 'Acme Corp signed a 1-year contract for 50 seats of our Enterprise platform. '
                           'Primary contact is Bob Wilson, IT Director. Deal value: $25,000/year. '
                           'Acme Corp is a manufacturing company based in Chicago.',
                'type': EpisodeType.text,
                'description': 'sales CRM note',
                'reference_time': datetime(2021, 2, 1, tzinfo=timezone.utc),
            },
            {
                'name': 'Acme Corp - Pilot Success',
                'content': {
                    'customer': 'Acme Corp',
                    'pilot_department': 'Engineering',
                    'pilot_users': 50,
                    'satisfaction_score': 4.5,
                    'key_feedback': 'Integration with existing tools was seamless',
                    'contact': 'Bob Wilson',
                },
                'type': EpisodeType.json,
                'description': 'pilot review',
                'reference_time': datetime(2021, 5, 15, tzinfo=timezone.utc),
            },
            {
                'name': 'Acme Corp - Expansion',
                'content': 'Acme Corp expanded their contract from 50 to 150 seats '
                           'after successful pilot in engineering department. New departments: '
                           'Operations and Finance. Deal value increased to $75,000/year.',
                'type': EpisodeType.text,
                'description': 'account update',
                'reference_time': datetime(2021, 8, 15, tzinfo=timezone.utc),
            },
            {
                'name': 'Acme Corp - Contact Change',
                'content': 'Bob Wilson has left Acme Corp to join another company. '
                           'New primary contact is Maria Garcia, VP of Engineering. '
                           'Maria is enthusiastic about expanding usage to more teams.',
                'type': EpisodeType.text,
                'description': 'CRM update',
                'reference_time': datetime(2022, 3, 1, tzinfo=timezone.utc),
            },
            {
                'name': 'Acme Corp - Support Escalation',
                'content': {
                    'customer': 'Acme Corp',
                    'issue': 'Performance degradation during peak hours',
                    'severity': 'High',
                    'contact': 'Maria Garcia',
                    'resolution': 'Upgraded to dedicated infrastructure',
                    'resolution_time': '24 hours',
                },
                'type': EpisodeType.json,
                'description': 'support ticket',
                'reference_time': datetime(2022, 7, 10, tzinfo=timezone.utc),
            },
            {
                'name': 'Acme Corp - Enterprise Upgrade',
                'content': 'Acme Corp upgraded to Enterprise tier with SSO, dedicated support, '
                           'and 99.9% SLA. Contract extended to 3 years. 250 seats total. '
                           'New deal value: $150,000/year. Signed by Maria Garcia.',
                'type': EpisodeType.text,
                'description': 'contract amendment',
                'reference_time': datetime(2022, 11, 1, tzinfo=timezone.utc),
            },
        ],
    },
}


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

async def main(query_only: bool = False, clear: bool = False, export: bool = False, scenario: str = 'politics'):
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

        # Load the selected scenario
        selected_scenario = SCENARIOS[scenario]
        print(f'\n=== Scenario: {selected_scenario["name"]} ===')
        print(f'Description: {selected_scenario["description"]}\n')

        if not query_only:
            # Get episodes from the selected scenario
            episodes = selected_scenario['episodes']

            # Add episodes to the graph with temporal information
            episode_uuids = []
            for episode in episodes:
                result = await graphiti.add_episode(
                    name=episode['name'],
                    episode_body=episode['content']
                    if isinstance(episode['content'], str)
                    else json.dumps(episode['content']),
                    source=episode['type'],
                    source_description=episode['description'],
                    reference_time=episode['reference_time'],
                )
                episode_uuids.append(result.episode.uuid)
                print(
                    f'Added episode: {episode["name"]} ({episode["type"].value}) '
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
        search_query = selected_scenario['search_query']
        print(f"\nSearching for: '{search_query}'")
        results = await graphiti.search(search_query)

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
                search_query, center_node_uuid=center_node_uuid
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
    # Build scenario choices description
    scenario_help = 'Scenario to run:\n'
    for key, val in SCENARIOS.items():
        scenario_help += f'  {key}: {val["name"]} - {val["description"]}\n'

    parser = argparse.ArgumentParser(
        description='Graphiti FalkorDB Quickstart Example',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f'''
Available Scenarios:
  politics  - California Politics: Political careers and government positions
  employee  - Employee Career Journey: Career progression at TechCorp
  customer  - B2B Customer Relationship: Acme Corp customer journey

Examples:
  python quickstart_falkordb.py --scenario employee --clear
  python quickstart_falkordb.py --scenario customer --query-only
  python quickstart_falkordb.py --scenario politics
''',
    )
    parser.add_argument(
        '--scenario',
        choices=list(SCENARIOS.keys()),
        default='politics',
        help='Scenario to run (default: politics)',
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
    asyncio.run(main(
        query_only=args.query_only,
        clear=args.clear,
        export=args.export,
        scenario=args.scenario,
    ))
