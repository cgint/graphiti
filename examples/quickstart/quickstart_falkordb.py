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
import glob
import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from logging import INFO  # noqa: F401

from dotenv import load_dotenv
from google.genai import types
from pydantic import BaseModel, Field

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.search import search_config_recipes
from graphiti_core.search.search_helpers import search_results_to_context_string
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
falkor_port = int(os.environ.get('FALKORDB_PORT', '6379') or '6379')

# Gemini API key configuration
gemini_api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')


#################################################
# HELPER FUNCTIONS
#################################################

def parse_wizard_of_oz_paragraphs() -> list[dict]:
    """Parse woo.txt into paragraph-based episodes for finer-grained knowledge extraction.
    
    This approach is better suited for:
    - General text documents (Confluence, wiki pages)
    - Business content (Jira descriptions, comments)
    - Any content where paragraphs represent logical semantic units
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    woo_path = os.path.join(script_dir, 'woo.txt')
    
    with open(woo_path, encoding='utf-8') as f:
        content = f.read()
    
    # Split content by chapter pattern to get chapter context
    chapter_pattern = r'^Chapter ([IVX]+)\n(.+?)$'
    chapter_matches = list(re.finditer(chapter_pattern, content, re.MULTILINE))
    
    paragraphs = []
    paragraph_index = 0
    
    for i, chapter_match in enumerate(chapter_matches):
        chapter_num = chapter_match.group(1)
        chapter_title = chapter_match.group(2).strip()
        
        # Find chapter content boundaries
        start_pos = chapter_match.end()
        end_pos = chapter_matches[i + 1].start() if i + 1 < len(chapter_matches) else len(content)
        chapter_content = content[start_pos:end_pos].strip()
        
        # Split chapter into paragraphs (separated by blank lines)
        raw_paragraphs = re.split(r'\n\s*\n', chapter_content)
        
        for para in raw_paragraphs:
            para_text = para.strip()
            # Skip empty paragraphs or very short ones (< 50 chars)
            if len(para_text) < 50:
                continue
            
            paragraphs.append({
                'chapter_num': chapter_num,
                'chapter_title': chapter_title,
                'paragraph_index': paragraph_index,
                'content': para_text,
            })
            paragraph_index += 1
    print(f"Parsed {len(paragraphs)} paragraphs from woo.txt")
    return paragraphs


def parse_wizard_of_oz_chapters() -> list[dict]:
    """Parse woo.txt into chapter-based episodes for coarser-grained knowledge extraction.
    
    This approach is better suited for:
    - Long-form narrative content where chapter context matters
    - Documents where you want fewer, larger episodes
    - Cases where cross-paragraph relationships within a chapter are important
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    woo_path = os.path.join(script_dir, 'woo.txt')
    
    with open(woo_path, encoding='utf-8') as f:
        content = f.read()
    
    # Split content by chapter pattern
    chapter_pattern = r'^Chapter ([IVX]+)\n(.+?)$'
    chapter_matches = list(re.finditer(chapter_pattern, content, re.MULTILINE))
    
    chapters = []
    
    for i, chapter_match in enumerate(chapter_matches):
        chapter_num = chapter_match.group(1)
        chapter_title = chapter_match.group(2).strip()
        
        # Find chapter content boundaries
        start_pos = chapter_match.end()
        end_pos = chapter_matches[i + 1].start() if i + 1 < len(chapter_matches) else len(content)
        chapter_content = content[start_pos:end_pos].strip()
        
        # Skip empty chapters
        if len(chapter_content) < 50:
            continue
        
        chapters.append({
            'chapter_num': chapter_num,
            'chapter_title': chapter_title,
            'chapter_index': i,
            'content': chapter_content,
        })
    print(f"Parsed {len(chapters)} chapters from woo.txt")
    return chapters


def parse_wizard_of_oz_episodes() -> list[dict]:
    """Parse woo.txt into chapter-based episodes for coarser-grained knowledge extraction."""
    episode_list = []
    for i, e in enumerate(parse_wizard_of_oz_chapters()):
        name = f'Wizard of Oz - Ch.{e["chapter_num"]} ({e["chapter_title"]})'
        if 'paragraphs' in e:
            name += f' - Para {e["paragraph_index"] + 1}'
        episode_list.append({
            'name': name,
            'content': e['content'],
            'type': EpisodeType.text,
            'description': 'book chapter',
            'reference_time': datetime(1900, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i),
        })
    print(f"Parsed {len(episode_list)} episodes from woo.txt")
    return episode_list

def parse_know_ai_files_to_episodes(dir_path: str, file_pattern: str) -> list[dict]:
    """Parse know-ai files to episodes, splitting by empty lines."""
    episode_list = []
    for file in glob.glob(os.path.join(dir_path, file_pattern)):
        with open(file, encoding='utf-8') as f:
            content = f.read()
        name = os.path.basename(file)
        # Split by one or more empty lines (double newlines)
        sections = re.split(r'\n\s*\n', content)
        for idx, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue
            # Use first line as section title if available
            first_line = section.split('\n', 1)[0][:80]
            episode_list.append({
                'name': f'{name}::{idx}::{first_line}',
                'content': section,
                'type': EpisodeType.text,
                'description': f'know-ai file: {name} - section {idx}: {first_line}',
                'reference_time': datetime.now(timezone.utc),
            })

    print(f"Parsed {len(episode_list)} episodes from {dir_path} with pattern {file_pattern}")
    return episode_list

#################################################
# ENTITY TYPE DEFINITIONS (Wizard of Oz)
#################################################
# Custom entity types guide the LLM to extract
# meaningful entities rather than every noun.
# The docstrings are critical - they tell the LLM
# what to extract and what to ignore.
#################################################

class Character(BaseModel):
    """A named character in the story - humans, talking animals, or personified beings.
    
    Only extract main characters and named individuals, NOT generic people or crowds.
    Examples: Dorothy, Toto, Scarecrow, Tin Woodman, Cowardly Lion, Wizard of Oz, Wicked Witch.
    Do NOT extract: "the man", "a farmer", "people", "someone".
    """
    role: str = Field(..., description="Character's role: protagonist, antagonist, helper, mentor, etc.")


class Place(BaseModel):
    """A significant named location or region in the story.
    
    Only extract named places, NOT generic locations like "house", "road", "field".
    Examples: Kansas, Land of Oz, Emerald City, Munchkin Country, Yellow Brick Road, Deadly Desert.
    Do NOT extract: "the house", "a forest", "the road", "outside".
    """
    region: str | None = Field(None, description="The larger region this place belongs to, if mentioned")


class MagicalObject(BaseModel):
    """An object with magical properties or major plot significance.
    
    Only extract items central to the plot, NOT everyday objects.
    Examples: Silver Shoes, Golden Cap, Magic Belt, Wizard's balloon.
    Do NOT extract: "basket", "door", "chair", "food".
    """
    powers: str = Field(..., description="What the object does or why it matters to the story")


class Event(BaseModel):
    """A significant story event or occurrence that changes the plot.
    
    Only extract major story events, NOT minor actions.
    Examples: The Cyclone, Dorothy landing on the Witch, Meeting the Scarecrow, Melting of the Wicked Witch.
    Do NOT extract: "walking", "eating", "sleeping", "talking".
    """
    outcome: str = Field(..., description="What resulted from this event")


class Quest(BaseModel):
    """A goal, desire, or mission that drives a character's actions.
    
    Examples: Dorothy's journey home, Scarecrow seeking a brain, Tin Woodman seeking a heart, Lion seeking courage.
    """
    seeker: str = Field(..., description="The character who wants to achieve this goal")


class Group(BaseModel):
    """A named group, faction, or species of beings.
    
    Examples: Munchkins, Winkies, Quadlings, Gillikins, Winged Monkeys, Kalidahs.
    Do NOT extract: "people", "creatures", "animals".
    """
    nature: str = Field(..., description="What kind of beings they are")


# Entity types for the Wizard of Oz scenario
WIZARD_OF_OZ_ENTITY_TYPES: dict[str, type[BaseModel]] = {
    'Character': Character,
    'Place': Place,
    'MagicalObject': MagicalObject,
    'Event': Event,
    'Quest': Quest,
    'Group': Group,
}


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
        'search_queries': ['Who was the California Attorney General?'],
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
        'search_queries': ['What is Sarah Chen\'s current role?'],
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
        'search_queries': ['Who is the primary contact at Acme Corp?', 'What is the satisfaction score for Acme Corp?'],
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
    'wizard_of_oz': {
        'name': 'The Wizard of Oz',
        'description': "Dorothy's journey through the Land of Oz (paragraph-level episodes)",
        'search_queries': [
            'Who is Dorothy?',
            'What is Toto?',
            'Who is the Scarecrow?',
            'What happened during the cyclone?',
            'Where is the Emerald City?',
        ],
        'entity_types': WIZARD_OF_OZ_ENTITY_TYPES,
        'excluded_entity_types': ['Entity'],  # Skip generic entities like "house", "sky"
        'episodes': parse_wizard_of_oz_episodes(),
    },
    'know_ai_file_starting_AI': {
        'name': 'Know-AI File Starting AI',
        'description': "Know-AI File Starting AI",
        'search_queries': [
            'What does Allen Holub write about?',
            'Tell me about Larger designs, implemented too far in advance.',
            'Tell me about Agile Certifications.'
        ],
        # 'entity_types': KNOW_AI_FILE_STARTING_AI_ENTITY_TYPES,
        # 'excluded_entity_types': ['Entity'],  # Skip generic entities like "house", "sky"
        'episodes': parse_know_ai_files_to_episodes("examples/quickstart/know-ai", "Allen_Holub_*.txt"),
    },
    'ssi_schaefer': {
        'name': 'SSI Schaefer',
        'description': "SSI Schaefer",
        'search_queries': [
            'What is SSI Schaefer?',
        ],
        'episodes': parse_know_ai_files_to_episodes("/Users/christian.gintenreiter/dev/Workspace-INTERNAL/split-data-agentspace/case-0-patient-zero-data/auto-eval/scenarios/patient-zero-schaefer/UseCase-0-SSI-Schaefer-addon-TXT", "*.txt"),
    }
}


async def export_graph_data(graphiti: Graphiti, scenario: str):
    """Exports all nodes and relationships to JSONL files for a specific scenario."""
    print(f"\nStarting graph data export for scenario '{scenario}' to JSONL files...")
    
    # Switch driver to the scenario's database
    graphiti.driver = graphiti.driver.clone(database=scenario)

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

    # Write all nodes to single file (named by scenario)
    nodes_filename = f"exported_nodes_{scenario}.jsonl"
    with open(nodes_filename, "w", encoding="utf-8") as f:
        for node in all_nodes:
            f.write(json.dumps(node, ensure_ascii=False) + "\n")
    print(f"Exported {len(all_nodes)} total nodes to {nodes_filename}")

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

    # Write all edges to single file (named by scenario)
    edges_filename = f"exported_edges_{scenario}.jsonl"
    with open(edges_filename, "w", encoding="utf-8") as f:
        for edge in all_edges:
            f.write(json.dumps(edge, ensure_ascii=False) + "\n")
    print(f"Exported {len(all_edges)} total edges to {edges_filename}")

    print(f"\nGraph data export for scenario '{scenario}' completed.")
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
                model='gemini-2.5-flash',
                max_tokens=65000
            ),
            thinking_config=types.ThinkingConfig(
                include_thoughts=False,
                thinking_budget=0
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
                model='gemini-2.5-flash-lite',
                max_tokens=65000
            )
        )
    )

    try:
        # For FalkorDB, each scenario uses a separate database.
        # Switch to the scenario's database FIRST, before checking/building indices
        # This ensures indices are created on the correct database where data lives.
        graphiti.use_database(scenario)

        #################################################
        # ENSURE INDICES EXIST
        #################################################
        # Check if indices exist and build only if missing
        # Skip for export-only mode since it doesn't need indices
        #################################################
        if not export and not clear and not await graphiti.driver.has_required_indices():
            print('Building indices (first-time setup)...')
            await graphiti.build_indices_and_constraints()

        #################################################
        # EXPORT GRAPH DATA (if requested)
        #################################################
        # Export all nodes and relationships to JSONL files
        #################################################
        if export:
            await export_graph_data(graphiti, scenario)
            return

        if clear:
            print(f'\nClearing graph data for scenario "{scenario}"...')
            await clear_data(graphiti.driver, group_ids=[scenario])
            await graphiti.build_indices_and_constraints()
            print(f'Graph cleared for scenario "{scenario}" and indices rebuilt successfully\n')

        print(f'Building indices for scenario "{scenario}"...')
        await graphiti.build_indices_and_constraints()

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
            
            # Get optional entity types configuration (for focused extraction)
            entity_types = selected_scenario.get('entity_types')
            excluded_entity_types = selected_scenario.get('excluded_entity_types')
            
            if entity_types:
                print(f'Using custom entity types: {list(entity_types.keys())}')
                if excluded_entity_types:
                    print(f'Excluding entity types: {excluded_entity_types}')

            # Add episodes to the graph with temporal information
            # Use scenario name as group_id to create separate FalkorDB database per scenario
            episode_uuids = []
            for episode in episodes:
                # Retry logic for transient network errors
                max_retries = 3
                retry_delay = 2  # seconds
                for attempt in range(max_retries):
                    try:
                        result = await graphiti.add_episode(
                            name=episode['name'],
                            episode_body=episode['content']
                            if isinstance(episode['content'], str)
                            else json.dumps(episode['content']),
                            source=episode['type'],
                            source_description=episode['description'],
                            reference_time=episode['reference_time'],
                            group_id=scenario,
                            entity_types=entity_types,
                            excluded_entity_types=excluded_entity_types,
                        )
                        episode_uuids.append(result.episode.uuid)
                        print(
                            f'Added episode: {episode["name"]} ({episode["type"].value}) '
                            f'at {episode["reference_time"].strftime("%Y-%m-%d")}'
                        )
                        break  # Success, exit retry loop
                    except Exception as e:
                        if attempt < max_retries - 1:
                            print(f'Attempt {attempt + 1} failed for episode "{episode["name"]}": {e}')
                            print(f'Retrying in {retry_delay} seconds...')
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                        else:
                            print(f'All {max_retries} attempts failed for episode "{episode["name"]}"')
                            raise
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
        # Use group_ids to scope search to the current scenario's database
        search_queries = selected_scenario['search_queries']
        for search_query in search_queries:
            print(f"\nSearching for: '{search_query}'")
            results = await graphiti.search(search_query, group_ids=[scenario])
            results_search = await graphiti.search_(search_query, group_ids=[scenario])

            # Print search results with traceability and temporal information
            print('\nSearch Results as text:')
            pretty_results = search_results_to_context_string(results_search)
            print(pretty_results)
            print('\n\nSearch Results:')
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
                    search_query, center_node_uuid=center_node_uuid, group_ids=[scenario]
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
        node_search_config = search_config_recipes.NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        node_search_config.limit = 5  # Limit to 5 results

        # Execute the node search
        node_search_results = await graphiti._search(
            query='California Governor',
            config=node_search_config,
            group_ids=[scenario],
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
        epilog='''
Available Scenarios:
  politics  - California Politics: Political careers and government positions
  employee  - Employee Career Journey: Career progression at TechCorp
  customer  - B2B Customer Relationship: Acme Corp customer journey
  wizard_of_oz - The Wizard of Oz: Dorothy's journey through the Land of Oz

Examples:
  python quickstart_falkordb.py --scenario employee --clear
  python quickstart_falkordb.py --scenario customer --query-only
  python quickstart_falkordb.py --scenario politics
  python quickstart_falkordb.py --scenario wizard_of_oz --clear
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
