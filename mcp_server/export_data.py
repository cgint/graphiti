#!/usr/bin/env python3
"""
Standalone export script to export Graphiti data to JSON.
This script can be run inside the Docker container.
"""

import asyncio
import json
import os
from datetime import datetime, timezone

# We'll import these only when running inside the container
async def export_memories():
    """Export all memories to JSON file."""
    
    # Import inside the function to avoid import errors
    from graphiti_core import Graphiti
    from vertex_ai_client import VertexAIClient
    from ollama_embedder import NonBatchingOllamaEmbedder
    from ollama_reranker_client import OllamaRerankerClient
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.embedder.openai import OpenAIEmbedderConfig

    print("Initializing Graphiti client...")
    
    # Initialize Graphiti client (same as your MCP server)
    llm_client = VertexAIClient(
        config=LLMConfig(
            model=os.environ.get('MODEL_NAME', 'gemini-2.5-flash-lite-preview-06-17'),
            temperature=0.3,
        ),
        project_id=os.environ.get('GOOGLE_CLOUD_PROJECT'),
        location=os.environ.get('GOOGLE_CLOUD_LOCATION', 'global'),
    )

    embedder = NonBatchingOllamaEmbedder(
        config=OpenAIEmbedderConfig(
            api_key="ollama",
            base_url=os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434'),
            embedding_model=os.environ.get('EMBEDDER_MODEL_NAME', 'nomic-embed-text')
        )
    )

    cross_encoder = OllamaRerankerClient(
        base_url=os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434'),
        model_name="bge-m3"
    )

    client = Graphiti(
        os.environ.get('NEO4J_URI', 'bolt://neo4j:7687'),
        os.environ.get('NEO4J_USER', 'neo4j'),
        os.environ.get('NEO4J_PASSWORD', 'demodemo'),
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder
    )

    print("Starting memory export...")

    # Get all group_ids
    print("Getting group IDs...")
    records, _, _ = await client.driver.execute_query(
        """
        MATCH (n) WHERE n.group_id IS NOT NULL
        RETURN DISTINCT n.group_id AS group_id
        """,
        database_="neo4j"
    )
    group_ids = [record['group_id'] for record in records]
    print(f"Found {len(group_ids)} group IDs: {group_ids}")

    export_data = {
        "export_metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "format_version": "1.0",
            "group_ids": group_ids,
            "total_groups": len(group_ids)
        },
        "episodes": [],
        "entities": [],
        "relationships": []
    }

    # Export Episodes
    print("Exporting episodes...")
    episode_query = """
        MATCH (e:Episodic)
        RETURN e.uuid AS uuid, e.name AS name, e.content AS content,
               e.source AS source, e.source_description AS source_description,
               e.group_id AS group_id, e.created_at AS created_at,
               e.valid_at AS valid_at, e.entity_edges AS entity_edges
        ORDER BY e.created_at ASC
    """
    
    episode_records, _, _ = await client.driver.execute_query(
        episode_query,
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

    print(f"Exported {len(export_data['episodes'])} episodes")

    # Export Entities
    print("Exporting entities...")
    entity_query = """
        MATCH (n:Entity)
        RETURN n.uuid AS uuid, n.name AS name, n.summary AS summary,
               n.group_id AS group_id, n.created_at AS created_at,
               labels(n) AS labels, properties(n) AS properties
        ORDER BY n.created_at ASC
    """
    
    entity_records, _, _ = await client.driver.execute_query(
        entity_query,
        database_="neo4j"
    )

    for record in entity_records:
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

    print(f"Exported {len(export_data['entities'])} entities")

    # Export Relationships
    print("Exporting relationships...")
    relationship_query = """
        MATCH (source:Entity)-[r:RELATES_TO]->(target:Entity)
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
        database_="neo4j"
    )

    for record in relationship_records:
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

    print(f"Exported {len(export_data['relationships'])} relationships")

    # Add summary statistics
    export_data["export_metadata"]["statistics"] = {
        "total_episodes": len(export_data["episodes"]),
        "total_entities": len(export_data["entities"]),
        "total_relationships": len(export_data["relationships"])
    }

    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"graphiti_export_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"Export completed: {export_data['export_metadata']['statistics']}")
    print(f"Data saved to: {filename}")
    
    await client.driver.close()
    return filename, export_data

if __name__ == "__main__":
    filename, data = asyncio.run(export_memories())
    print(f"\n✅ Export successful!")
    print(f"📁 File: {filename}")
    print(f"📊 Statistics: {data['export_metadata']['statistics']}") 