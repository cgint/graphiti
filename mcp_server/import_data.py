#!/usr/bin/env python3
"""
Import script to reload exported Graphiti data back into the system.
"""

import asyncio
import json
import os
from datetime import datetime, timezone

async def import_memories(export_file_path):
    """Import memories from exported JSON file."""
    
    # Import inside the function to avoid import errors
    from graphiti_core import Graphiti
    from vertex_ai_client import VertexAIClient
    from ollama_embedder import NonBatchingOllamaEmbedder
    from ollama_reranker_client import OllamaRerankerClient
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.embedder.openai import OpenAIEmbedderConfig
    from graphiti_core.nodes import EpisodeType

    print("🔄 Loading exported data...")
    
    # Load the exported data
    with open(export_file_path, 'r', encoding='utf-8') as f:
        export_data = json.load(f)
    
    print(f"📊 Found: {export_data['export_metadata']['statistics']}")

    print("🚀 Initializing Graphiti client...")
    
    # Initialize Graphiti client (same as your MCP server)
    # Google Cloud credentials are handled by environment variables
    # Docker: GOOGLE_APPLICATION_CREDENTIALS=/app/adc.json (set in docker-compose.yml)
    # Local: GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/application_default_credentials.json (set in shell)
    if 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
        print(f"🔐 Using Google Cloud credentials: {os.environ['GOOGLE_APPLICATION_CREDENTIALS']}")
    else:
        print("⚠️  GOOGLE_APPLICATION_CREDENTIALS not set")
    
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

    print("📥 Starting data import...")

    episodes = export_data.get('episodes', [])
    imported_count = 0
    error_count = 0
    
    for i, episode in enumerate(episodes, 1):
        try:
            print(f"📝 Importing episode {i}/{len(episodes)}: {episode['name'][:50]}...")
            
            # Convert source string back to EpisodeType
            source_type = EpisodeType.text
            if episode.get('source') == 'message':
                source_type = EpisodeType.message
            elif episode.get('source') == 'json':
                source_type = EpisodeType.json

            # Parse timestamps if they exist - handle various precision levels
            valid_at = datetime.now(timezone.utc)
            if episode.get('valid_at'):
                timestamp_str = episode['valid_at']
                # Handle different timestamp formats and precision levels
                try:
                    # First try direct parsing
                    valid_at = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                except ValueError:
                    # If that fails, try truncating microseconds to 6 digits max
                    if '.' in timestamp_str and '+' in timestamp_str:
                        parts = timestamp_str.split('.')
                        if len(parts) == 2:
                            microsecond_part = parts[1].split('+')[0]
                            # Truncate to 6 digits (standard microsecond precision)
                            truncated_microseconds = microsecond_part[:6]
                            timezone_part = '+' + parts[1].split('+')[1]
                            fixed_timestamp = f"{parts[0]}.{truncated_microseconds}{timezone_part}"
                            valid_at = datetime.fromisoformat(fixed_timestamp)
                    else:
                        # Fallback to current time if parsing fails
                        print(f"⚠️  Could not parse timestamp {timestamp_str}, using current time")
                        valid_at = datetime.now(timezone.utc)

            # Add the episode without custom entity types to avoid Pydantic model issues
            # Let Graphiti generate new UUIDs since we cleared the graph
            await client.add_episode(
                name=episode['name'],
                episode_body=episode['content'],
                source=source_type,
                source_description=episode.get('source_description', ''),
                group_id=episode.get('group_id', ''),
                reference_time=valid_at,
                entity_types={}  # Use empty dict to avoid model field errors
            )
            
            imported_count += 1
            
        except Exception as e:
            print(f"❌ Error importing episode {episode['uuid']}: {str(e)}")
            error_count += 1
            # Continue with next episode
            continue

    print(f"\n✅ Import completed!")
    print(f"📊 Results:")
    print(f"   - Successfully imported: {imported_count} episodes")
    print(f"   - Errors: {error_count} episodes")
    print(f"   - Total processed: {len(episodes)} episodes")
    
    await client.driver.close()
    return imported_count, error_count

if __name__ == "__main__":
    import sys
    
    export_file = "/app/graphiti_export_20250623_134554.json"
    if len(sys.argv) > 1:
        export_file = sys.argv[1]
    
    imported, errors = asyncio.run(import_memories(export_file))
    
    if errors == 0:
        print(f"\n🎉 Perfect! All {imported} episodes imported successfully!")
    else:
        print(f"\n⚠️  Import completed with {errors} errors out of {imported + errors} total episodes") 