This is a guided walkthrough of the quickstart example. This file does not need to be executed.

The script `examples/quickstart/quickstart_neo4j.py` provides a complete demonstration of Graphiti's core features. Here's a breakdown of what it does:

### 1. Configuration and Initialization
First, the script sets up logging and loads environment variables for the Neo4j connection (`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`). It then initializes the `Graphiti` object, which is the main entry point for interacting with the library.

```python
# ... existing code ...
# Initialize Graphiti with Neo4j connection
graphiti = Graphiti(neo4j_uri, neo4j_user, neo4j_password)

try:
    # Initialize the graph database with graphiti's indices. This only needs to be done once.
    await graphiti.build_indices_and_constraints()
# ... existing code ...
```

### 2. Ingesting Data (Adding Episodes)
The script defines a list of "episodes". An episode is a piece of information, which can be a snippet of text or structured JSON. These episodes are then added to the knowledge graph. Graphiti automatically processes this content to extract entities (nodes) and their relationships (edges).

```python
# ... existing code ...
# Example: Add Episodes
# Episodes list containing both text and JSON episodes
episodes = [
    {
        'content': 'Kamala Harris is the Attorney General of California. She was previously '
        'the district attorney for San Francisco.',
        'type': EpisodeType.text,
        'description': 'podcast transcript',
    },
    # ... more episodes
]

# Add episodes to the graph
for i, episode in enumerate(episodes):
    await graphiti.add_episode(
        name=f'Freakonomics Radio {i}',
        episode_body=episode['content']
        if isinstance(episode['content'], str)
        else json.dumps(episode['content']),
        source=episode['type'],
        source_description=episode['description'],
        reference_time=datetime.now(timezone.utc),
    )
    print(f'Added episode: Freakonomics Radio {i} ({episode["type"].value})')
# ... existing code ...
```

### 3. Querying the Graph (Searching for Edges)
After ingesting data, you can search the graph. The script demonstrates a simple yet powerful hybrid search that combines semantic (vector) search with traditional keyword (BM25) search to find relevant relationships (edges) in the graph.

```python
# ... existing code ...
# Perform a hybrid search combining semantic similarity and BM25 retrieval
print("\nSearching for: 'Who was the California Attorney General?'")
results = await graphiti.search('Who was the California Attorney General?')

# Print search results
print('\nSearch Results:')
for result in results:
    print(f'UUID: {result.uuid}')
    print(f'Fact: {result.fact}')
    # ...
    print('---')
# ... existing code ...
```

### 4. Advanced Querying (Node and Reranked Search)
The script also shows two more advanced search methods:
-   **Reranking Search**: It takes the top result from the previous search and uses it as a "center node" to rerank other results based on their proximity in the graph. This helps find more contextually relevant information.
-   **Node Search**: It uses a predefined "search recipe" (`NODE_HYBRID_SEARCH_RRF`) to search for entities (nodes) directly, rather than relationships.

```python
# ... existing code ...
# Use the top search result's UUID as the center node for reranking
if results and len(results) > 0:
    # ...
    reranked_results = await graphiti.search(
        'Who was the California Attorney General?', center_node_uuid=center_node_uuid
    )
    # ...
# ... existing code ...
# Example: Perform a node search using _search method with standard recipes
# ...
node_search_results = await graphiti._search(
    query='California Governor',
    config=node_search_config,
)
# ...
```

## How to Run the Example

To see this in action, you can run the script yourself.

### 1. Prerequisites
- Make sure you have Python 3.9+ installed.
- You need a running Neo4j instance. The easiest way is with [Neo4j Desktop](https://neo4j.com/download/).
- You need an OpenAI API key.

### 2. Setup
First, install the necessary Python packages:
```bash
pip install -r examples/quickstart/requirements.txt
```
Then, set the required environment variables in your terminal. You can also create a `.env` file in the root of the project.

```bash
export OPENAI_API_KEY='your_openai_api_key'

# These are the default values, change them if your Neo4j setup is different
export NEO4J_URI='bolt://localhost:7687'
export NEO4J_USER='neo4j'
export NEO4J_PASSWORD='password'
```

### 3. Execute the script
Finally, run the script from the root of the repository:

```bash
python examples/quickstart/quickstart_neo4j.py
```

You will see output in your console showing the episodes being added and the results of the different search queries.

I can also run these commands for you if you'd like. 