#!/bin/bash

# Start FalkorDB only (no graph service needed)
docker-compose --profile falkordb up -d falkordb

sleep 5

echo
echo "Now we are ready to run the quickstart script"
echo
echo "Run: example: uv run python examples/quickstart/quickstart_falkordb.py"

# python quickstart_falkordb.py --query-only
# python quickstart_falkordb.py --export
# python quickstart_falkordb.py --clear