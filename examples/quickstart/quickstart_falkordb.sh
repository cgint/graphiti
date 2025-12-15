#!/bin/bash

# Start FalkorDB only (no graph service needed)
docker-compose --profile falkordb up -d falkordb

sleep 5

echo
echo "Now we are ready to run the quickstart script"
echo
echo "Run (default politics scenario):"
echo "  uv run python examples/quickstart/quickstart_falkordb.py"
echo
echo "Other scenario examples:"
echo "  uv run python examples/quickstart/quickstart_falkordb.py --scenario employee --clear"
echo "  uv run python examples/quickstart/quickstart_falkordb.py --scenario customer --query-only"
echo "  uv run python examples/quickstart/quickstart_falkordb.py --scenario politics --clear"

# python quickstart_falkordb.py --scenario employee --clear
# python quickstart_falkordb.py --scenario customer --query-only
# python quickstart_falkordb.py --scenario politics --export
# python quickstart_falkordb.py --scenario politics --clear