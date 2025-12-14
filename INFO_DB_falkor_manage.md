To apply this change:

- Stop the current container: docker-compose --profile falkordb down

- Start it again: docker-compose --profile falkordb up -d

After restarting, FalkorDB data (including dump.rdb) will be persisted to ./data/falkordb/test1 on your host machine.
