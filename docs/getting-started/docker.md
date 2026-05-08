---
title: Docker Setup
---

# Docker Setup

The fastest way to run InterSystems IRIS locally is with the official Community Edition Docker image.

---

## Start IRIS with a single command

```bash
docker run -d \
  --name iris \
  -p 1972:1972 \
  -p 52773:52773 \
  intersystemsdc/iris-community:latest
```

| Flag | Purpose |
|---|---|
| `-d` | Run in detached (background) mode |
| `--name iris` | Name the container for easy reference |
| `-p 1972:1972` | Superserver port — used by the Python DB-API driver |
| `-p 52773:52773` | Web port — Management Portal and REST APIs |

---

## Using docker-compose (recommended for projects)

Create a `docker-compose.yml` at the root of your project:

```yaml title="docker-compose.yml"
version: "3.8"

services:
  iris:
    image: intersystemsdc/iris-community:latest
    container_name: iris
    ports:
      - "1972:1972"    # DB-API superserver
      - "52773:52773"  # Management Portal
    volumes:
      - iris-data:/usr/irissys/mgr   # persist data between restarts
    healthcheck:
      test: ["CMD", "iris", "status"]
      interval: 15s
      timeout: 10s
      retries: 5
      start_period: 30s

volumes:
  iris-data:
```

Start and stop:

```bash
docker-compose up -d      # start in background
docker-compose down       # stop (data persists in volume)
docker-compose down -v    # stop AND delete all data
```

---

## Verify IRIS is running

```bash
# Check container status
docker ps --filter name=iris

# Check logs
docker logs iris --tail 30
```

You should see `IRIS for UNIX ... startup successful` in the logs.

---

## Management Portal

Open [http://localhost:52773/csp/sys/UtilHome.csp](http://localhost:52773/csp/sys/UtilHome.csp) in your browser.

Default credentials:

| Field | Value |
|---|---|
| Username | `_system` |
| Password | `SYS` |

### Exploring your data via SQL

Once documents are indexed, you can inspect them directly:

1. Navigate to **System Explorer → SQL**
2. Change the namespace to **USER** (top-left dropdown)
3. Run any SQL query:

```sql
-- List all indexed documents
SELECT id, content, meta FROM SQLUser.HaystackDocuments

-- Count documents
SELECT COUNT(*) FROM SQLUser.HaystackDocuments

-- Self-similarity check (should return 1.0 for each row)
SELECT id, VECTOR_COSINE(embedding, embedding) AS self_sim
FROM SQLUser.HaystackDocuments
WHERE embedding IS NOT NULL
```

---

## Connecting with Python

```python
import iris

conn = iris.connect("localhost:1972/USER", "_system", "SYS")
cur = conn.cursor()
cur.execute("SELECT 1")
print(cur.fetchone())   # (1,)
conn.close()
```

If this works, IRIS is ready and `iris-haystack` can connect.

---

## Common issues

### Port already in use

```
Error: Bind for 0.0.0.0:1972 failed: port is already allocated
```

Change the host port in `docker-compose.yml`:

```yaml
ports:
  - "1973:1972"   # host:container
```

Then update your connection string: `localhost:1973/USER`.

### Container exits immediately

```bash
docker logs iris
```

This usually means the IRIS data directory has permission issues. Try removing the volume:

```bash
docker-compose down -v
docker-compose up -d
```

### Cannot connect from Python

Check that the superserver port is open:

```bash
nc -zv localhost 1972
# Connection to localhost 1972 port [tcp/*] succeeded!
```

If this fails, the container is not running or the port is blocked by a firewall.