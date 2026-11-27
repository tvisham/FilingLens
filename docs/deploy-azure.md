# Deploying to Azure (optional)

Not done automatically: this creates billable cloud resources. A runbook for when you want it.

| Piece | Azure service |
|---|---|
| Postgres + pgvector | Azure Database for PostgreSQL – Flexible Server (allow-list the `vector` extension: server parameter `azure.extensions = VECTOR`) |
| API (`Dockerfile`, port 8000) | Azure Container Apps |
| MCP server (same image, `mcp` argument, port 8765) | Azure Container Apps, internal ingress only |
| Web (`web/Dockerfile`, port 3000) | Azure Container Apps (or Static Web Apps + the API) |
| Secrets | Container Apps secrets / Key Vault references |
| Nightly eval | Container Apps job on a cron schedule, or keep the GitHub Actions workflow |

## Steps

```bash
az group create -n filinglens -l eastus
az postgres flexible-server create -g filinglens -n filinglens-db --tier Burstable --sku-name Standard_B1ms \
  --version 16 --public-access None            # private networking recommended
az postgres flexible-server parameter set -g filinglens -s filinglens-db --name azure.extensions --value VECTOR

az acr create -g filinglens -n <registry> --sku Basic
az acr build -r <registry> -t filinglens:latest .
az acr build -r <registry> -t filinglens-web:latest --build-arg NEXT_PUBLIC_FILINGLENS_API_URL=https://<api-host> web

az containerapp env create -g filinglens -n filinglens-env -l eastus
az containerapp create -g filinglens -n filinglens-api --environment filinglens-env \
  --image <registry>.azurecr.io/filinglens:latest --target-port 8000 --ingress external \
  --secrets db-url=<postgresql+psycopg://...?sslmode=require> sec-ua="FilingLens <name> <email>" \
            anthropic-key=<key> api-token=<random> \
  --env-vars DATABASE_URL=secretref:db-url SEC_USER_AGENT=secretref:sec-ua \
             ANTHROPIC_API_KEY=secretref:anthropic-key API_TOKEN=secretref:api-token \
             API_CORS_ORIGINS='["https://<web-host>"]'
```

Then run the migrations once from a job or a shell with the same environment:
`alembic upgrade head`.

## Before exposing it

- Set `API_TOKEN` (review decisions require it) and restrict `API_CORS_ORIGINS` to the web host.
- Keep the MCP server on internal ingress, or put an authenticating proxy in front of it.
- The SEC response cache (`CACHE_DIR`) is per-container; mount an Azure Files share if you
  run several replicas, to stay well under SEC's request-rate limit.
- Azure OpenAI as a second provider for the eval comparison is not implemented; the
  `Classifier` and `Judge` take any client with the Messages API shape, which would be the
  place to add it.
