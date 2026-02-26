---
description: How to start the full Zdrowie App stack (Docker + NestJS API + Next.js Frontend)
---

# Run All Servers — Zdrowie App

This workflow starts all required services for local development. Run from the monorepo root: `c:\od_zera_do_ai\Zdrowie_v1\zdrowiev1`

## Architecture Overview
- **PostgreSQL** (Docker) → port `5432` — database
- **Redis** (Docker) → port `6379` — cache
- **NestJS API** → port `3001` — backend
- **Next.js Frontend** → port `3000` — frontend (proxies `/api/*` → `localhost:3001`)

## Steps

### 1. Start Docker containers (PostgreSQL + Redis)
// turbo
```bash
docker-compose up -d
```
Wait until both `gesundheit-db` and `gesundheit-redis` are running. Verify with:
```bash
docker ps
```

### 2. Build the NestJS API
// turbo
```bash
cd apps/api; npm run build
```
This compiles TypeScript to `dist/` directory using NestJS CLI.

### 3. Start the NestJS API (port 3001)
```powershell
cd apps/api; $env:DOTENV_CONFIG_PATH="../../.env"; npx cross-env NODE_OPTIONS=--no-experimental-strip-types npx ts-node -r tsconfig-paths/register -r dotenv/config src/main.ts
```
> **Why not `npm run start`?** The compiled JS build has ESM resolution issues with `drizzle-orm`, and `dotenv` can't find `.env` from the `apps/api/` subdirectory. Using `ts-node` with `tsconfig-paths/register` and explicit `DOTENV_CONFIG_PATH` resolves both issues.

Wait for `Nest application successfully started` log.
The API seeds a demo user on startup (`demo@example.com` / `Password123!`).

### 4. Start the Next.js Frontend (port 3000)
In a **separate terminal**:
```bash
cd apps/web && npm run dev
```
Wait for `✓ Ready` log.

### 5. Verify login works
// turbo
```bash
powershell -File ../../scripts/check-stack.ps1
```
This script automatically checks: Docker containers, DB tables, API on port 3001, and login endpoint.
If ALL checks pass → stack is healthy. If any FAIL → follow the printed fix instructions.

### 6. Manual verification (optional)
// turbo
```bash
$body = '{"email":"demo@example.com","password":"Password123!"}'; Invoke-RestMethod -Uri http://localhost:3001/api/auth/login -Method POST -ContentType 'application/json' -Body $body
```
Should return an `access_token`. If yes, open `http://localhost:3000/login` in the browser.

## Demo Credentials
- **Email:** `demo@example.com`
- **Password:** `Password123!`

## Troubleshooting

### API returns 500 on login
- Check if Docker containers are running: `docker ps`
- Check if NestJS is running on port 3001: `curl.exe http://localhost:3001/api`
- Rebuild and restart the API (follow Steps 2 and 3 above)

### Next.js proxy not reaching API
- Verify `next.config.ts` rewrites destination points to `http://localhost:3001/api/:path*`
- Verify NestJS `main.ts` listens on port `3001`

### Database tables missing
```bash
docker exec gesundheit-db psql -U postgres -d health -c "\dt"
```
If tables are missing, restart the API — the NestJS `SeedModule` creates tables and the demo user on startup.

### Port conflict
- Next.js **must** be on port `3000`
- NestJS API **must** be on port `3001`
- If ports are taken, kill node processes: `Get-Process -Name node | Stop-Process -Force`
