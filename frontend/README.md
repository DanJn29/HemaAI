# HemaAI Frontend

React + TypeScript dashboard for the HemaAI backend.

## Local Run

```bash
cd frontend
npm install
FRONTEND_PORT=5174 npm run dev -- --port 5174
```

By default, Vite proxies `/api` requests to `http://localhost:8000`.

## Build

```bash
cd frontend
npm run build
```

## Environment

- `VITE_API_BASE_URL` defaults to `/api/v1`
- `FRONTEND_API_PROXY_TARGET` defaults to `http://localhost:8000`
- `FRONTEND_PORT` defaults to `5174`
