# Petopia Frontend

React + Vite dashboard for the Petopia Intelligence Hub.

## Local dev

```bash
npm install
npm run dev   # http://localhost:5173
```

## Environment

Set `VITE_API_BASE` to your backend URL (HF Space), e.g.
`https://<user>-petopia-backend.hf.space`. Defaults to `http://localhost:8000`.

## Test / build

```bash
npm run test
npm run build
```

## Deploy (Vercel)

Import the repo, set **Root Directory = frontend**, add env var `VITE_API_BASE`,
deploy. SPA routing handled by `vercel.json`.
