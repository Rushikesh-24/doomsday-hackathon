# AccessMap web app

Front end for the PS32 campus accessibility platform. It needs the Python API running (see the main
README); it holds no data of its own.

```bash
pnpm install
```
```bash
pnpm dev
```

Open http://localhost:3000. API location: `NEXT_PUBLIC_API_URL` in `.env.local` (default
`http://localhost:8000`).

Checks: `pnpm lint`, `pnpm build`.

Screens, endpoints and the demo flow: [../docs/PLATFORM.md](../docs/PLATFORM.md).
