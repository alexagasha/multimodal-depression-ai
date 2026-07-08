import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies /api -> the local FastAPI backend, so the frontend
// code can always just call fetch("/api/...") — no CORS juggling, and the
// exact same relative-path calls work once both are deployed behind one
// origin (or with VITE_API_BASE_URL set — see src/api.ts).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
