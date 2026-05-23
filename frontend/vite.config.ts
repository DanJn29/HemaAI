import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const proxyTarget = env.FRONTEND_API_PROXY_TARGET || "http://localhost:8000";
  const frontendPort = Number(env.FRONTEND_PORT || "5174");

  return {
    plugins: [react()],
    server: {
      host: "0.0.0.0",
      port: frontendPort,
      proxy: {
        "/api": {
          target: proxyTarget,
          changeOrigin: true
        }
      }
    }
  };
});
