import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  // Pin the workspace root to this app: a sibling package-lock.json one
  // level up (unrelated, pre-existing tooling in ../) otherwise makes
  // Next.js guess wrong.
  turbopack: {
    root: path.join(__dirname),
  },
};

export default nextConfig;
