import express from "express";
import cors from "cors";
import path from "node:path";
import { fileURLToPath } from "node:url";
import dotenv from "dotenv";
import apiRoutes from "./routes/api.routes.js";
import { firmsIngestService } from "./services/firmsIngestService.js";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, "../..");

const app = express();
const PORT = process.env.PORT || 5001;

// Enable CORS for frontend running on port 3000, 5173, or any local origin
app.use(
  cors({
    origin: true,
    credentials: true,
  })
);

app.use(express.json());

// Request logger
app.use((req, _res, next) => {
  console.log(`[${new Date().toISOString().slice(11, 19)}] ${req.method} ${req.url}`);
  next();
});

// Serve Sentinel chip GeoTIFFs/metadata statically if requested
const chipsDir = path.join(REPO_ROOT, "ml/data/images/demo_chips");
app.use("/chips", express.static(chipsDir));
app.use("/api/chips", express.static(chipsDir));

// Mount API routes under /api
app.use("/api", apiRoutes);

// Also mount apiRoutes directly on / for direct requests
app.use(apiRoutes);

// Serve client/dist static assets as a web fallback
const distDir = path.join(REPO_ROOT, "client/dist");
if (express.static(distDir)) {
  app.use(express.static(distDir));
  app.get("*", (req, res, next) => {
    if (req.path.startsWith("/api") || req.path.startsWith("/chips")) {
      return next();
    }
    const indexPath = path.join(distDir, "index.html");
    res.sendFile(indexPath, (err) => {
      if (err) next();
    });
  });
}

const server = app.listen(PORT, () => {
  console.log(`=======================================================`);
  console.log(`🔥 THERMOGRID Backend Running on http://localhost:${PORT}`);
  console.log(`📊 API Base URL: http://localhost:${PORT}/api`);
  console.log(`📡 Health Check: http://localhost:${PORT}/api/health`);
  console.log(`=======================================================`);
  // Start automated 3-hour NASA FIRMS ingestion scheduler
  firmsIngestService.startScheduler();
});

export { app, server };
export default app;
