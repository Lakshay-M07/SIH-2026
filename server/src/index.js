import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import apiRouter from "./routes/index.js";
import { initFirebase } from "./firebase/config.js";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors());
app.use(express.json());

// Initialize Firebase in safe/decoupled mode
initFirebase();

// Root route
app.get("/", (_req, res) => {
  res.json({
    service: "SIH26162 backend",
    status: "initialized",
    version: "0.1.0",
    endpoints: {
      health: "/api/health",
    },
  });
});

// Mount modular API routes under /api
app.use("/api", apiRouter);

// Start server if run directly
const server = app.listen(PORT, () => {
  console.log(`Backend running on port ${PORT}`);
});

export { app, server };
export default app;
