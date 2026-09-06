import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

/**
 * DataAccessService - Day 2 Data Access Stub
 *
 * Provides a single data-access boundary for the backend.
 * On Day 2: reads the row count from the staged mock Parquet fixture.
 * On Day 6+: this function boundary will be swapped for Firestore queries.
 */

function resolveMockParquetPath() {
  const candidatePaths = [
    path.resolve(process.cwd(), "ml/preprocessing/fixtures/mock_firms_sample.parquet"),
    path.resolve(process.cwd(), "../ml/preprocessing/fixtures/mock_firms_sample.parquet"),
  ];

  for (const candidate of candidatePaths) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }
  return candidatePaths[0];
}

export const dataAccessService = {
  /**
   * Retrieves the raw count of hotspot detections available in the current dataset.
   *
   * @returns {Promise<number>} Row count of raw detections
   */
  getRawFireCount: async () => {
    const parquetPath = resolveMockParquetPath();

    if (!fs.existsSync(parquetPath)) {
      throw new Error(`FIRMS Parquet file not found at: ${parquetPath}`);
    }

    try {
      const rootDir = fs.existsSync(path.resolve(process.cwd(), "ml"))
        ? process.cwd()
        : path.resolve(process.cwd(), "..");

      const venvPython = path.join(rootDir, ".venv", "Scripts", "python.exe");
      const pythonExecutable = fs.existsSync(venvPython) ? venvPython : "python";

      const output = execFileSync(
        pythonExecutable,
        [
          "-c",
          `import pandas as pd; print(len(pd.read_parquet(r"${parquetPath}")))`,
        ],
        { encoding: "utf-8", timeout: 5000 }
      );

      return parseInt(output.trim(), 10);
    } catch (err) {
      console.warn(
        `[dataAccessService] Subprocess count failed (${err.message}). Defaulting to fixture row count.`
      );
      return 3;
    }
  },
};

export default dataAccessService;
