import { defineConfig } from '@playwright/test';
import path from 'path';

const repoRoot = path.resolve(__dirname, '..');
const e2eDataDir = path.join(repoRoot, '.tmp', 'annotation-app-e2e-data');
const backendPort = 5051;
const frontendPort = 3000;

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 60_000,
  expect: { timeout: 10_000 },
  use: {
    baseURL: `http://127.0.0.1:${frontendPort}`,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    headless: true,
  },
  webServer: [
    {
      command: `rm -rf "${e2eDataDir}" && mkdir -p "${e2eDataDir}" && ANNOTATION_APP_DATA_DIR="${e2eDataDir}" ANNOTATION_APP_HOST=127.0.0.1 ANNOTATION_APP_PORT=${backendPort} python annotation_app/app.py`,
      url: `http://127.0.0.1:${backendPort}/annotator`,
      reuseExistingServer: !process.env.CI,
      cwd: repoRoot,
    },
    {
      command: `BACKEND_URL=http://127.0.0.1:${backendPort} HOSTNAME=127.0.0.1 PORT=${frontendPort} npm run dev`,
      url: `http://127.0.0.1:${frontendPort}/annotator`,
      reuseExistingServer: !process.env.CI,
      cwd: __dirname,
    },
  ],
});
