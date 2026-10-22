// Locate playwright-core + a Chromium headless shell, launch with SwiftShader WebGL2.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));

function newestDir(parent, prefix) {
  try {
    return fs.readdirSync(parent).filter((d) => d.startsWith(prefix))
      .sort((a, b) => parseInt(b.slice(prefix.length), 10) - parseInt(a.slice(prefix.length), 10))
      .map((d) => path.join(parent, d));
  } catch { return []; }
}

export function loadPlaywright() {
  const tried = [];
  const cands = [];
  if (process.env.PLAYWRIGHT_CORE) cands.push(process.env.PLAYWRIGHT_CORE);
  cands.push(path.join(here, 'node_modules', 'playwright-core'));
  const npx = path.join(os.homedir(), '.npm', '_npx');
  try {
    for (const d of fs.readdirSync(npx)) cands.push(path.join(npx, d, 'node_modules', 'playwright-core'));
  } catch { /* none */ }
  for (const c of cands) {
    tried.push(c);
    if (!fs.existsSync(path.join(c, 'package.json'))) continue;
    try {
      const req = createRequire(path.join(c, 'package.json'));
      const pw = req(c);
      if (pw.chromium) return { chromium: pw.chromium, from: c };
    } catch { /* try next */ }
  }
  throw new Error('playwright-core not found. Tried:\n  ' + tried.join('\n  ') +
    '\nFix: (cd tools/golden && npm install) or set PLAYWRIGHT_CORE=/path/to/node_modules/playwright-core');
}

export function findChromium() {
  if (process.env.FG_CHROMIUM) return process.env.FG_CHROMIUM;
  const root = process.env.PLAYWRIGHT_BROWSERS_PATH || path.join(os.homedir(), '.cache', 'ms-playwright');
  for (const d of newestDir(root, 'chromium_headless_shell-')) {
    const p = path.join(d, 'chrome-headless-shell-linux64', 'chrome-headless-shell');
    if (fs.existsSync(p)) return p;
  }
  for (const d of newestDir(root, 'chromium-')) {
    const p = path.join(d, 'chrome-linux64', 'chrome');
    if (fs.existsSync(p)) return p;
  }
  return null; // let playwright decide
}

export const CHROMIUM_ARGS = [
  '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
  '--ignore-gpu-blocklist', '--disable-gpu-sandbox', '--no-sandbox',
];

// Hardware GL (bench.mjs --gpu): ANGLE on the system's OpenGL / Vulkan driver instead of SwiftShader.
export const CHROMIUM_GPU_ARGS = [
  '--use-gl=angle', '--use-angle=' + (process.env.FG_ANGLE || 'vulkan'), '--enable-features=Vulkan', '--ignore-gpu-blocklist',
  '--disable-gpu-sandbox', '--no-sandbox', '--enable-gpu', '--disable-vsync-for-tests', '--disable-frame-rate-limit',
];

export async function launch({ gpu = false } = {}) {
  const { chromium, from } = loadPlaywright();
  const executablePath = findChromium();
  const browser = await chromium.launch({ headless: true, executablePath: executablePath || undefined, args: gpu ? CHROMIUM_GPU_ARGS : CHROMIUM_ARGS });
  return { browser, playwright: from, executablePath };
}
