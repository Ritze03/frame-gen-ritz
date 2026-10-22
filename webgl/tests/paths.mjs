// Shared locations of the WebGL2 test harness (webgl/tests). Everything the tests write goes to $FG_WORK
// (default <repo>/.local-work/webgl, gitignored).
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

export const here = path.dirname(fileURLToPath(import.meta.url));
export const webgl = path.resolve(here, '..');
export const repo = path.resolve(webgl, '..');
export const work = process.env.FG_WORK || path.join(repo, '.local-work', 'webgl');
export const engineFile = path.join(webgl, 'framegen.js');
export const shaderDir = path.join(webgl, 'shaders');

// The Vulkan library vkhdr.cpp links against: $FG_GPU (its gpu/ sources, default <repo>/gpu) and $FG_GPU_BUILD (a build dir with
// libframegen.a, default <repo>/gpu/build-webgl), named $FG_VKTAG (default 'repo'; the binary and the HDR references are per tag,
// so another library version, e.g. a branch checked out with git archive, can be compared side by side).
export const vkTag = process.env.FG_VKTAG || 'repo';
export function vkhdrBinary() {
  const bin = path.join(work, 'bin', 'vkhdr-' + vkTag);
  if (fs.existsSync(bin)) return bin;
  fs.mkdirSync(path.dirname(bin), { recursive: true });
  const gpu = process.env.FG_GPU || path.join(repo, 'gpu');
  const lib = path.join(process.env.FG_GPU_BUILD || path.join(repo, 'gpu', 'build-webgl'), 'libframegen.a');
  const r = spawnSync('g++', ['-O2', '-std=c++17', '-I', gpu, path.join(here, 'vkhdr.cpp'), lib, '-lvulkan', '-o', bin], { stdio: 'inherit' });
  if (r.status !== 0) { console.error(`building vkhdr failed (libframegen.a at ${lib}: run vkref.mjs --build-only first)`); process.exit(2); }
  return bin;
}
