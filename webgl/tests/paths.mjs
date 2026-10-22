// Shared locations of the WebGL2 test harness (webgl/tests). Everything the tests write goes to $FG_WORK
// (default <repo>/.local-work/webgl, gitignored).
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const here = path.dirname(fileURLToPath(import.meta.url));
export const webgl = path.resolve(here, '..');
export const repo = path.resolve(webgl, '..');
export const work = process.env.FG_WORK || path.join(repo, '.local-work', 'webgl');
export const engineFile = path.join(webgl, 'framegen.js');
export const shaderDir = path.join(webgl, 'shaders');
