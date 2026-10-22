#!/usr/bin/env node
// Validates every program of the WebGL2 engine with glslangValidator, on exactly the text the host compiles (__fg.assemblePrograms:
// common.glsl + defines + include + file for fragment shaders, vertex shaders standalone). One OK / FAIL line per shader.
//   node webgl/tests/validate.mjs [--shaders DIR]
// Exit 0 = all OK, 1 = a shader failed, 2 = setup error (glslangValidator missing, a source file missing).
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { engineFile, shaderDir } from './paths.mjs';

const argv = process.argv.slice(2);
let dir = shaderDir;
for (let i = 0; i < argv.length; i++) if (argv[i] === '--shaders') dir = path.resolve(argv[++i]);
createRequire(import.meta.url)(engineFile);   // classic script: attaches globalThis.__fg
const fg = globalThis.__fg;
const sources = {};
for (const f of fg.shaderFiles) {
  const p = path.join(dir, f);
  if (!fs.existsSync(p)) { console.error('validate: missing ' + p); process.exit(2); }
  sources[f] = fs.readFileSync(p, 'utf8');
}
const probe = spawnSync('glslangValidator', ['--version']);
if (probe.error) { console.error('validate: glslangValidator not found in PATH'); process.exit(2); }

let n = 0, fail = 0;
function check(label, stage, src) {
  n++;
  // flag order matters in some glslangValidator builds: --stdin before -S
  const r = spawnSync('glslangValidator', ['--stdin', '-S', stage], { input: src, encoding: 'utf8' });
  if (r.status === 0) console.log('OK    ' + label);
  else { fail++; console.log('FAIL  ' + label + '\n' + (r.stdout + r.stderr).split('\n').map((l) => '      ' + l).join('\n')); }
}
const seenVert = new Set();
// every frame-format variant the host can build (common.glsl "frame formats")
for (const [tag, fmt] of [['rgba8', {}], ['rgba16f', { inputFormat: 'rgba16f' }], ['rgba16f linear', { inputFormat: 'rgba16f', transfer: 'linear' }]]) {
  for (const p of fg.assemblePrograms(sources, fmt)) {
    if (!seenVert.has(p.vertexFile)) { seenVert.add(p.vertexFile); check(p.vertexFile, 'vert', p.vertex); }
    check(`${p.fragmentFile} (${p.name}, ${tag})`, 'frag', p.fragment);
  }
}
console.log(`---- ${n - fail}/${n} OK`);
process.exit(fail ? 1 : 0);
