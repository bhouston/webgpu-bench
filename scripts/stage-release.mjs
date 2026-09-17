import { cpSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));
const readPackage = (name) => JSON.parse(readFileSync(`${root}packages/${name}/package.json`, 'utf8'));
const version = process.argv[2] ?? readPackage('core').version;
if (!/^\d+\.\d+\.\d+$/.test(version)) throw new Error(`Invalid release version: ${version}`);

// Only staged manifests change; source manifests retain their development versions.
// Stage before semantic-release so both npm plugins can verify their package roots.
for (const name of ['core', 'cli']) {
  const dir = `${root}packages/${name}`;
  const target = `${dir}/publish`;
  const pkg = readPackage(name);
  pkg.version = version;
  delete pkg.scripts;
  delete pkg.devDependencies;
  if (name === 'cli') pkg.dependencies['webgpu-bench-core'] = version;
  pkg.publishConfig = { access: 'public' };
  rmSync(target, { recursive: true, force: true });
  mkdirSync(target, { recursive: true });
  cpSync(`${dir}/dist`, `${target}/dist`, { recursive: true });
  cpSync(`${dir}/README.md`, `${target}/README.md`);
  cpSync(`${root}LICENSE`, `${target}/LICENSE`);
  writeFileSync(`${target}/package.json`, `${JSON.stringify(pkg, null, 2)}\n`);
}
