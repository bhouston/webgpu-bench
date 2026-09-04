/**
 * Everything a page can learn about the machine it's running on from
 * standard browser APIs, gathered client-side with no network requests:
 * OS + version and browser + version (User-Agent Client Hints where
 * available, falling back to parsing the frozen UA string), CPU threads /
 * architecture / coarse RAM, the WebGL unmasked renderer string (the most
 * reliable way to learn the actual GPU — and on Apple silicon, the CPU —
 * model from a browser), and the WebGPU adapter's vendor/architecture.
 */
export interface EnvironmentInfo {
  os?: string;
  osVersion?: string;
  /** Where the OS fields came from: UA-CH is exact; the UA string is frozen/generalized in modern browsers. */
  osSource: 'client-hints' | 'user-agent' | 'unknown';
  browser?: string;
  browserVersion?: string;
  cpuArchitecture?: string;
  cpuThreads?: number;
  /** navigator.deviceMemory: coarse GiB buckets, capped at 8 in Chromium; undefined where unsupported. */
  memoryGb?: number;
  /** WEBGL_debug_renderer_info unmasked renderer, e.g. "ANGLE (Apple, ANGLE Metal Renderer: Apple M3, Unspecified Version)". */
  webglRenderer?: string;
  webglVendor?: string;
  /** The hardware name pulled out of the WebGL renderer string, e.g. "Apple M3" or "NVIDIA GeForce RTX 4090". */
  gpuModel?: string;
  webgpuVendor?: string;
  webgpuArchitecture?: string;
  webgpuDevice?: string;
  webgpuDescription?: string;
}

interface UaDataBrand {
  brand: string;
  version: string;
}

interface UaData {
  brands?: UaDataBrand[];
  platform?: string;
  getHighEntropyValues?: (hints: string[]) => Promise<{
    platform?: string;
    platformVersion?: string;
    architecture?: string;
    bitness?: string;
    fullVersionList?: UaDataBrand[];
  }>;
}

const OS_PATTERNS: Array<[RegExp, string, (m: RegExpMatchArray) => string | undefined]> = [
  [/iPhone OS ([\d_]+)/, 'iOS', (m) => m[1]?.replaceAll('_', '.')],
  [/iPad; CPU OS ([\d_]+)/, 'iPadOS', (m) => m[1]?.replaceAll('_', '.')],
  [/Mac OS X ([\d_]+)/, 'macOS', (m) => m[1]?.replaceAll('_', '.')],
  [/Windows NT ([\d.]+)/, 'Windows', (m) => m[1]],
  [/Android ([\d.]+)/, 'Android', (m) => m[1]],
  [/CrOS \S+ ([\d.]+)/, 'ChromeOS', (m) => m[1]],
  [/Linux/, 'Linux', () => undefined],
];

const BROWSER_PATTERNS: Array<[RegExp, string]> = [
  [/Edg\/([\d.]+)/, 'Microsoft Edge'],
  [/OPR\/([\d.]+)/, 'Opera'],
  [/Firefox\/([\d.]+)/, 'Firefox'],
  [/Chrome\/([\d.]+)/, 'Chrome'],
  [/Version\/([\d.]+).*Safari/, 'Safari'],
];

function pickBrand(brands: UaDataBrand[] | undefined): UaDataBrand | undefined {
  if (!brands) return undefined;
  const real = brands.filter((b) => !/not.?a.?brand/i.test(b.brand));
  // Prefer the specific product over the generic "Chromium" entry.
  return real.find((b) => b.brand !== 'Chromium') ?? real[0];
}

function parseUserAgent(ua: string): Pick<EnvironmentInfo, 'os' | 'osVersion' | 'browser' | 'browserVersion'> {
  const out: Pick<EnvironmentInfo, 'os' | 'osVersion' | 'browser' | 'browserVersion'> = {};
  for (const [re, name, version] of OS_PATTERNS) {
    const m = ua.match(re);
    if (m) {
      out.os = name;
      out.osVersion = version(m);
      break;
    }
  }
  for (const [re, name] of BROWSER_PATTERNS) {
    const m = ua.match(re);
    if (m) {
      out.browser = name;
      out.browserVersion = m[1];
      break;
    }
  }
  return out;
}

/** Pulls the hardware name out of a WebGL renderer string. */
export function extractGpuModel(renderer: string): string | undefined {
  // Chromium wraps everything in ANGLE: "ANGLE (Vendor, Renderer, Version)".
  const angle = renderer.match(/^ANGLE \((.*)\)$/);
  const inner = angle ? angle[1]! : renderer;
  const parts = inner.split(', ');
  // Metal: "Apple, ANGLE Metal Renderer: Apple M3, Unspecified Version". D3D:
  // "NVIDIA, NVIDIA GeForce RTX 4090 (0x00002684) Direct3D11 vs_5_0 ps_5_0, D3D11".
  const model = (parts.length >= 2 ? parts[1]! : parts[0]!)
    .replace(/^ANGLE Metal Renderer: /, '')
    .replace(/ \(0x[0-9a-fA-F]+\)/, '')
    .replace(/ Direct3D.*$/, '')
    .replace(/ OpenGL Engine$/, '')
    .trim();
  return model || undefined;
}

function collectWebGl(): Pick<EnvironmentInfo, 'webglRenderer' | 'webglVendor' | 'gpuModel'> {
  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2') ?? canvas.getContext('webgl');
    if (!gl) return {};
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    const renderer = ext ? String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)) : String(gl.getParameter(gl.RENDERER));
    const vendor = ext ? String(gl.getParameter(ext.UNMASKED_VENDOR_WEBGL)) : String(gl.getParameter(gl.VENDOR));
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    return { webglRenderer: renderer, webglVendor: vendor, gpuModel: extractGpuModel(renderer) };
  } catch {
    return {};
  }
}

async function collectWebGpu(): Promise<
  Pick<EnvironmentInfo, 'webgpuVendor' | 'webgpuArchitecture' | 'webgpuDevice' | 'webgpuDescription'>
> {
  try {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
    const info = adapter?.info;
    if (!info) return {};
    return {
      webgpuVendor: info.vendor || undefined,
      webgpuArchitecture: info.architecture || undefined,
      webgpuDevice: info.device || undefined,
      webgpuDescription: info.description || undefined,
    };
  } catch {
    return {};
  }
}

export async function collectEnvironmentInfo(): Promise<EnvironmentInfo> {
  const nav = navigator as Navigator & { userAgentData?: UaData; deviceMemory?: number };
  const info: EnvironmentInfo = { osSource: 'unknown' };

  const fromUa = parseUserAgent(nav.userAgent);
  Object.assign(info, fromUa);
  if (info.os) info.osSource = 'user-agent';

  const uaData = nav.userAgentData;
  if (uaData) {
    const brand = pickBrand(uaData.brands);
    if (brand) {
      info.browser = brand.brand;
      info.browserVersion = brand.version;
    }
    if (uaData.platform) info.os = uaData.platform;
    try {
      const high = await uaData.getHighEntropyValues?.(['platformVersion', 'architecture', 'bitness', 'fullVersionList']);
      if (high) {
        if (high.platform) info.os = high.platform;
        if (high.platformVersion) {
          info.osVersion = high.platformVersion;
          info.osSource = 'client-hints';
        }
        if (high.architecture) {
          info.cpuArchitecture = high.bitness ? `${high.architecture} (${high.bitness}-bit)` : high.architecture;
        }
        const full = pickBrand(high.fullVersionList);
        if (full) {
          info.browser = full.brand;
          info.browserVersion = full.version;
        }
      }
    } catch {
      // High-entropy hints can be denied; the low-entropy values above still stand.
    }
  }

  if (typeof nav.hardwareConcurrency === 'number') info.cpuThreads = nav.hardwareConcurrency;
  if (typeof nav.deviceMemory === 'number') info.memoryGb = nav.deviceMemory;

  Object.assign(info, collectWebGl());
  Object.assign(info, await collectWebGpu());
  return info;
}
