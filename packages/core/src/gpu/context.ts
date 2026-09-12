import type { DeviceInfo } from '../types.ts';

export interface GpuContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  info: DeviceInfo;
}

const OPTIONAL_FEATURES: GPUFeatureName[] = ['shader-f16' as GPUFeatureName, 'timestamp-query' as GPUFeatureName];

// `packed_4x8_integer_dot_product` is a WGSL *language* extension (enabled in
// shader source via `enable packed_4x8_integer_dot_product;`), not a
// GPUFeatureName — it's never granted via requestDevice()'s requiredFeatures
// and never appears in adapter.features / device.features. Support is
// reported separately through navigator.gpu.wgslLanguageFeatures.
const WGSL_I8_DOT_EXTENSION = 'packed_4x8_integer_dot_product';

/**
 * Acquires a GPUAdapter/GPUDevice directly from the browser's `navigator.gpu`
 * (high-performance power preference), requesting every optional device
 * feature this suite can make use of — f16 shaders and GPU timestamp
 * queries — without failing if the device lacks some of them. Each benchmark
 * checks `info.supports*` and reports itself as "skipped" rather than
 * crashing the suite when a feature is missing.
 */
export async function acquireGpuContext(): Promise<GpuContext> {
  if (!('gpu' in navigator) || !navigator.gpu) {
    throw new Error('WebGPU is not available in this browser (navigator.gpu is undefined).');
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    throw new Error('No suitable WebGPU adapter was found.');
  }

  const requiredFeatures = OPTIONAL_FEATURES.filter((f) => adapter.features.has(f));

  const requiredLimits: Record<string, number> = {};
  // Ask for the largest storage-buffer binding & workgroup budgets the adapter allows,
  // so large matrices and tuned workgroup sizes aren't capped by conservative defaults.
  for (const key of [
    'maxStorageBufferBindingSize',
    'maxBufferSize',
    'maxComputeWorkgroupSizeX',
    'maxComputeInvocationsPerWorkgroup',
    'maxComputeWorkgroupStorageSize',
  ] as const) {
    const value = (adapter.limits as unknown as Record<string, number>)[key];
    if (typeof value === 'number') requiredLimits[key] = value;
  }

  const device = await adapter.requestDevice({ requiredFeatures, requiredLimits });

  const limits: Record<string, number> = {};
  // `for...in`, not Object.keys: GPUSupportedLimits' values are getters on
  // its prototype (in browsers and Dawn's node binding alike), so they're
  // inherited-enumerable, not own properties.
  for (const key in device.limits) {
    const value = device.limits[key as keyof GPUSupportedLimits];
    if (typeof value === 'number') limits[key] = value;
  }

  let adapterInfoDescription: string | undefined;
  let vendor: string | undefined;
  let architecture: string | undefined;
  const anyAdapter = adapter as GPUAdapter & { info?: GPUAdapterInfo };
  if (anyAdapter.info) {
    vendor = anyAdapter.info.vendor || undefined;
    architecture = anyAdapter.info.architecture || undefined;
    adapterInfoDescription = anyAdapter.info.description || undefined;
  }

  const info: DeviceInfo = {
    vendor,
    architecture,
    description: adapterInfoDescription,
    features: [...device.features].toSorted(),
    limits,
    supportsF16: device.features.has('shader-f16' as GPUFeatureName),
    supportsI8Dot: Boolean(navigator.gpu.wgslLanguageFeatures?.has(WGSL_I8_DOT_EXTENSION)),
    supportsTimestampQuery: device.features.has('timestamp-query' as GPUFeatureName),
  };

  device.addEventListener('uncapturederror', (event) => {
    // eslint-disable-next-line no-console
    console.error('WebGPU uncaptured error:', (event as GPUUncapturedErrorEvent).error);
  });

  return { adapter, device, info };
}
