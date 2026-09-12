/**
 * Creates a 1-row `rgba32float` 2D texture (WebGPU's 1D textures are patchy
 * across backends, so a width x 1 texture stands in for one) and uploads
 * `data` (vec4-packed, `texelCount * 4` floats) into it.
 */
export function createFloat32Texture(
  device: GPUDevice,
  data: Float32Array,
  texelCount: number,
  label?: string,
): GPUTexture {
  const texture = device.createTexture({
    label,
    size: [texelCount, 1, 1],
    format: 'rgba32float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  device.queue.writeTexture(
    { texture },
    data as unknown as ArrayBuffer,
    { bytesPerRow: texelCount * 16 },
    { width: texelCount, height: 1 },
  );
  return texture;
}
