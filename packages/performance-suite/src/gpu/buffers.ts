/** Creates a GPU buffer, uploads `data` into it, and returns the buffer. */
export function createUploadedBuffer(
  device: GPUDevice,
  data: ArrayBufferView,
  usage: GPUBufferUsageFlags,
  label?: string,
): GPUBuffer {
  const buffer = device.createBuffer({
    label,
    size: Math.max(4, Math.ceil(data.byteLength / 4) * 4),
    usage,
    mappedAtCreation: true,
  });
  new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  buffer.unmap();
  return buffer;
}

export function createStorageBuffer(device: GPUDevice, data: ArrayBufferView, label?: string): GPUBuffer {
  return createUploadedBuffer(
    device,
    data,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    label,
  );
}

export function createUniformBuffer(device: GPUDevice, data: ArrayBufferView, label?: string): GPUBuffer {
  return createUploadedBuffer(device, data, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label);
}

export function createEmptyStorageBuffer(device: GPUDevice, byteLength: number, label?: string): GPUBuffer {
  return device.createBuffer({
    label,
    size: Math.max(4, Math.ceil(byteLength / 4) * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
}

/** Reads a storage buffer back to the CPU as a Float32Array (used only for correctness spot-checks, not timing). */
export async function readFloat32(device: GPUDevice, buffer: GPUBuffer, count: number): Promise<Float32Array> {
  const staging = device.createBuffer({
    size: count * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, count * 4);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return result;
}
