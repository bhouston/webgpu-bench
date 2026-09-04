/**
 * Wraps a single compute pass in GPU timestamp queries when the device
 * supports `timestamp-query`, giving pure device-side elapsed time (no CPU
 * dispatch/driver overhead in the number). Falls back to CPU wall-clock
 * around `queue.onSubmittedWorkDone()` otherwise — still a fair comparison
 * across kernels, just with a little more noise.
 */
export class GpuTimer {
  private readonly querySet: GPUQuerySet | null = null;
  private readonly resolveBuffer: GPUBuffer | null = null;
  private readonly resultBuffer: GPUBuffer | null = null;

  constructor(
    device: GPUDevice,
    readonly supported: boolean,
  ) {
    if (supported) {
      this.querySet = device.createQuerySet({ type: 'timestamp', count: 2 });
      this.resolveBuffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      this.resultBuffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
    }
  }

  get timestampWrites(): GPUComputePassTimestampWrites | undefined {
    if (!this.supported || !this.querySet) return undefined;
    return { querySet: this.querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 };
  }

  resolve(encoder: GPUCommandEncoder): void {
    if (!this.supported || !this.querySet || !this.resolveBuffer || !this.resultBuffer) return;
    encoder.resolveQuerySet(this.querySet, 0, 2, this.resolveBuffer, 0);
    encoder.copyBufferToBuffer(this.resolveBuffer, 0, this.resultBuffer, 0, 16);
  }

  /** Reads back the two timestamps written by the most recently submitted pass and returns elapsed ms. */
  async readElapsedMs(): Promise<number> {
    if (!this.supported || !this.resultBuffer) return Number.NaN;
    await this.resultBuffer.mapAsync(GPUMapMode.READ);
    const times = new BigInt64Array(this.resultBuffer.getMappedRange().slice(0));
    this.resultBuffer.unmap();
    const start = times[0]!;
    const end = times[1]!;
    return Number(end - start) / 1e6; // nanoseconds -> milliseconds
  }

  destroy(): void {
    this.querySet?.destroy();
    this.resolveBuffer?.destroy();
    this.resultBuffer?.destroy();
  }
}
