/**
 * Deterministic data generation for the benchmark suite. Every benchmark run
 * (in every browser, on every machine) sees byte-identical matrix/vector
 * inputs, so results are reproducible and correctness can be cross-checked
 * between kernels.
 */

/** Small, fast, seedable PRNG (mulberry32) — deterministic across platforms. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Round up to the next multiple of 4 (so vec4/mat4 kernels can index without bounds checks). */
export function padToMultipleOf4(n: number): number {
  return Math.ceil(n / 4) * 4;
}

export interface GeneratedData {
  rows: number;
  cols: number;
  /** row-major, rows x cols */
  matrix: Float32Array;
  vector: Float32Array;
}

/**
 * Generates a deterministic pseudo-random matrix and vector, values in
 * [-1, 1), sized to LLM-weight-like magnitudes (scaled by 1/sqrt(cols) so
 * dot products don't blow up in magnitude as cols grows).
 */
export function generateMatVecData(rows: number, cols: number, seed = 1234): GeneratedData {
  const paddedRows = padToMultipleOf4(rows);
  const paddedCols = padToMultipleOf4(cols);
  const rand = mulberry32(seed);
  const scale = 1 / Math.sqrt(paddedCols);

  const matrix = new Float32Array(paddedRows * paddedCols);
  for (let i = 0; i < matrix.length; i++) {
    matrix[i] = (rand() * 2 - 1) * scale;
  }

  const vector = new Float32Array(paddedCols);
  for (let i = 0; i < vector.length; i++) {
    vector[i] = rand() * 2 - 1;
  }

  return { rows: paddedRows, cols: paddedCols, matrix, vector };
}

/** IEEE-754 binary16 encode of a single float (used for the f16 storage buffers). */
export function f32ToF16Bits(value: number): number {
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  f32[0] = value;
  const x = u32[0]!;
  const sign = (x >>> 16) & 0x8000;
  let exp = ((x >>> 23) & 0xff) - 127 + 15;
  let mantissa = x & 0x7fffff;

  if (((x >>> 23) & 0xff) === 0xff) {
    // Inf/NaN
    return sign | 0x7c00 | (mantissa ? 0x200 : 0);
  }
  if (exp >= 0x1f) {
    // Overflow -> Inf
    return sign | 0x7c00;
  }
  if (exp <= 0) {
    // Subnormal or underflow to zero
    if (exp < -10) return sign;
    mantissa |= 0x800000;
    const shift = 14 - exp;
    return sign | (mantissa >> shift);
  }
  // Round to nearest-even on the dropped 13 mantissa bits.
  const roundBit = mantissa & 0x1000;
  mantissa >>= 13;
  if (roundBit !== 0 && (mantissa & 1 || (mantissa & 0xfff) !== 0)) {
    mantissa += 1;
    if (mantissa === 0x400) {
      mantissa = 0;
      exp += 1;
    }
  }
  return sign | (exp << 10) | mantissa;
}

/** Converts an f32 array to a Uint16Array of IEEE-754 half-float bit patterns. */
export function toF16Buffer(values: Float32Array): Uint16Array {
  const out = new Uint16Array(values.length);
  for (let i = 0; i < values.length; i++) {
    out[i] = f32ToF16Bits(values[i]!);
  }
  return out;
}

export interface QuantizedInt8 {
  data: Int8Array;
  scale: number;
}

/** Symmetric per-tensor int8 quantization: value ≈ q * scale, q in [-127, 127]. */
export function quantizeInt8(values: Float32Array): QuantizedInt8 {
  let maxAbs = 0;
  for (const v of values) maxAbs = Math.max(maxAbs, Math.abs(v));
  const scale = maxAbs > 0 ? maxAbs / 127 : 1;
  const data = new Int8Array(values.length);
  for (let i = 0; i < values.length; i++) {
    data[i] = Math.max(-127, Math.min(127, Math.round(values[i]! / scale)));
  }
  return { data, scale };
}

/** Packs 4 consecutive int8 values into one little-endian u32 (matches dot4I8Packed's expected layout). */
export function packInt8x4(data: Int8Array): Uint32Array {
  if (data.length % 4 !== 0) throw new Error('packInt8x4: length must be a multiple of 4');
  const out = new Uint32Array(data.length / 4);
  for (let i = 0; i < out.length; i++) {
    const b0 = data[i * 4]! & 0xff;
    const b1 = data[i * 4 + 1]! & 0xff;
    const b2 = data[i * 4 + 2]! & 0xff;
    const b3 = data[i * 4 + 3]! & 0xff;
    out[i] = (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)) >>> 0;
  }
  return out;
}

/** Reference CPU dot-product matvec, used to sanity-check GPU kernel correctness in dev/tests. */
export function referenceMatVec(matrix: Float32Array, vector: Float32Array, rows: number, cols: number): Float32Array {
  const out = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    let sum = 0;
    const base = r * cols;
    for (let c = 0; c < cols; c++) sum += matrix[base + c]! * vector[c]!;
    out[r] = sum;
  }
  return out;
}
