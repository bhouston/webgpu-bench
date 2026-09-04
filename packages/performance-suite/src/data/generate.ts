/**
 * Deterministic data generation for the bandwidth benchmarks. Every
 * benchmark run (in every browser, on every machine) sees a byte-identical
 * buffer to stream, so results are reproducible run to run.
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
