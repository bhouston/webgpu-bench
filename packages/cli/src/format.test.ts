import { expect, test } from 'vitest';
import { formatRate, globToRegExp } from './format.ts';

test('globToRegExp', () => {
  expect(globToRegExp('f16-*').test('f16-vec4')).toBe(true);
  expect(globToRegExp('f16-*').test('f32-vec4')).toBe(false);
  expect(globToRegExp('read-*').test('read-linear')).toBe(true);
  expect(globToRegExp('f?2-div').test('f32-div')).toBe(true);
  expect(globToRegExp('read.linear').test('readxlinear')).toBe(false);
});

test('formatRate', () => {
  expect(formatRate(1.544e12, 'FLOP')).toBe('1.54 TFLOP/s');
  expect(formatRate(82_040_000_000, 'B')).toBe('82.04 GB/s');
  expect(formatRate(12, 'OP')).toBe('12.00 OP/s');
});
