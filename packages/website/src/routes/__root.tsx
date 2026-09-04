import '../app.css';

import { createRootRoute, HeadContent, Outlet, Scripts } from '@tanstack/react-router';

export const Route = createRootRoute({
  head: () => ({
    meta: [
      { charSet: 'utf-8' },
      { name: 'viewport', content: 'width=device-width, initial-scale=1' },
      { title: 'WebGPU Profiler — Bandwidth & FLOPS Benchmarks' },
      {
        name: 'description',
        content:
          'Measures WebGPU read/write memory bandwidth and fp32/fp16/int8 FLOPS (scalar, vec4-SIMD, mat4-SIMD, and packed int8 dot-product) — this device\'s raw compute and memory ceilings.',
      },
    ],
  }),
  shellComponent: RootDocument,
  component: Outlet,
});

function RootDocument({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <HeadContent />
      </head>
      <body className="bg-background text-foreground min-h-svh antialiased">
        {children}
        <Scripts />
      </body>
    </html>
  );
}
