import '../app.css';

import { createRootRoute, HeadContent, Outlet, Scripts } from '@tanstack/react-router';

export const Route = createRootRoute({
  head: () => ({
    meta: [
      { charSet: 'utf-8' },
      { name: 'viewport', content: 'width=device-width, initial-scale=1' },
      { title: 'WebGPU Profiler — LLM MatVec Benchmarks' },
      {
        name: 'description',
        content:
          'Benchmarks WebGPU compute-shader strategies for large matrix-vector multiplication (the core op of LLM inference): f32, f16, int8, SIMD layouts, tuned workgroup sizes, and fused multi-layer MLPs.',
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
