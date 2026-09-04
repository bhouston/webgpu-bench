import '../app.css';

import { createRootRoute, HeadContent, Outlet, Scripts } from '@tanstack/react-router';

import { SiteFooter } from '@/components/SiteFooter';
import { SiteHeader } from '@/components/SiteHeader';

export const Route = createRootRoute({
  head: () => ({
    meta: [
      { charSet: 'utf-8' },
      { name: 'viewport', content: 'width=device-width, initial-scale=1' },
      { title: 'webgpu-bench — Bandwidth & FLOPS Benchmarks' },
      {
        name: 'description',
        content:
          "Measures WebGPU read/write memory bandwidth and fp32/fp16/int8 FLOPS — this device's raw compute and memory ceilings.",
      },
    ],
  }),
  shellComponent: RootDocument,
  component: RootLayout,
});

function RootLayout() {
  return (
    <div className="flex min-h-svh flex-col">
      <SiteHeader />
      <div className="flex min-h-0 flex-1 flex-col">
        <Outlet />
      </div>
      <SiteFooter />
    </div>
  );
}

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
