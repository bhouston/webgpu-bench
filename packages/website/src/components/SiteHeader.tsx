import { CpuIcon, GithubIcon } from 'lucide-react';

import { Button } from '@/components/ui/button';

function NpmIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden {...props}>
      <path d="M0 0v24h24V0H0zm19.2 19.2h-4.8V8.4H9.6v10.8H4.8V4.8h14.4v14.4z" />
    </svg>
  );
}

export function SiteHeader() {
  return (
    <header className="border-b bg-background">
      <div className="mx-auto flex w-full max-w-6xl items-center justify-between gap-4 px-6 py-3">
        <div className="flex min-w-0 items-center gap-3">
          <CpuIcon className="size-6 shrink-0" aria-hidden />
          <h1 className="font-heading text-base font-medium tracking-tight">webgpu-bench</h1>
        </div>
        <div className="flex shrink-0 items-center gap-1">
          <Button variant="ghost" size="icon" asChild>
            <a href="https://github.com/bhouston/webgpu-profiler" aria-label="GitHub" title="GitHub">
              <GithubIcon />
            </a>
          </Button>
          <Button variant="ghost" size="icon" asChild>
            <a href="https://www.npmjs.com/package/webgpu-bench" aria-label="npm" title="npm">
              <NpmIcon />
            </a>
          </Button>
        </div>
      </div>
    </header>
  );
}
