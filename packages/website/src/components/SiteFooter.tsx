import { HeartIcon } from 'lucide-react';

export function SiteFooter() {
  return (
    <footer className="border-t">
      <p className="text-muted-foreground flex items-center justify-center gap-1 px-4 py-3 text-sm">
        Made with
        <span className="inline-flex items-center">
          <HeartIcon className="size-3.5 fill-current text-destructive" aria-hidden />
          <span className="sr-only">love</span>
        </span>
        by
        <a href="https://ben3d.ca" className="text-foreground underline underline-offset-4">
          Ben Houston
        </a>
      </p>
    </footer>
  );
}
