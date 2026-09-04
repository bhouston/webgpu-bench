import { createFileRoute } from '@tanstack/react-router';

import { BenchmarkSuiteApp } from '@/components/BenchmarkSuiteApp';

export const Route = createFileRoute('/')({
  ssr: false,
  component: BenchmarkSuiteApp,
});
