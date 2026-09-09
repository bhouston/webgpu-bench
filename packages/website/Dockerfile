# syntax=docker/dockerfile:1.7
FROM --platform=linux/amd64 node:22-bookworm-slim AS base

ENV PNPM_HOME="/pnpm" \
    PATH="$PNPM_HOME:$PATH" \
    NODE_ENV=production \
    PORT=8080 \
    CI=true

RUN npm install -g pnpm@11.1.3

WORKDIR /app

COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
COPY packages/vgpu-bench/package.json packages/vgpu-bench/package.json
COPY packages/website/package.json packages/website/package.json

RUN pnpm fetch --reporter=silent

COPY . .

RUN pnpm install --frozen-lockfile --prefer-offline

RUN pnpm --filter vgpu-bench build
RUN pnpm --filter website build

EXPOSE 8080

CMD ["node", "packages/website/.output/server/index.mjs"]
