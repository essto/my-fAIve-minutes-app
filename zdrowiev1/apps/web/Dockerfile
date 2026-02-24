# Stage 1: Prune
FROM node:24-alpine AS pruner
RUN apk add --no-cache libc6-compat
WORKDIR /app
# Install turbo globally for pruning
RUN npm install -g turbo
COPY . .
# Prune the workspace for the web app
RUN turbo prune web --docker

# Stage 2: Install and Build
FROM node:24-alpine AS builder
RUN apk add --no-cache libc6-compat
WORKDIR /app

# First install the dependencies
COPY .gitignore .dockerignore ./
COPY --from=pruner /app/out/json/ .
COPY --from=pruner /app/out/package-lock.json ./package-lock.json
RUN npm ci

# Build the project
COPY --from=pruner /app/out/full/ .
# Note: In Next.js with standalone output, the build must trace dependencies accurately.
RUN npx turbo run build --filter=web

# Stage 3: Runtime
FROM node:24-alpine AS runner
RUN addgroup -g 1001 -S nodejs && adduser -S nextjs -u 1001 -G nodejs
WORKDIR /app

ENV NODE_ENV=production
ENV PORT=3001
ENV HOSTNAME=0.0.0.0

# Copy Next.js standalone output
# The standalone build includes minimal node_modules required for production
COPY --from=builder --chown=nextjs:nodejs /app/apps/web/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/apps/web/.next/static ./apps/web/.next/static
COPY --from=builder --chown=nextjs:nodejs /app/apps/web/public ./apps/web/public

USER nextjs
EXPOSE 3001

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3001/ || exit 1

# Start standalone server
CMD ["node", "apps/web/server.js"]
