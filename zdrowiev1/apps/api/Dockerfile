# Stage 1: Install dependencies
FROM node:24-alpine AS deps
RUN apk add --no-cache libc6-compat
WORKDIR /app
COPY package.json package-lock.json turbo.json ./
COPY apps/api/package.json ./apps/api/
COPY packages/zod-schemas/package.json ./packages/zod-schemas/
COPY packages/shared-types/package.json ./packages/shared-types/

# Module workspaces
COPY modules/activity/package.json ./modules/activity/
COPY modules/diagnosis/package.json ./modules/diagnosis/
COPY modules/diet/package.json ./modules/diet/
COPY modules/heart-rate/package.json ./modules/heart-rate/
COPY modules/notifications/package.json ./modules/notifications/
COPY modules/sleep/package.json ./modules/sleep/
COPY modules/weight/package.json ./modules/weight/

# Shared module workspaces
COPY modules/shared/auth/package.json ./modules/shared/auth/
COPY modules/shared/consent/package.json ./modules/shared/consent/
COPY modules/shared/database/package.json ./modules/shared/database/
COPY modules/shared/user/package.json ./modules/shared/user/

RUN npm ci

# Stage 2: Build
FROM node:24-alpine AS builder
RUN apk add --no-cache libc6-compat
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ENV NODE_OPTIONS="--no-experimental-strip-types"
RUN cd apps/api && npm run build

# Stage 3: Runtime
FROM node:24-alpine AS runner
RUN addgroup -g 1001 -S nodejs && adduser -S nestjs -u 1001 -G nodejs
WORKDIR /app

# Copy built files and production dependencies
COPY --from=builder --chown=nestjs:nodejs /app/dist ./dist
COPY --from=builder --chown=nestjs:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=nestjs:nodejs /app/package.json ./package.json

USER nestjs
EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000/api/health || exit 1

# Start the application
CMD ["node", "--no-experimental-strip-types", "dist/apps/api/apps/api/src/main.js"]
