FROM node:22-alpine AS builder

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

# --- Stage 2: Servir con Node ---
FROM node:22-alpine AS runtime

WORKDIR /app

COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./

ENV HOST=0.0.0.0
ENV PORT=4321

EXPOSE 4321

# Astro en modo static genera archivos estáticos, usamos un server simple
RUN npm install -g serve

CMD ["serve", "dist", "-l", "4321", "--single"]
