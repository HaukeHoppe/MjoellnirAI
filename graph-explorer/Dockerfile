# Graph explorer: TypeScript build (type check, unit tests, Vite) -> static files served by Caddy.
# The concept graph is not baked in; it is mounted at /srv/data (see docker-compose.yaml).
FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY . .
RUN npm test && npm run build

FROM caddy:2.11-alpine
COPY Caddyfile /etc/caddy/Caddyfile
COPY --from=build /app/dist /srv
