# FilingLens web app (Next.js). The API URL is read at build time for client components:
#   docker build --build-arg NEXT_PUBLIC_FILINGLENS_API_URL=https://api.example.com -t filinglens-web web
FROM node:22-slim AS build
WORKDIR /web
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
ARG NEXT_PUBLIC_FILINGLENS_API_URL=http://127.0.0.1:8000
ENV NEXT_PUBLIC_FILINGLENS_API_URL=$NEXT_PUBLIC_FILINGLENS_API_URL NEXT_TELEMETRY_DISABLED=1
RUN npm run build && npm prune --omit=dev

FROM node:22-slim
WORKDIR /web
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1
COPY --from=build --chown=node:node /web ./
USER node
EXPOSE 3000
CMD ["npx", "next", "start", "-p", "3000"]
