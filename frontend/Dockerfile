# Multi-stage build: compile the Vite app, then serve the static bundle
# with nginx (which also reverse-proxies /api -> the api service, so the
# frontend JS can keep calling relative /api/... paths in every environment).
#
# Build from frontend/:
#   docker build -t depression-dashboard .
#
# To point a standalone build at a separately-deployed backend instead
# (e.g. Render, matching the LASA-VOTING-SYSTEM split) bake it in at build
# time:
#   docker build --build-arg VITE_API_BASE_URL=https://your-api.onrender.com -t depression-dashboard .

FROM node:20-slim AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
ARG VITE_API_BASE_URL=
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
