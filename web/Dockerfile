# 멀티 스테이지 빌드: Node.js로 빌드 → Nginx로 서빙
FROM node:20-alpine AS build

WORKDIR /app

# 의존성 설치 (캐시 최적화)
COPY package.json package-lock.json* pnpm-lock.yaml* yarn.lock* ./
RUN npm install --no-audit --no-fund

# 소스 코드 복사 및 빌드
COPY . .

# 빌드 시 API 엔드포인트 주입
ARG VITE_API_BASE_URL=http://api:8000
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

RUN npm run build

# 런타임 스테이지: Nginx
FROM nginx:1.27-alpine

# 빌드된 파일을 Nginx로 복사
COPY --from=build /app/dist /usr/share/nginx/html

# SPA 라우팅 대응 (모든 요청을 index.html로)
RUN echo 'server { \
    listen 80; \
    server_name localhost; \
    root /usr/share/nginx/html; \
    index index.html; \
    location / { \
        try_files $uri $uri/ /index.html; \
    } \
}' > /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
