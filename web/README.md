# 트래블쉴드(TravelShield) 웹 UI

React + Vite + TypeScript 기반의 사용자용 챗 UI입니다. 백엔드는 FastAPI (`/rag/*`)를 사용합니다.

## 실행

```bash
# 의존성 설치 (웹 폴더)
cd web
npm i   # 또는 yarn, pnpm

# 개발 서버
npm run dev
# http://localhost:5173 접속
```

서버 URL을 바꾸려면 `.env`에 다음 값을 설정하세요:

```bash
VITE_API_BASE_URL=http://localhost:8000
```

## 빌드/프리뷰

```bash
npm run build
npm run preview
```

## 기능
- 새 세션 생성 및 새 채팅 시작
- 질문 전송 → 응답 표시
- 모바일 대응, 다크모드 기본 지원

## 주의
- 현재 응답 스트리밍은 미지원(서버 엔드포인트 확장 시 UI 점진 반영)
