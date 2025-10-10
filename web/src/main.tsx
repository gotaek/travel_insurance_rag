import React from 'react';
import { createRoot } from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import App from './App';

// 간단 전역 스타일 (Tailwind 도입 전 임시)
import './index.css';

const queryClient = new QueryClient();

const container = document.getElementById('root');
if (!container) throw new Error('root 엘리먼트를 찾을 수 없습니다.');

const root = createRoot(container);
root.render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </React.StrictMode>
);


