import axios from 'axios';
import type { AskResponse, CreateSessionResponse } from './types';

const API_BASE_URL = (import.meta as any).env?.VITE_API_BASE_URL || 'http://localhost:8000';

console.log('API_BASE_URL:', API_BASE_URL);

/**
 * 새 세션을 생성합니다.
 */
export async function createSession(userId?: string): Promise<CreateSessionResponse> {
  const url = `${API_BASE_URL}/rag/session/create`;
  console.log('세션 생성 요청:', { url, userId });
  
  try {
    const { data } = await axios.post(url, userId ? { user_id: userId } : undefined, {
      headers: { 'Content-Type': 'application/json' },
    });
    console.log('세션 생성 성공:', data);
    return data as CreateSessionResponse;
  } catch (error) {
    console.error('세션 생성 실패:', error);
    throw error;
  }
}

/**
 * 질문을 전송하고 응답을 반환합니다.
 */
export async function askQuestion(params: {
  question: string;
  session_id?: string;
  include_context?: boolean;
}): Promise<AskResponse> {
  const url = `${API_BASE_URL}/rag/ask`;
  console.log('질문 요청:', { url, params });
  
  const maxRetries = 3;
  const retryDelay = 2000; // 2초
  
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const { data } = await axios.post(url, params, {
        headers: { 'Content-Type': 'application/json' },
        timeout: 120_000,
      });
      console.log('질문 응답 성공:', data);
      return data as AskResponse;
    } catch (error: unknown) {
      console.error(`질문 요청 실패 (시도 ${attempt}/${maxRetries}):`, error);
      
      if (axios.isAxiosError(error)) {
        const status = error.response?.status;
        const isRetryable = status === 503 || status === 429 || (status !== undefined && status >= 500);
        
        if (isRetryable && attempt < maxRetries) {
          console.log(`${retryDelay * attempt}ms 후 재시도...`);
          await new Promise(resolve => setTimeout(resolve, retryDelay * attempt));
          continue;
        }
        
        console.error('Axios 에러 상세:', {
          message: error.message,
          status: error.response?.status,
          statusText: error.response?.statusText,
          data: error.response?.data,
        });
      }
      
      throw error;
    }
  }
  
  throw new Error('최대 재시도 횟수 초과');
}


