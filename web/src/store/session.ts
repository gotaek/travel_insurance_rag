import { create } from 'zustand';
import { createSession } from '../lib/api';

type SessionState = {
  sessionId: string | null;
  initializing: boolean;
  newSession: () => Promise<void>;
  restore: () => void;
};

/**
 * 세션 ID를 관리하는 전역 스토어.
 */
export const useSessionStore = create<SessionState>((set, get) => ({
  sessionId: null,
  initializing: false,
  restore: () => {
    const saved = localStorage.getItem('travelshield.session_id');
    if (saved) {
      console.log('세션 복원:', saved);
      set({ sessionId: saved });
    }
  },
  newSession: async () => {
    const currentState = get();
    if (currentState.initializing) {
      console.log('세션 생성 중...');
      return;
    }
    
    set({ initializing: true });
    try {
      console.log('새 세션 생성 시작');
      const res = await createSession();
      console.log('세션 생성 응답:', res);
      const id = res.session_id;
      localStorage.setItem('travelshield.session_id', String(id));
      set({ sessionId: id });
      console.log('세션 생성 완료:', id);
    } catch (error) {
      console.error('세션 생성 실패:', error);
    } finally {
      set({ initializing: false });
    }
  },
}));


