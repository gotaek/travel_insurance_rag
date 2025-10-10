import React, { useEffect } from 'react';
import { Header } from './components/Header';
import { ChatWindow } from './components/ChatWindow';
import { useSessionStore } from './store/session';

/**
 * 앱 루트 컴포넌트.
 */
export default function App(): JSX.Element {
  const sessionId = useSessionStore((s) => s.sessionId);
  const restore = useSessionStore((s) => s.restore);
  const newSession = useSessionStore((s) => s.newSession);

  useEffect(() => {
    restore();
  }, [restore]);

  useEffect(() => {
    if (!sessionId) {
      void newSession();
    }
  }, [sessionId, newSession]);

  return (
    <div className="min-h-screen flex flex-col bg-white">
      {/* 배경 패턴 */}
      <div className="fixed inset-0 opacity-30" style={{
        backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23000000' fill-opacity='0.02'%3E%3Ccircle cx='30' cy='30' r='2'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`
      }}></div>
      
      <Header />
      <main className="flex-1 relative z-10">
        <ChatWindow sessionId={sessionId} />
      </main>
    </div>
  );
}


