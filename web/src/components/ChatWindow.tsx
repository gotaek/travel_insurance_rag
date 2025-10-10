import React, { useCallback, useState } from 'react';
import { MessageList } from './MessageList';
import type { ExtendedChatMessage } from '../lib/types';
import { ChatInput } from './ChatInput';
import { askQuestion } from '../lib/api';

type ChatWindowProps = {
  sessionId?: string | null;
};

/**
 * 채팅 창 컴포넌트.
 */
export function ChatWindow({ sessionId }: ChatWindowProps): JSX.Element {
  const [messages, setMessages] = useState<ExtendedChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const handleSend = useCallback(async (text: string) => {
    const userMsg: ExtendedChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: text,
      createdAt: Date.now(),
    };
    setMessages((prev) => [...prev, userMsg]);

    setIsLoading(true);
    try {
      console.log('질문 전송 중:', { question: text, session_id: sessionId });
      const res = await askQuestion({ question: text, session_id: sessionId ?? undefined, include_context: true });
      console.log('API 응답:', res);
      const content = res.final_answer?.conclusion || res.draft_answer?.conclusion || '답변을 생성할 수 없습니다.';
      const aiMsg: ExtendedChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content,
        createdAt: Date.now(),
        response: res, // 전체 응답 데이터 포함
      };
      setMessages((prev) => [...prev, aiMsg]);
    } catch (e) {
      console.error('API 호출 오류:', e);
      
      let errorMessage = '알 수 없는 오류가 발생했습니다.';
      
      if (e instanceof Error) {
        if (e.message.includes('503') || e.message.includes('overloaded')) {
          errorMessage = 'AI 서버가 일시적으로 과부하 상태입니다. 잠시 후 다시 시도해주세요.';
        } else if (e.message.includes('timeout')) {
          errorMessage = '요청 시간이 초과되었습니다. 다시 시도해주세요.';
        } else if (e.message.includes('Network Error') || e.message.includes('ERR_NETWORK')) {
          errorMessage = '네트워크 연결을 확인해주세요.';
        } else {
          errorMessage = e.message;
        }
      }
      
      const errMsg: ExtendedChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: `❌ ${errorMessage}`,
        createdAt: Date.now(),
      };
      setMessages((prev) => [...prev, errMsg]);
    } finally {
      setIsLoading(false);
    }
  }, [sessionId]);

  const hasMessages = messages.length > 0;

  return (
    <div className="mx-auto w-full max-w-3xl py-4 px-4 flex flex-col gap-4 min-h-screen">
      <div className={`flex-1 transition-all duration-500 ease-in-out ${hasMessages ? 'min-h-[40vh]' : 'flex items-center justify-center'}`}>
        <MessageList messages={messages} isLoading={isLoading} />
      </div>
      <div className={`transition-all duration-500 ease-in-out ${hasMessages ? 'mt-auto' : ''}`}>
        <ChatInput disabled={isLoading} onSend={handleSend} />
      </div>
    </div>
  );
}


