import React, { useEffect, useRef } from 'react';
import { MessageItem } from './MessageItem';
import type { ExtendedChatMessage } from '../lib/types';

type MessageListProps = {
  messages: ExtendedChatMessage[];
  isLoading?: boolean;
};

/**
 * 메시지 리스트. 자동 스크롤을 지원합니다.
 */
export function MessageList({ messages, isLoading }: MessageListProps): JSX.Element {
  const endRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages.length, isLoading]);

  return (
    <div className="flex flex-col gap-4" role="list" aria-live="polite">
      {messages.map((m) => (
        <MessageItem key={m.id} message={m} />
      ))}
      {isLoading && (
        <div className="flex items-center gap-3 text-gray-600" aria-label="응답 생성 중">
          <div className="flex space-x-1">
            <div className="w-2 h-2 bg-blue-500/60 rounded-full animate-bounce"></div>
            <div className="w-2 h-2 bg-blue-500/60 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
            <div className="w-2 h-2 bg-blue-500/60 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
          </div>
          <span className="text-base">🤖 AI가 여행자보험 정보를 분석하고 있습니다...</span>
        </div>
      )}
      <div ref={endRef} />
    </div>
  );
}


