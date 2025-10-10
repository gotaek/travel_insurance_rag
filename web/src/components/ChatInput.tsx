import React, { useCallback, useState } from 'react';

type ChatInputProps = {
  disabled?: boolean;
  onSend: (text: string) => void;
};

/**
 * 채팅 입력영역.
 */
export function ChatInput({ disabled, onSend }: ChatInputProps): JSX.Element {
  const [text, setText] = useState('');

  const doSend = useCallback(() => {
    const value = text.trim();
    if (!value || disabled) return;
    onSend(value);
    setText('');
  }, [text, onSend, disabled]);

  return (
    <div className="flex items-center gap-3">
      <label className="sr-only" htmlFor="chat-input">질문 입력</label>
      <div className="flex-1 relative">
        <textarea
          id="chat-input"
          value={text}
          disabled={disabled}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              if (e.metaKey || e.ctrlKey) {
                // Cmd/Ctrl+Enter: 줄바꿈 (기본 동작 허용)
                return;
              } else {
                // Enter: 전송
                e.preventDefault();
                doSend();
              }
            }
          }}
          rows={2}
          placeholder="여행자보험에 대해 질문해 보세요… (Enter 전송, Cmd/Ctrl+Enter 줄바꿈)"
          className="w-full resize-none rounded-2xl border-2 border-gray-200/50 bg-white/80 backdrop-blur-md px-4 py-3 text-base text-gray-800 placeholder-gray-500 shadow-lg focus:outline-none focus:ring-2 focus:ring-blue-300/50 focus:border-blue-400/50 transition-all duration-200 hover:bg-white/90"
        />
      </div>
      <button
        type="button"
        onClick={doSend}
        disabled={disabled || !text.trim()}
        className="rounded-2xl px-6 py-3 text-base font-semibold text-white bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-200 active:scale-95"
        aria-label="메시지 전송"
      >
        {disabled ? (
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
            전송 중
          </div>
        ) : (
          '전송'
        )}
      </button>
    </div>
  );
}


