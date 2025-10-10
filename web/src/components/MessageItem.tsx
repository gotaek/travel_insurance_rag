import React from 'react';
import clsx from 'clsx';
import type { ExtendedChatMessage } from '../lib/types';
import { ComparisonTable } from './ComparisonTable';

type MessageItemProps = {
  message: ExtendedChatMessage;
};

/**
 * 단일 메시지 렌더링 컴포넌트.
 */
export function MessageItem({ message }: MessageItemProps): JSX.Element {
  const isUser = message.role === 'user';
  const isAssistant = message.role === 'assistant';

  // AI 응답에서 추가 데이터 추출
  const response = message.response;
  const evidence = response?.final_answer?.evidence || response?.draft_answer?.evidence || [];
  const caveats = response?.final_answer?.caveats || response?.draft_answer?.caveats || [];
  const comparisonTable = response?.final_answer?.comparison_table || response?.draft_answer?.comparison_table;

  return (
    <div
      className={clsx('w-full flex', isUser ? 'justify-end' : 'justify-start')}
      role="listitem"
      aria-label={isUser ? '사용자 메시지' : isAssistant ? 'AI 메시지' : '시스템 메시지'}
    >
      <div
        className={clsx(
          'max-w-[85%] rounded-2xl px-4 py-3 text-base whitespace-pre-wrap shadow-lg',
          isUser 
            ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white' 
            : 'bg-white/80 backdrop-blur-md text-gray-800 border border-gray-200/50'
        )}
      >
        <div className="whitespace-pre-wrap">{message.content}</div>
        
        {/* AI 응답에만 추가 정보 표시 */}
        {isAssistant && (evidence.length > 0 || caveats.length > 0 || comparisonTable) && (
          <div className="mt-4 space-y-3">
            {/* 근거(Evidence) 섹션 */}
            {evidence.length > 0 && (
              <div className="mt-3">
                <div className="flex items-center gap-2 text-sm text-gray-600 mb-2">
                  <span className="text-gray-500">📋</span>
                  <span>근거 {evidence.length}개</span>
                </div>
                <div className="space-y-2">
                  {evidence.map((item, index) => {
                    const itemText = typeof item === 'string' ? item : (item as any).text || String(item);
                    const itemSource = typeof item === 'object' && item !== null ? (item as any).source : null;
                    return (
                      <div key={index} className="text-sm text-gray-700 bg-gray-50/80 rounded-lg p-2 border border-gray-200/50">
                        <div>{itemText}</div>
                        {itemSource && (
                          <div className="text-gray-500 mt-1 text-xs">
                            출처: {itemSource}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* 주의사항(Caveats) 섹션 */}
            {caveats.length > 0 && (
              <div className="mt-3">
                <div className="flex items-center gap-2 text-sm text-gray-600 mb-2">
                  <span className="text-gray-500">⚠️</span>
                  <span>주의사항 {caveats.length}개</span>
                </div>
                <div className="space-y-2">
                  {caveats.map((item, index) => {
                    const itemText = typeof item === 'string' ? item : (item as any).text || String(item);
                    const itemSource = typeof item === 'object' && item !== null ? (item as any).source : null;
                    return (
                      <div key={index} className="text-sm text-gray-700 bg-yellow-50/80 rounded-lg p-2 border border-yellow-200/50">
                        <div>{itemText}</div>
                        {itemSource && (
                          <div className="text-gray-500 mt-1 text-xs">
                            출처: {itemSource}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* 비교 테이블 */}
            {comparisonTable && (
              <ComparisonTable headers={comparisonTable.headers} rows={comparisonTable.rows} />
            )}
          </div>
        )}
      </div>
    </div>
  );
}


