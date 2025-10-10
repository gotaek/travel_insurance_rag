import React from 'react';

/**
 * 상단 헤더. 서비스명을 제공합니다.
 */
export function Header(): JSX.Element {
  return (
    <header 
      className="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-gray-200/50 shadow-sm"
      role="banner"
    >
      <div className="mx-auto max-w-3xl px-4 py-4 flex items-center justify-center">
        <div className="flex items-center gap-3">
          <div className="text-2xl">🛡️</div>
          <div className="font-bold text-gray-800 text-xl">트래블쉴드</div>
          <div className="text-gray-500 text-base">TravelShield</div>
        </div>
      </div>
    </header>
  );
}


